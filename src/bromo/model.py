import torch
import torch.nn as nn
import math


# ─── Positional Encoding ─────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # create constant 'pe' matrix with sin/cos
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): How many epochs to wait before stopping
                            when validation loss is not improving.
            min_delta (float): Minimum change in the monitored quantity
                               to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Call this after each validation phase.
        """
        # First epoch call: Just record the current loss
        if self.best_loss is None:
            self.best_loss = val_loss
            return

        # Check improvement
        if val_loss < self.best_loss - self.min_delta:
            # There is an improvement
            self.best_loss = val_loss
            self.counter = 0
        else:
            # No improvement: Increase counter
            self.counter += 1

        # Check if we should stop
        if self.counter >= self.patience:
            self.early_stop = True


# ─── Transformer Model ────────────────────────────────────────────────────
class PeptidePairTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        max_len: int,
        max_charge: int,
        charge_emb_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        # token embeddings + positional
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.use_flash_attention = use_flash_attention

        if use_flash_attention:
            from flash_attn import flash_attn_func
            from flash_attn.modules.mha import FlashSelfAttention

            # Create custom encoder layers with Flash Attention
            self.flash_attn = FlashSelfAttention(causal=False, softmax_scale=None)
            encoder_layers = []
            for _ in range(num_layers):
                layer = self._create_flash_encoder_layer(
                    d_model, nhead, dim_feedforward, dropout
                )
                encoder_layers.append(layer)
            self.transformer = nn.ModuleList(encoder_layers)
        else:
            # shared Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="relu",
                batch_first=True,  # <— tell it you're using (batch, seq, d_model)
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers,
                # enable_nested_tensor=True  # this is already the default
            )

        # Add attention pooling layer
        self.attn_pooling = nn.Sequential(nn.Linear(d_model, 1), nn.Tanh())

        # charge embedding
        self.charge_emb = nn.Embedding(max_charge + 1, charge_emb_dim)

        # final classifier MLP
        total_dim = d_model * 2 + charge_emb_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # logits for {0,1}
        )

    def encode_nomask(self, x):
        # x: (batch, seq_len)
        x = self.token_emb(x)  # → (batch, seq_len, d_model)
        x = self.pos_enc(x)  # still (batch, seq_len, d_model)
        y = self.transformer(x)  # batch‑first now!
        return y.mean(dim=1)  # pooled repr

    def encode(self, x):
        mask = (x != 0).float().unsqueeze(-1)  # Create mask for padded values
        x = self.token_emb(x)  # → (batch, seq_len, d_model)
        x = self.pos_enc(x)  # still (batch, seq_len, d_model)
        if self.use_flash_attention:
            # Apply Flash Attention manually through layers
            y = x
            for layer in self.transformer:
                # Self-attention block
                residual = y
                y = layer["norm1"](y)

                # Reshape for Flash Attention and apply
                batch_size, seq_len, hidden_dim = y.shape
                y = y.view(batch_size, seq_len, hidden_dim)
                y = layer["flash_attn"](y)

                # Add residual connection and dropout
                y = residual + layer["dropout"](y)

                # FFN block
                residual = y
                y = layer["norm2"](y)
                y = residual + layer["dropout"](layer["ffn"](y))
        else:
            # Use standard transformer
            y = self.transformer(x)

        # Attention pooling instead of mean pooling
        attn_weights = self.attn_pooling(y)  # (batch, seq_len, 1)
        attn_weights = attn_weights * mask  # Zero out padding tokens
        attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-9)
        weighted_repr = (y * attn_weights).sum(dim=1)  # (batch, d_model)
        return weighted_repr

    def forward(self, a, ch_a, b, ch_b):
        rep_a = self.encode(a)  # (batch, d_model)
        rep_b = self.encode(b)
        emb_a = self.charge_emb(ch_a)  # (batch, charge_emb_dim)
        emb_b = self.charge_emb(ch_b)
        z = torch.cat([rep_a, rep_b, emb_a, emb_b], dim=1)
        return self.classifier(z)  # (batch, 2)

    def _create_flash_encoder_layer(self, d_model, nhead, dim_feedforward, dropout):
        """Create a custom encoder layer with Flash Attention"""
        from flash_attn import flash_attn_func
        from flash_attn.modules.mha import FlashSelfAttention

        return nn.ModuleDict(
            {
                "norm1": nn.LayerNorm(d_model),
                "flash_attn": FlashSelfAttention(
                    softmax_scale=None, attention_dropout=dropout
                ),
                "norm2": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model),
                    nn.Dropout(dropout),
                ),
                "dropout": nn.Dropout(dropout),
            }
        )

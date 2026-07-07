import torch
import torch.nn as nn
import math


# ─── Positional Encoding ─────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
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
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.attn_pooling = nn.Sequential(nn.Linear(d_model, 1), nn.Tanh())

        self.charge_emb = nn.Embedding(max_charge + 1, charge_emb_dim)

        total_dim = d_model * 2 + charge_emb_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def encode(self, x):
        mask = (x != 0).float().unsqueeze(-1)
        x = self.token_emb(x)
        x = self.pos_enc(x)
        y = self.transformer(x)

        attn_weights = self.attn_pooling(y)  # (batch, seq_len, 1)
        attn_weights = attn_weights * mask
        attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-9)
        return (y * attn_weights).sum(dim=1)  # (batch, d_model)

    def forward(self, a, ch_a, b, ch_b):
        rep_a = self.encode(a)
        rep_b = self.encode(b)
        emb_a = self.charge_emb(ch_a)
        emb_b = self.charge_emb(ch_b)
        z = torch.cat([rep_a, rep_b, emb_a, emb_b], dim=1)
        return self.classifier(z)

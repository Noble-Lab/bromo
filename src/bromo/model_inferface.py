#!/usr/bin/env python3
"""
Train a Transformer-based binary classifier on peptide-pair data.
Each peptide column is like "CEMEGCGTVLAHPR|3" (sequence|charge).
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # Add this import
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
import matplotlib
import os
import sys
import optuna
from pathlib import Path
from bromo.model import PeptidePairTransformer

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bromo.data import (
    PeptidePairDataset,
    collate_fn,
    get_train_test_datasets,
    filter_peptide_pairs_by_observation,
)


################################### Helpers ####################################


def plot_training_history(history, out_dir):
    # Set up the figure
    plt.figure(figsize=(15, 6))

    # Plot 1: Training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(
        range(1, len(history["train_loss"]) + 1),
        history["train_loss"],
        "b-",
        label="Training Loss",
    )
    plt.plot(
        range(1, len(history["val_loss"]) + 1),
        history["val_loss"],
        "r-",
        label="Validation Loss",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Accuracy and AUROC
    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, len(history["accuracy"]) + 1),
        history["accuracy"],
        "g-",
        label="Accuracy",
    )
    plt.plot(range(1, len(history["auroc"]) + 1), history["auroc"], "y-", label="AUROC")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Model Performance Metrics")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/training_history_plot.png", dpi=300)
    plt.close()

    print(
        f"Created training history visualization: {out_dir}/training_history_plot.png"
    )


def run_eval(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for a_seq, a_ch, b_seq, b_ch, lbl in val_loader:
            a_seq, a_ch = a_seq.to(device), a_ch.to(device)
            b_seq, b_ch = b_seq.to(device), b_ch.to(device)
            lbl = lbl.to(device)

            logits = model(a_seq, a_ch, b_seq, b_ch)
            loss = criterion(logits, lbl)
            val_loss += loss.item() * a_seq.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(lbl.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_val_loss = val_loss / len(all_labels)
    val_acc = accuracy_score(all_labels, all_preds)
    val_auroc = roc_auc_score(all_labels, all_probs)
    return avg_val_loss, val_acc, val_auroc, all_labels, all_probs, all_preds


def save_checkpoint(
    model,
    model_out_dir,
    step,
    suffix="",
    save_predictions=False,
    all_labels=None,
    all_probs=None,
    all_preds=None,
    out_dir_trainvaltest=None,
    plot_roc_fn=None,
):
    """Save model checkpoint at the given step."""
    if suffix:
        suffix = f"_{suffix}"

    state_path = f"{model_out_dir}/peptide_transformer_state_step{step}{suffix}.pth"
    full_path = f"{model_out_dir}/peptide_transformer_full_step{step}{suffix}.pth"

    torch.save(model.state_dict(), state_path)
    print(f"Saved model.state_dict() → {state_path}")

    torch.save(model, full_path)
    print(f"Saved full model → {full_path}")

    # Optionally save predictions and plot ROC if best model
    if save_predictions and all_labels is not None:
        df_pred_label = pd.DataFrame({"pred_label": all_preds, "pred_score": all_probs})
        df_pred_label.to_csv(
            f"{out_dir_trainvaltest}/pred_step{step}{suffix}.tsv", sep="\t", index=False
        )
        if plot_roc_fn is not None:
            plot_roc_fn(
                f"{out_dir_trainvaltest}/val.tsv",
                f"{out_dir_trainvaltest}/pred_step{step}{suffix}.tsv",
                model_out_dir,
            )


def shuffle_peptide_pairs(f: str, out_file: str, random_seed=42):
    # 1) load
    df = pd.read_csv(f, sep="\t")

    # 2) pick half of the negatives
    neg_idx = df[df["label"] == 0].sample(frac=0.5, random_state=random_seed).index
    # 3) pick half of the positives
    pos_idx = df[df["label"] == 1].sample(frac=0.5, random_state=random_seed).index

    # 4) swap peptide columns for both sets
    swap_cols(df, neg_idx, "peptide_a", "peptide_b")
    swap_cols(df, pos_idx, "peptide_a", "peptide_b")

    # 5) toggle labels on those same rows
    df.loc[neg_idx, "label"] = 1
    df.loc[pos_idx, "label"] = 0

    # (Optional) if your features n_pos/n_neg are also order‑dependent,
    # you can swap them too:
    swap_cols(df, neg_idx, "n_pos", "n_neg")
    swap_cols(df, pos_idx, "n_pos", "n_neg")

    # 6) save to disk
    df.to_csv(out_file, sep="\t", index=False)
    print(f"Written augmented file to {out_file}")


# helper to swap two columns in place
def swap_cols(df, idx, col1, col2):
    df.loc[idx, [col1, col2]] = df.loc[idx, [col2, col1]].values


def balance_detection(
    input_file: str, output_file: str, seed: int = 42
) -> pd.DataFrame:
    """
    Reads a tab-delimited file with a 'detection' column,
    balances the number of rows for detection=0 and detection=1 by sampling,
    shuffles the result, and writes to output_file.

    Parameters:
    - input_file:  Path to the input .txt/.tsv file.
    - output_file: Path to write the balanced file.
    - seed:        Random seed for reproducibility.

    Returns:
    - A pandas DataFrame containing the balanced data.
    """
    print("Load file:", input_file)
    # Load the data
    df = pd.read_csv(input_file, sep="\t")

    # Determine the minimum count across detection classes
    min_count = df["detection"].value_counts().min()

    print(f"Minimum count: {min_count}")

    # Sample equally from each class
    df0 = df[df["detection"] == 0].sample(n=min_count, random_state=seed)
    df1 = df[df["detection"] == 1].sample(n=min_count, random_state=seed)

    print(f"Number of 0s after sampling: {len(df0)}")
    print(f"Number of 1s after sampling: {len(df1)}")

    # Combine and shuffle
    balanced_df = (
        pd.concat([df0, df1]).sample(frac=1, random_state=seed).reset_index(drop=True)
    )

    # Save to file
    balanced_df.to_csv(output_file, sep="\t", index=False)


def plot_roc(test_file: str, pred_file: str, out_dir: str):
    # 1) Load your data
    test = pd.read_csv(test_file, sep="\t")
    pred = pd.read_csv(pred_file, sep="\t")

    # 2) Combine side by side (assumes same number of rows and same order)
    df = test.copy()
    df["pred_label"] = pred["pred_label"]
    df["score"] = pred["pred_score"]

    # 3) Define your subsets
    subsets = {
        "All": df.index,
        "Detected only": df["detection"] == 1,
        "Hybrid only": df["detection"] == 0,
    }

    # 4) Compute & plot ROC for each subset
    plt.figure(figsize=(4.5, 4.5))

    for name, mask in subsets.items():
        y_true = df.loc[mask, "label"]
        y_score = df.loc[mask, "score"]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")

    # Diagonal line for random chance
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.legend(loc="lower right", fontsize=14)
    # and to bump up tick labels:
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/roc.png", dpi=300)
    plt.close()


################################### Main ####################################
def main():
    if len(sys.argv) == 1:
        print("python rank.py [train, predict]")
        sys.exit(0)
    else:
        mode = sys.argv[1]
        if mode == "train":
            parser = argparse.ArgumentParser(
                description="Train peptide‑pair Transformer classifier"
            )
            parser.add_argument(
                "--tuning",
                action="store_true",
                help="Flag whether to do hyperparamter tuning",
            )
            parser.add_argument(
                "--tuned",
                action="store_true",
                help="Flag whether to load params from optuna tuning",
            )
            parser.add_argument(
                "--params_db_path",
                default=None,
                help="Path to the params database",
            )
            parser.add_argument(
                "--params_study_name",
                default=None,
                help="Name of the optuna study",
            )
            parser.add_argument(
                "--model_out_dir",
                default="./",
                help="Directory to save the model and results",
            )
            parser.add_argument(
                "--data_out_dir",
                required=True,
                help="Directory to save the data (train, val, test files)",
            )
            parser.add_argument(
                "--train_file",
                "-i",
                required=True,
                help="TSV file with columns: protein, peptide_pair, peptide_a, peptide_b, n_pos, n_neg, label",
            )

            parser.add_argument(
                "--val_file",
                "-v",
                required=False,
                help="TSV file with columns: protein, peptide_pair, peptide_a, peptide_b, n_pos, n_neg, label",
            )

            parser.add_argument(
                "--train_test_split_method",
                "-s",
                default="protein",
                choices=["protein", "peptide"],
                help="train/test split method: 'protein' or 'peptide'",
            )

            ## load pretrained model
            parser.add_argument(
                "--load_model",
                "-m",
                default=None,
                help="Load pretrained model from this path",
            )
            parser.add_argument(
                "--load_pretrained_datasets",
                "-pd",
                default=None,
                help="Load pretrained datasets from this path",
            )

            parser.add_argument(
                "--epochs", "-e", type=int, default=10, help="Number of training epochs"
            )
            parser.add_argument(
                "--batch-size", "-b", type=int, default=2048, help="Batch size"
            )
            parser.add_argument(
                "--weight-decay",
                type=float,
                default=0.0,
                help="Weight decay (L2 regularization strength)",
            )
            parser.add_argument(
                "--ls",
                type=float,
                default=0.0,
                help="Label smoothing (0.0 = no smoothing)",
            )

            parser.add_argument(
                "--sample",
                type=float,
                default=1.0,
                help="Randomly sample this fraction of the dataset for training",
            )

            parser.add_argument(
                "--dr",
                type=float,
                default=0.0,
                help="The percentage of samples a pair is needed to be detected to be included in the training set",
            )
            parser.add_argument(
                "--seed",
                type=int,
                default=42,
                help="Random seed for reproducibility",
            )

            parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

            parser.add_argument(
                "--cpu",
                type=float,
                default=4,
                help="The number of CPU cores to use for training",
            )

            parser.add_argument(
                "--odd_ratio", type=float, default=0.0, help="Odd ratio for the dataset"
            )

            parser.add_argument(
                "--flash",
                action="store_true",
                help="Use Flash Attention for faster training",
            )

            parser.add_argument(
                "--swap", action="store_true", help="swap peptide pair labels"
            )

            parser.add_argument("--balance", action="store_true", help="balance")

            parser.add_argument(
                "--max-len",
                type=int,
                default=30,
                help="Max peptide sequence length (pad/truncate)",
            )
            parser.add_argument(
                "--max-charge",
                type=int,
                default=5,
                help="Maximum charge state (for embedding)",
            )
            parser.add_argument(
                "--val-interval",
                type=int,
                default=100,
                help="Run validation every training steps (0 = only at end of epochs). Saves best checkpoint when validation AUC improves.",
            )
            args = parser.parse_args(sys.argv[2 : len(sys.argv)])

            torch.backends.cudnn.benchmark = True
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
            # 4.1 Read and parse
            AA = list("ACDEFGHIKLMNPQRSTVWY")
            aa2idx = {aa: i + 1 for i, aa in enumerate(AA)}
            aa2idx["X"] = len(AA) + 1  # unknown
            VOCAB_SIZE = len(AA) + 2  # +1 for pad(0), +1 for X
            train_file = args.train_file
            val_file = args.val_file

            if args.dr > 0:
                # filter peptide pairs by observation times
                filter_peptide_pairs_by_observation(
                    train_file=train_file,
                    o_ratio=args.dr,
                    out_file="filtered_peptide_pairs.tsv",
                )
                train_file = "filtered_peptide_pairs.tsv"

            if args.odd_ratio > 0:
                df = pd.read_csv(train_file, sep="\t")
                n_original = len(df)
                ## abs(log2((n_pos+0.01)/(n_neg+0.01))) > args.odd_ratio
                ## add odd_ratio to data frame
                df["odd_ratio"] = abs(
                    np.log2((df["n_pos"] + 0.01) / (df["n_neg"] + 0.01))
                )
                max_ratio = df["odd_ratio"].max()
                print(f"Max odd ratio: {max_ratio:.2f}")
                odd_ratio_threshold = min(max_ratio, args.odd_ratio)
                print(f"Odd ratio threshold: {odd_ratio_threshold:.2f}")
                df = df[
                    abs(np.log2((df["n_pos"] + 0.01) / (df["n_neg"] + 0.01)))
                    >= odd_ratio_threshold
                ]
                df.to_csv("odd_ratio_filtered_peptide_pairs.tsv", sep="\t", index=False)
                train_file = "odd_ratio_filtered_peptide_pairs.tsv"
                n_filtered = len(df)
                ## show the number of filtered rows and the ratio
                print(f"Original number of rows: {n_original}")
                print(f"Filtered number of rows: {n_filtered}")
                ratio = (n_original - n_filtered) / n_original
                print(f"Filtered ratio: {ratio:.2f}")

            out_dir_trainvaltest = args.data_out_dir

            if args.swap:
                out_file = out_dir_trainvaltest + "/swap.tsv"
                shuffle_peptide_pairs(train_file, out_file)
                train_file = out_file

            if args.balance:
                out_file = out_dir_trainvaltest + "/balance.tsv"
                balance_detection(train_file, out_file)
                train_file = out_file

            if args.sample < 1.0:
                # Randomly sample a fraction of the dataset
                df = pd.read_csv(train_file, sep="\t")
                rng = np.random.default_rng(args.seed)  # set seed for reproducibility
                proteins = df["protein"].dropna().unique()
                proteins = rng.permutation(proteins)  # ONE fixed random order

                def df_for_frac(df, proteins_ordered, frac):
                    k = int(np.ceil(frac * len(proteins_ordered)))
                    chosen = set(proteins_ordered[:k])
                    return df[df["protein"].isin(chosen)].copy(), chosen

                df_sampled, prot_sampled = df_for_frac(df, proteins, args.sample)

                df_sampled.to_csv(
                    out_dir_trainvaltest + f"/sampled_proteins_{args.sample}.tsv",
                    sep="\t",
                    index=False,
                )
                train_file = (
                    out_dir_trainvaltest + f"/sampled_proteins_{args.sample}.tsv"
                )

            if args.cpu > 0:
                # Use the specified number of CPU cores
                n_cpus = int(args.cpu)
            else:
                n_cpus = 8
            train_loader, val_loader, val_detections = get_train_test_datasets(
                train_file=train_file,
                val_file=val_file,
                max_len=args.max_len,
                batch_size=args.batch_size,
                train_test_split_method=args.train_test_split_method,
                out_dir=out_dir_trainvaltest,
                n_cpus=int(n_cpus),
                load_pretrained_datasets=args.load_pretrained_datasets,
            )
            train_size = len(train_loader.dataset)
            val_size = len(val_loader.dataset)
            print(f"Train size: {train_size}, Val size: {val_size}")
            # 4.4 Model, loss, optimizer
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if args.tuning:
                model = PeptidePairTransformer(
                    vocab_size=VOCAB_SIZE,
                    d_model=int(os.environ.get("TUNE_D_MODEL", 128)),
                    nhead=int(os.environ.get("TUNE_NHEAD", 8)),
                    num_layers=int(os.environ.get("TUNE_NUM_LAYERS", 4)),
                    dim_feedforward=int(os.environ.get("TUNE_DIM_FF", 512)),
                    max_len=args.max_len,
                    max_charge=args.max_charge,
                    charge_emb_dim=int(os.environ.get("TUNE_CHARGE_EMB_DIM", 8)),
                    hidden_dim=int(os.environ.get("TUNE_HIDDEN_DIM", 256)),
                    dropout=float(os.environ.get("TUNE_DROPOUT", 0.3)),
                    use_flash_attention=args.flash,
                ).to(device)
            elif args.tuned:
                abs_path = os.path.abspath(args.params_db_path)
                storage = f"sqlite:///{abs_path}"
                study = optuna.load_study(
                    study_name=args.params_study_name, storage=storage
                )
                params = study.best_trial.params
                print(f"Loading params from optuna study: {params}")
                model = PeptidePairTransformer(
                    vocab_size=VOCAB_SIZE,
                    d_model=params["d_model"],
                    nhead=params["n_head"],
                    num_layers=params["num_layers"],
                    dim_feedforward=params["dim_feedforward"],
                    max_len=args.max_len,
                    max_charge=args.max_charge,
                    charge_emb_dim=params["charge_emb_dim"],
                    hidden_dim=params["hidden_dim"],
                    dropout=params["dropout"],
                    use_flash_attention=args.flash,
                ).to(device)
            else:
                ## use default parameters from bo
                model = PeptidePairTransformer(
                    vocab_size=VOCAB_SIZE,
                    d_model=128,
                    nhead=8,
                    num_layers=4,
                    dim_feedforward=512,
                    max_len=args.max_len,
                    max_charge=args.max_charge,
                    charge_emb_dim=8,
                    hidden_dim=256,
                    dropout=0.3,
                    use_flash_attention=args.flash,
                ).to(device)

            if args.load_model:
                # Load pretrained model
                print(f"Loading pretrained model from {args.load_model}")
                model.load_state_dict(
                    torch.load(args.load_model, map_location=device, weights_only=True)
                )
                model.to(device)
                print("Loaded pretrained model")

            # Print model summary
            print(model)
            print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
            print(
                f"Model parameters (trainable): {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
            )
            print(
                f"Model parameters (non-trainable): {sum(p.numel() for p in model.parameters() if not p.requires_grad)}"
            )

            ## pytorch version
            print(f"Using PyTorch version: {torch.__version__}")
            # model = torch.compile(model)

            if args.tuned:
                label_smoothing = params["ls"]
            else:
                label_smoothing = args.ls
            if label_smoothing > 0:
                # Use label smoothing
                print(f"Using label smoothing: {label_smoothing}")
                criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            else:
                criterion = nn.CrossEntropyLoss()

            if args.tuned:
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=params["lr"], weight_decay=args.weight_decay
                )
            else:
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
                )

            # Add learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2
            )

            history = {
                "train_loss": [],
                "val_loss": [],
                "accuracy": [],
                "auroc": [],
                "learning_rate": [],
                "step": [],
            }

            # ---- initial evaluation ----
            best_val_auroc = 0
            best_step = 0
            val_zero_epoch = True
            if val_zero_epoch:
                avg_val_loss, val_acc, val_auroc, all_labels, all_probs, all_preds = (
                    run_eval(model, val_loader, device, criterion)
                )
                print(
                    f"Epoch {0:02d}/{args.epochs:02d} — "
                    f"val loss: {avg_val_loss:.4f} | "
                    f"accuracy: {val_acc:.4f} | "
                    f"AUROC: {val_auroc:.4f}"
                )

                best_val_auroc = val_auroc
                if val_detections is not None:
                    for group in np.unique(val_detections):
                        idx = np.where(val_detections == group)
                        y_true = np.array(all_labels)[idx]
                        y_pred = np.array(all_preds)[idx]
                        y_prob = np.array(all_probs)[idx]
                        acc = accuracy_score(y_true, y_pred)
                        auroc = roc_auc_score(y_true, y_prob)
                        print(
                            f"detection group {group}: acc={acc:.4f}, auroc={auroc:.4f}"
                        )

            # 4.5 Training loop
            global_step = 0
            best_step = 0
            for epoch in range(1, args.epochs + 1):
                model.train()
                running_loss = 0.0
                # Add learning rate to history
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"Current learning rate: {current_lr:.6f}")
                # Add tqdm progress bar
                pbar = tqdm(
                    train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True
                )
                for a_seq, a_ch, b_seq, b_ch, lbl in pbar:
                    a_seq, a_ch = a_seq.to(device), a_ch.to(device)
                    b_seq, b_ch = b_seq.to(device), b_ch.to(device)
                    lbl = lbl.to(device)

                    logits = model(a_seq, a_ch, b_seq, b_ch)
                    loss = criterion(logits, lbl)

                    optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    # scheduler.step()  # No parameters!

                    batch_loss = loss.item()
                    running_loss += loss.item() * a_seq.size(0)
                    global_step += 1

                    # Update progress bar with current batch loss and step
                    pbar.set_postfix(
                        {"batch_loss": f"{batch_loss:.4f}", "step": global_step}
                    )

                    # Run validation at specified intervals
                    if args.val_interval > 0 and global_step % args.val_interval == 0:
                        history["step"].append(global_step)
                        history["train_loss"].append(running_loss / global_step)

                        (
                            avg_val_loss,
                            val_acc,
                            val_auroc,
                            val_labels,
                            val_probs,
                            val_preds,
                        ) = run_eval(model, val_loader, device, criterion)

                        print(
                            f"Step {global_step} — "
                            f"val loss: {avg_val_loss:.4f} | "
                            f"accuracy: {val_acc:.4f} | "
                            f"AUROC: {val_auroc:.4f}"
                        )

                        history["val_loss"].append(avg_val_loss)
                        history["accuracy"].append(val_acc)
                        history["auroc"].append(val_auroc)

                        # Save best checkpoint if validation AUC improves
                        if val_auroc > best_val_auroc:
                            best_val_auroc = val_auroc
                            best_step = global_step
                            save_checkpoint(
                                model,
                                args.model_out_dir,
                                global_step,
                                suffix="best",
                                save_predictions=True,
                                all_labels=val_labels,
                                all_probs=val_probs,
                                all_preds=val_preds,
                                out_dir_trainvaltest=out_dir_trainvaltest,
                                plot_roc_fn=plot_roc,
                            )
                            # Also save with standard names for backward compatibility
                            torch.save(
                                model.state_dict(),
                                f"{args.model_out_dir}/peptide_transformer_state.pth",
                            )
                            torch.save(
                                model,
                                f"{args.model_out_dir}/peptide_transformer_full.pth",
                            )
                            print(
                                f"New best model at step {global_step} with AUROC {val_auroc:.4f}"
                            )

                        model.train()

                avg_train_loss = running_loss / train_size
                print(
                    f"Epoch {epoch:02d}/{args.epochs:02d} — train loss: {avg_train_loss:.4f} (step {global_step})"
                )
                if val_detections is not None:
                    for group in np.unique(val_detections):
                        idx = np.where(val_detections == group)
                        y_true = np.array(all_labels)[idx]
                        y_pred = np.array(all_preds)[idx]
                        y_prob = np.array(all_probs)[idx]
                        acc = accuracy_score(y_true, y_pred)
                        auroc = roc_auc_score(y_true, y_prob)
                        print(
                            f"detection group {group}: acc={acc:.4f}, auroc={auroc:.4f}"
                        )

                # Step the scheduler based on validation metrics
                old_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(avg_val_loss)
                new_lr = optimizer.param_groups[0]["lr"]

                # Check if learning rate changed
                if new_lr != old_lr:
                    print(f"Learning rate changed: {old_lr:.6f} → {new_lr:.6f}")

            ## save history
            history_df = pd.DataFrame(history)
            history_df.to_csv(f"{args.model_out_dir}/training_history.csv", index=False)
            print(
                f"Saved training history to {args.model_out_dir}/training_history.csv"
            )
            plot_training_history(history, args.model_out_dir)
            print(f"FINAL_AUROC: {best_val_auroc:.4f} (best at step {best_step})")

        elif mode == "predict":
            parser = argparse.ArgumentParser(
                description="Predict peptide-pair Transformer classifier"
            )
            parser.add_argument(
                "--model", "-m", required=True, help="Path to the trained model"
            )
            parser.add_argument(
                "--test-file",
                "-i",
                required=True,
                help="TSV file with columns: protein, peptide_pair, peptide_a, peptide_b",
            )
            parser.add_argument(
                "--max-len",
                type=int,
                default=30,
                help="Max peptide sequence length (pad/truncate)",
            )
            parser.add_argument(
                "--max-charge",
                type=int,
                default=5,
                help="Maximum charge state (for embedding)",
            )
            parser.add_argument(
                "--out-file", "-o", required=True, help="Output file for predictions"
            )
            parser.add_argument(
                "--eval_path",
                "-e",
                required=True,
                help="Path to the evaluation file",
            )

            parser.add_argument(
                "--swap", action="store_true", help="swap peptide pair labels"
            )

            parser.add_argument(
                "--balance",
                action="store_true",
                help="balance peptide pair labels",
            )

            parser.add_argument(
                "--reverse",
                "-rp",
                default=False,
                action="store_true",
                help="reverse peptide pair labels",
            )

            args = parser.parse_args(sys.argv[2 : len(sys.argv)])

            # Load the model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = torch.load(args.model, map_location=device, weights_only=False)
            model.to(device)
            model.eval()
            test_file = args.test_file

            if args.swap:
                out_file = str(Path(args.out_file).parent) + "/swap.tsv"
                shuffle_peptide_pairs(test_file, out_file)
                test_file = out_file

            if args.balance:
                out_file = str(Path(args.out_file).parent) + "/balance.tsv"
                balance_detection(test_file, out_file)
                test_file = out_file

            # Load the data
            df = pd.read_csv(test_file, sep="\t")

            if args.reverse:
                print("Reversing peptide pair labels")
                df_flipped = df.copy()
                flip_indices = df_flipped.index

                df_flipped.loc[
                    flip_indices, ["peptide_a", "peptide_b", "n_pos", "n_neg"]
                ] = df_flipped.loc[
                    flip_indices, ["peptide_b", "peptide_a", "n_neg", "n_pos"]
                ].values

                df_flipped.loc[flip_indices, "label"] = (
                    1 - df_flipped.loc[flip_indices, "label"]
                )

                df_flipped.loc[flip_indices, "win_ratio"] = (
                    1 - df_flipped.loc[flip_indices, "win_ratio"]
                )

                df_flipped.loc[flip_indices, "peptide_pair"] = (
                    df_flipped.loc[flip_indices, "peptide_a"]
                    + ":"
                    + df_flipped.loc[flip_indices, "peptide_b"]
                )
                del df
                df = df_flipped

            records = []
            for _, row in df.iterrows():
                seq_a_full = row["peptide_a"]
                seq_b_full = row["peptide_b"]
                try:
                    seq_a, ch_a = seq_a_full.split("|")
                    seq_b, ch_b = seq_b_full.split("|")
                except ValueError:
                    raise ValueError(f"Bad peptide format: {seq_a_full}, {seq_b_full}")
                # add dummy label=0 so Dataset.__getitem__ unpacks cleanly
                records.append((seq_a, int(ch_a), seq_b, int(ch_b), 0))

            print(f"Loaded {len(records)} records!")
            # 4.2 Build vocab (20 AA + unknown + pad=0)
            AA = list("ACDEFGHIKLMNPQRSTVWY")
            aa2idx = {aa: i + 1 for i, aa in enumerate(AA)}
            aa2idx["X"] = len(AA) + 1
            VOCAB_SIZE = len(AA) + 2
            # 4.3 Dataset
            full_dataset = PeptidePairDataset(records, aa2idx, args.max_len)
            test_loader = DataLoader(
                full_dataset,
                batch_size=2048,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                prefetch_factor=2,
                collate_fn=collate_fn,
            )
            # 4.4 Predict
            all_probs = []
            all_preds = []
            with torch.no_grad():
                for a_seq, a_ch, b_seq, b_ch, lbl in test_loader:
                    a_seq, a_ch = a_seq.to(device), a_ch.to(device)
                    b_seq, b_ch = b_seq.to(device), b_ch.to(device)
                    lbl = lbl.to(device)

                    logits = model(a_seq, a_ch, b_seq, b_ch)

                    # get predicted probability for class 1 and predicted class
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    preds = torch.argmax(logits, dim=1)

                    all_probs.extend(probs.cpu().tolist())
                    all_preds.extend(preds.cpu().tolist())
            # Save predictions
            ## add this to the original data frame
            df["pred_label"] = all_preds
            df["pred_score"] = all_probs
            df.to_csv(args.out_file, sep="\t", index=False)
            print(f"Saved predictions to {args.out_file}")

            ## hack, just read the same file
            plot_roc(
                args.out_file,
                args.out_file,
                args.eval_path,
            )
        else:
            print("Unknown mode:", mode)
            sys.exit(1)


if __name__ == "__main__":
    main()

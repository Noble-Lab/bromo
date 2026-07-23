#!/usr/bin/env python3
"""
Runtime benchmark for Bromo (Transformer) and XGBoost inference.

Usage:
    python -m bromo.benchmark_runtime \
        --pairs_file   path/to/test.tsv \
        --bromo_model  path/to/model.pth \
        --xgb_model    path/to/xgboost_model.json \
        --xgb_mode     dimers \
        --load_config  path/to/model_config.json \
        --out_dir      ./benchmark_results

Produces:
    - runtime_results.tsv  : raw timing data
    - runtime_plot.pdf     : wall-clock time vs number of pairs
"""

import argparse
import os
import time
import json
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from bromo.model import PeptidePairTransformer
from bromo.data import PeptidePairDataset, collate_fn
from bromo.xgboost_baseline import build_features


# ─── helpers ─────────────────────────────────────────────────────────────────


def _build_bromo_loader(df, aa2idx, max_len, batch_size=4096):
    split_a = df["peptide_a"].str.split("|", n=1, expand=True)
    split_b = df["peptide_b"].str.split("|", n=1, expand=True)
    records = list(
        zip(
            split_a[0],
            split_a[1].astype(int),
            split_b[0],
            split_b[1].astype(int),
            [0] * len(df),
        )
    )
    dataset = PeptidePairDataset(records, aa2idx, max_len)
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
        collate_fn=collate_fn,
    )


def _time_bromo(model, loader, device):
    """Returns wall-clock seconds for one full inference pass."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for a_seq, a_ch, b_seq, b_ch, _ in loader:
            a_seq, a_ch = a_seq.to(device), a_ch.to(device)
            b_seq, b_ch = b_seq.to(device), b_ch.to(device)
            model(a_seq, a_ch, b_seq, b_ch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def _time_xgb(bst, df, mode, add_terminal=False):
    """Returns wall-clock seconds for feature building + XGBoost predict."""
    import xgboost as xgb

    t0 = time.perf_counter()
    X = build_features(df, mode=mode, add_terminal=add_terminal)
    dmat = xgb.DMatrix(X)
    bst.predict(dmat)
    return time.perf_counter() - t0


# ─── main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Bromo runtime benchmark")
    parser.add_argument(
        "--pairs_file", required=True, help="Test TSV with peptide pairs"
    )
    parser.add_argument(
        "--bromo_model", default=None, help="Path to Bromo .pth checkpoint"
    )
    parser.add_argument("--xgb_model", default=None, help="Path to XGBoost .json model")
    parser.add_argument(
        "--xgb_mode", default="dimers", choices=["dimers", "both", "single"]
    )
    parser.add_argument(
        "--add_terminal",
        action="store_true",
        help="Use terminal one-hot features for XGBoost (must match training)",
    )
    parser.add_argument(
        "--load_config", default=None, help="model_config.json for Bromo arch"
    )
    parser.add_argument("--max_len", type=int, default=30)
    parser.add_argument("--max_charge", type=int, default=4)
    parser.add_argument(
        "--n_steps",
        type=int,
        default=8,
        help="Number of logarithmically spaced sizes to benchmark",
    )
    parser.add_argument(
        "--min_pairs", type=int, default=100, help="Smallest subsample size"
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Timing repeats per size (median taken)"
    )
    parser.add_argument("--out_dir", default=".", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df_full = pd.read_csv(args.pairs_file, sep="\t")
    print(f"Loaded {len(df_full):,} pairs from {args.pairs_file}")

    AA = list("ACDEFGHIKLMNPQRSTVWY")
    aa2idx = {aa: i + 1 for i, aa in enumerate(AA)}
    aa2idx["X"] = len(AA) + 1

    # ── Load models once ──────────────────────────────────────────────────────
    bromo_model = None
    if args.bromo_model:
        config_path = args.load_config or os.path.join(
            os.path.dirname(args.bromo_model), "model_config.json"
        )
        with open(config_path) as f:
            cfg = json.load(f)
        arch_keys = {
            "vocab_size",
            "d_model",
            "nhead",
            "num_layers",
            "dim_feedforward",
            "max_len",
            "max_charge",
            "charge_emb_dim",
            "hidden_dim",
            "dropout",
        }
        bromo_model = PeptidePairTransformer(
            **{k: v for k, v in cfg.items() if k in arch_keys}
        ).to(device)
        bromo_model.load_state_dict(
            torch.load(args.bromo_model, map_location=device, weights_only=True)
        )
        bromo_model.eval()
        print("Loaded Bromo model")

    xgb_model = None
    if args.xgb_model:
        import xgboost as xgb

        xgb_model = xgb.Booster()
        xgb_model.load_model(args.xgb_model)
        print("Loaded XGBoost model")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    sizes = np.unique(
        np.logspace(
            np.log10(args.min_pairs),
            np.log10(len(df_full)),
            num=args.n_steps,
        )
        .astype(int)
        .clip(args.min_pairs, len(df_full))
    ).tolist()
    print(f"Benchmarking at sizes: {sizes}")

    rows = []
    for n in sizes:
        df_sub = df_full.sample(n=n, random_state=42).reset_index(drop=True)

        if bromo_model is not None:
            loader = _build_bromo_loader(df_sub, aa2idx, args.max_len)
            times = [
                _time_bromo(bromo_model, loader, device) for _ in range(args.repeats)
            ]
            rows.append(
                {"method": "Bromo", "n_pairs": n, "time_s": float(np.median(times))}
            )
            print(f"  Bromo  n={n:>6,}  median={np.median(times):.3f}s")

        if xgb_model is not None:
            times = [
                _time_xgb(xgb_model, df_sub, args.xgb_mode, args.add_terminal)
                for _ in range(args.repeats)
            ]
            rows.append(
                {"method": "XGBoost", "n_pairs": n, "time_s": float(np.median(times))}
            )
            print(f"  XGBoost n={n:>6,}  median={np.median(times):.3f}s")

    results = pd.DataFrame(rows)
    results_path = os.path.join(args.out_dir, "runtime_results.tsv")
    results.to_csv(results_path, sep="\t", index=False)
    print(f"\nSaved results → {results_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    PALETTE = {"Bromo": "#0072B2", "XGBoost": "#E69F00"}

    fig, ax = plt.subplots(figsize=(4, 3.5))
    for method, grp in results.groupby("method"):
        grp = grp.sort_values("n_pairs")
        ax.plot(
            grp["n_pairs"],
            grp["time_s"],
            marker="o",
            markersize=5,
            linewidth=2.5,
            color=PALETTE.get(method, "#333333"),
            label=method,
        )

    ax.set_xlabel("Number of peptide pairs", fontsize=12)
    ax.set_ylabel("Inference time (s)", fontsize=12)
    ax.set_title("Runtime benchmark", fontsize=12, fontweight="bold", pad=8)
    ax.tick_params(labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.4, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(fontsize=10, frameon=False)
    for line in leg.get_lines():
        line.set_linewidth(2.5)

    fig.tight_layout(pad=0.5)
    plot_path = os.path.join(args.out_dir, "runtime_plot.pdf")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot     → {plot_path}")


if __name__ == "__main__":
    main()

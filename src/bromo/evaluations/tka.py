import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from evaluations.utils import break_ties, invert_num_pairs
import matplotlib.pyplot as plt


def eval_q1(df, topk):
    q1_dict = defaultdict(list)

    for protein, df_protein in df.groupby("protein"):
        # build universe of peptides for this protein
        peptides = np.concatenate(
            [df_protein["peptide_a"].to_numpy(), df_protein["peptide_b"].to_numpy()]
        )
        unique_peptides = np.unique(peptides)

        # basic filters
        if len(unique_peptides) < 10:
            continue

        n_detected_pairs = int(df_protein["detection"].sum())
        n_detected = invert_num_pairs(n_detected_pairs)
        if n_detected < 7:
            continue

        # build wins table
        rows = []
        for peptide in unique_peptides:
            b = df_protein[df_protein["peptide_b"] == peptide]
            a = df_protein[df_protein["peptide_a"] == peptide]
            pred_wins = int((b["pred_label"] == 0).sum() + (a["pred_label"] == 1).sum())
            true_wins = int((b["label"] == 0).sum() + (a["label"] == 1).sum())
            rows.append((peptide, pred_wins, true_wins))
        wins_df = pd.DataFrame(rows, columns=["peptide", "pred_wins", "true_wins"])

        # tie-break on the FULL list, then take topk
        sorted_by_true = break_ties(
            wins_df.sort_values(by="true_wins", ascending=False, inplace=False),
            "true_wins",
            df_protein,
        )
        sorted_by_pred = break_ties(
            wins_df.sort_values(by="pred_wins", ascending=False, inplace=False),
            "pred_wins",
            df_protein,
        )

        # take top-k sets (order within top-k doesn’t matter for Q1)
        topk_true = sorted_by_true["peptide"].head(topk).tolist()
        topk_pred = sorted_by_pred["peptide"].head(topk).tolist()

        if not topk_true or not topk_pred:
            continue

        # compute Q1 for k = 1..K, K = min(len(topk_*), topk)
        K = min(topk, len(topk_true), len(topk_pred))
        q1_vals = []
        for k in range(1, K + 1):
            set_true = set(topk_true[:k])
            set_pred = set(topk_pred[:k])
            q1_vals.append(len(set_true & set_pred) / float(k))
        q1_dict[protein] = q1_vals

    if not q1_dict:
        print("No proteins passed filters.")
        return

    return q1_dict


def _aggregate_curves(curve_dict, topk, band="std"):
    """
    curve_dict: dict[str, list[float]] mapping protein -> length-K list
    Returns x, mean, lo, hi, n_used
    """
    if not curve_dict:
        return None

    # Pad to topk with NaNs so we can nan-aggregate
    arrs = []
    for v in curve_dict.values():
        a = np.asarray(v, dtype=float)
        if a.size < topk:
            pad = np.full(topk - a.size, np.nan)
            a = np.concatenate([a, pad])
        elif a.size > topk:
            a = a[:topk]
        arrs.append(a)
    M = np.vstack(arrs)  # shape: (n_proteins, topk)

    mean = np.nanmean(M, axis=0)
    if band == "std":
        spread = np.nanstd(M, axis=0)
    elif band == "sem":
        counts = np.sum(~np.isnan(M), axis=0).astype(float)
        spread = np.nanstd(M, axis=0) / np.sqrt(np.maximum(counts, 1.0))
    else:
        raise ValueError("band must be 'std' or 'sem'")

    x = np.arange(1, topk + 1)
    return x, mean, mean - spread, mean + spread, np.sum(~np.isnan(M), axis=0)


def _wins_and_means(df_protein: pd.DataFrame):
    # pred wins: a wins if pred_label==1, b wins if pred_label==0
    a_pw = df_protein.loc[df_protein["pred_label"] == 1, "peptide_a"].value_counts()
    b_pw = df_protein.loc[df_protein["pred_label"] == 0, "peptide_b"].value_counts()
    pred_wins = a_pw.add(b_pw, fill_value=0).astype(int)

    # true wins: a wins if label==1, b wins if label==0
    a_tw = df_protein.loc[df_protein["label"] == 1, "peptide_a"].value_counts()
    b_tw = df_protein.loc[df_protein["label"] == 0, "peptide_b"].value_counts()
    true_wins = a_tw.add(b_tw, fill_value=0).astype(int)

    # mean_pred: peptide_a contributes pred_score; peptide_b contributes 1 - pred_score
    pred_contrib = pd.concat(
        [
            df_protein[["peptide_a", "pred_score"]].rename(
                columns={"peptide_a": "peptide", "pred_score": "val"}
            ),
            df_protein[["peptide_b", "pred_score"]]
            .assign(pred_score=lambda d: 1.0 - d["pred_score"])
            .rename(columns={"peptide_b": "peptide", "pred_score": "val"}),
        ],
        ignore_index=True,
    )
    mean_pred = pred_contrib.groupby("peptide", sort=False)["val"].mean()

    mean_win_ratio = None
    if "win_ratio" in df_protein.columns:
        wr_contrib = pd.concat(
            [
                df_protein[["peptide_a", "win_ratio"]].rename(
                    columns={"peptide_a": "peptide", "win_ratio": "val"}
                ),
                df_protein[["peptide_b", "win_ratio"]]
                .assign(win_ratio=lambda d: 1.0 - d["win_ratio"])
                .rename(columns={"peptide_b": "peptide", "win_ratio": "val"}),
            ],
            ignore_index=True,
        )
        mean_win_ratio = wr_contrib.groupby("peptide", sort=False)["val"].mean()

    return pred_wins, true_wins, mean_pred, mean_win_ratio


def eval_curve_list(
    dfs,
    topk,
    metric="q1",
    labels=None,
    band="std",
    alpha_band=0.2,
    ylabel=None,
    title=None,
    tick_fs=8,
    label_fs=8,
    title_fs=8,
    legend_fs=8,
    ttest_k: int = 3,
    ttest_ref_idx: int = 0,
    ttest_show: bool = True,
    ttest_text_fs: int = 8,
    ttest_loc: tuple = (0.02, 0.95),
    save_path: str = None,
    save_dpi: int = 1200,
    colors=None,
):
    if isinstance(metric, str):
        m = metric.lower()
        if m == "q1":
            metric_fn = eval_q1
            default_ylabel = "Mean TKA"
            default_title = "TKA across protein groups"
        else:
            raise ValueError("metric must be 'q1', 'q2', or a callable")
    elif callable(metric):
        metric_fn = metric
        default_ylabel = ylabel or "Metric"
        default_title = title or "Metric across protein groups"
    else:
        raise ValueError("metric must be 'q1', 'q2', or a callable")

    if labels is None:
        labels = [f"DF{i + 1}" for i in range(len(dfs))]
    if ylabel is None:
        ylabel = default_ylabel
    if title is None:
        title = default_title

    curve_dicts = []
    used_labels = []
    for df, label in zip(dfs, labels):
        cd = metric_fn(df, topk)
        curve_dicts.append(cd if cd else None)
        used_labels.append(label)

    PALETTE = [
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#CC79A7",
        "#56B4E9",
        "#D55E00",
        "#F0E442",
        "#000000",
    ]

    fig, ax = plt.subplots(figsize=(3.5, 3.5))  # single-column

    # ---- plot ----
    for idx, (cd, label) in enumerate(zip(curve_dicts, used_labels)):
        if not cd:
            continue
        agg = _aggregate_curves(cd, topk, band=band)
        if agg is None:
            continue
        x, mean, lo, hi, _ = agg
        color = colors[idx] if colors is not None else PALETTE[idx % len(PALETTE)]
        ax.plot(
            x,
            mean,
            marker="o",
            markersize=4,
            linewidth=2.5,
            color=color,
            label=label,
            clip_on=False,
        )

    # ---- styling ----
    ax.set_xlabel("k", fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)
    ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=8)
    ax.set_xticks(range(1, topk + 1))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.4, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    ax.tick_params(axis="both", labelsize=tick_fs, direction="out", width=0.8, length=3)
    leg = ax.legend(
        fontsize=legend_fs,
        frameon=False,
        loc="lower right",
        handlelength=2.0,
        handletextpad=0.6,
        markerscale=1.8,
    )
    for line in leg.get_lines():
        line.set_linewidth(2.5)

    # ---- paired t-test annotations ----
    if ttest_show:
        ref_cd = (
            curve_dicts[ttest_ref_idx]
            if 0 <= ttest_ref_idx < len(curve_dicts)
            else None
        )
        ref_label = (
            used_labels[ttest_ref_idx]
            if 0 <= ttest_ref_idx < len(used_labels)
            else "REF"
        )

        lines = []
        if ref_cd:
            for i, (cd, label) in enumerate(zip(curve_dicts, used_labels)):
                if i == ttest_ref_idx or not cd:
                    continue
                t, p, n, md = paired_ttest_at_k(ref_cd, cd, k=ttest_k)
                lines.append(
                    f"Paired t-test at k={ttest_k}:\n n={n}, Δ={md:.3g}, p={p:.3g}"
                )

        if lines:
            ax.text(
                ttest_loc[0],
                ttest_loc[1],
                "\n\n".join(lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=ttest_text_fs,
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    edgecolor="#cccccc",
                    linewidth=0.6,
                    alpha=0.9,
                ),
            )

    fig.tight_layout(pad=0.5)

    if save_path:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
    plt.show()

    return curve_dicts, used_labels


def paired_values_at_k(
    curve_dict_a: Dict[str, List[float]],
    curve_dict_b: Dict[str, List[float]],
    k: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns paired arrays (a_vals, b_vals) at rank k (1-based),
    aligned by protein intersection, dropping NaNs / missing / too-short curves.
    """
    idx = k - 1
    common = sorted(set(curve_dict_a.keys()) & set(curve_dict_b.keys()))

    a_vals, b_vals, kept = [], [], []
    for prot in common:
        va = curve_dict_a[prot]
        vb = curve_dict_b[prot]
        if va is None or vb is None:
            continue
        if len(va) <= idx or len(vb) <= idx:
            continue
        xa, xb = float(va[idx]), float(vb[idx])
        if np.isnan(xa) or np.isnan(xb):
            continue
        a_vals.append(xa)
        b_vals.append(xb)
        kept.append(prot)

    return np.asarray(a_vals, float), np.asarray(b_vals, float), kept


def paired_ttest_at_k(curve_dict_a, curve_dict_b, k: int):
    """
    Returns (t_stat, p_value, n_pairs, mean_diff) for paired t-test at k.
    mean_diff = mean(a - b)
    """
    from scipy.stats import ttest_rel

    a, b, _ = paired_values_at_k(curve_dict_a, curve_dict_b, k=k)
    n = len(a)
    if n < 2:
        return np.nan, np.nan, n, np.nan

    t, p = ttest_rel(a, b, nan_policy="omit")
    return float(t), float(p), int(n), float(np.mean(a - b))

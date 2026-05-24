import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from bromo.evaluations.utils import break_ties, invert_num_pairs
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.ticker import MaxNLocator


def _aggregate_curves(curve_dict: Dict[str, List[float]], topk: int, band: str = "std"):
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


def _build_sorted_wins(df: pd.DataFrame, topk: int, break_ties_bool: bool = True):
    """Yields (protein, sorted_by_true, sorted_by_pred) for each protein passing filters."""
    for protein, df_protein in df.groupby("protein"):
        peptides = np.concatenate(
            [df_protein["peptide_a"].to_numpy(), df_protein["peptide_b"].to_numpy()]
        )
        unique_peptides = np.unique(peptides)

        if len(unique_peptides) < 10:
            continue

        n_detected_pairs = int(df_protein["detection"].sum())
        n_detected = invert_num_pairs(n_detected_pairs)
        if n_detected < 7:
            continue

        rows = []
        for peptide in unique_peptides:
            b = df_protein[df_protein["peptide_b"] == peptide]
            a = df_protein[df_protein["peptide_a"] == peptide]
            pred_wins = int((b["pred_label"] == 0).sum() + (a["pred_label"] == 1).sum())
            true_wins = int((b["label"] == 0).sum() + (a["label"] == 1).sum())
            rows.append((peptide, pred_wins, true_wins))
        wins_df = pd.DataFrame(rows, columns=["peptide", "pred_wins", "true_wins"])

        if break_ties_bool:
            sorted_by_true = break_ties(
                wins_df.sort_values(by="true_wins", ascending=False),
                "true_wins",
                df_protein,
            )
            sorted_by_pred = break_ties(
                wins_df.sort_values(by="pred_wins", ascending=False),
                "pred_wins",
                df_protein,
            )
        else:
            sorted_by_true = wins_df.sort_values(
                by="true_wins", ascending=False, inplace=False
            )
            sorted_by_pred = wins_df.sort_values(
                by="pred_wins", ascending=False, inplace=False
            )

        yield protein, sorted_by_true, sorted_by_pred


def eval_q1_dataframes(df: pd.DataFrame, topk: int, break_ties_bool=True):
    sorted_by_trues, sorted_by_preds, protein_groups = [], [], []

    for protein, sorted_by_true, sorted_by_pred in _build_sorted_wins(
        df, topk, break_ties_bool
    ):
        sorted_by_true.rename(
            columns={"sort_columns": "sort_columns_true"}, inplace=True
        )
        sorted_by_pred.rename(
            columns={"sort_columns": "sort_columns_pred"}, inplace=True
        )
        protein_groups.append(protein)
        sorted_by_trues.append(sorted_by_true)
        sorted_by_preds.append(sorted_by_pred)

    for idx, df_true in enumerate(sorted_by_trues):
        df_true["protein"] = protein_groups[idx]
    for idx, df_pred in enumerate(sorted_by_preds):
        df_pred["protein"] = protein_groups[idx]

    return pd.concat(sorted_by_trues), pd.concat(sorted_by_preds)


def eval_q1_scores(
    df: pd.DataFrame,
    topk: int,
    true_col: str = "sort_columns_true",
    pred_cols: List[str] = ["sort_columns_pred"],
):
    """
    df:        merged dataframe with protein, peptide, and score columns
    topk:      number of top peptides to consider
    true_col:  column to use as ground truth ranking
    pred_cols: list of columns to evaluate against true_col

    Returns: {col_name: {protein: [q1@1, ..., q1@topk]}}
    """
    results = {col: {} for col in pred_cols}

    for protein, df_protein in df.groupby("protein"):
        topk_true = (
            df_protein.sort_values(by=true_col, ascending=False)["peptide"]
            .head(topk)
            .tolist()
        )
        if not topk_true:
            continue

        for col in pred_cols:
            topk_pred = (
                df_protein.sort_values(by=col, ascending=False)["peptide"]
                .head(topk)
                .tolist()
            )
            if not topk_pred:
                continue

            K = min(topk, len(topk_true), len(topk_pred))
            results[col][protein] = [
                len(set(topk_true[:k]) & set(topk_pred[:k])) / float(k)
                for k in range(1, K + 1)
            ]

    if not any(results.values()):
        print("No proteins passed filters.")
        return None

    return results


def _eval_q1_single(df: pd.DataFrame, topk: int):
    """Adapter for eval_curve_list: works on raw pair dfs, evaluates pred vs true."""
    q1_dict = {}
    for protein, sorted_by_true, sorted_by_pred in _build_sorted_wins(df, topk):
        topk_true = sorted_by_true["peptide"].head(topk).tolist()
        topk_pred = sorted_by_pred["peptide"].head(topk).tolist()
        if not topk_true or not topk_pred:
            continue
        K = min(topk, len(topk_true), len(topk_pred))
        q1_dict[protein] = [
            len(set(topk_true[:k]) & set(topk_pred[:k])) / float(k)
            for k in range(1, K + 1)
        ]
    return q1_dict if q1_dict else None


def eval_curve_list(
    dfs: List[pd.DataFrame],
    topk: int,
    metric="q1",
    labels: List[str] = None,
    band="std",
    alpha_band=0.2,
    ylabel: str = "Mean TKA",
    title: str = None,
    tick_fs=12,
    label_fs=12,
    title_fs=12,
    legend_fs=9,
    legend_loc=(0, 0.79),
    ttest_k: int = 3,
    ttest_ref_idx: int = 0,
    ttest_show: bool = True,
    ttest_text_fs: int = 9,
    ttest_loc: tuple = (0.29, 0.06),
    save_path: str = None,
    save_dpi: int = 1200,
    ylim: tuple = (0.0, 1.0),
    colors: List[str] = None,
    curve_dicts: List[Dict[str, List[float]]] = None,
    figsize: tuple = (3.5, 3.5),
):
    if curve_dicts is None:
        if isinstance(metric, str):
            m = metric.lower()
            if m == "q1":
                metric_fn = _eval_q1_single
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
    else:
        used_labels = labels

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

    fig, ax = plt.subplots(figsize=figsize)  # single-column

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
        ax.set_ylim(ylim[0], ylim[1])

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
        loc=legend_loc,
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
                if len(curve_dicts) > 2:
                    lines.append(
                        f"{ref_label} vs {label}\nn={n}, Δ={md:.3g}\np={p:.3g}"
                    )
                else:
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


def plot_protein_ranking(
    df,
    x_col="true_ranking",
    y_col="bromo ranking",
    xlabel="True Ranking",
    ylabel="Bromo Ranking",
    title=None,
    tick_fs=12,
    label_fs=12,
    title_fs=12,
    color="#0072B2",
    label_col="peptide",
    annotation_fs=9,
    save_path=None,
    save_dpi=1200,
):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    df = df.sort_values(x_col).reset_index(drop=True)  # ← sort here

    (line,) = ax.plot(
        df[x_col],
        df[y_col],
        marker="o",
        markersize=4,
        linewidth=2.5,
        color=color,
        clip_on=False,
        linestyle=":",
    )

    if label_col is not None:
        texts = []
        for _, row in df.iterrows():
            t = ax.text(
                row[x_col],
                row[y_col],
                row[label_col],
                fontsize=annotation_fs,
                color="#444444",
                fontweight="bold",
            )
            texts.append(t)

        x_dense = np.linspace(df[x_col].min(), df[x_col].max(), 300)
        y_dense = np.interp(x_dense, df[x_col].values, df[y_col].values)

        adjust_text(
            texts,
            x=x_dense,
            y=y_dense,
            ax=ax,
            expand=(2.0, 2.5),
            force_text=(0.8, 1.0),
            force_points=(1.5, 1.5),
            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
            shrinkA=0,  # no gap at the label end
            shrinkB=4,
        )

    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)
    if title:
        ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.4, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=tick_fs, direction="out", width=0.8, length=3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout(pad=0.5)

    if save_path:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
    plt.show()


def plot_tka_learning_curves(
    x,
    methods,
    xlabel="Number of peptide pairs in training set",
    ylabel="Mean TKA at k=3",
    title=None,
    tick_fs=12,
    label_fs=12,
    title_fs=12,
    legend_fs=12,
    colors=None,
    secondary_x_labels=None,  # ← new
    save_path=None,
    save_dpi=1200,
):
    PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]

    fig, ax = plt.subplots(figsize=(4, 4))

    for idx, (name, y) in enumerate(methods.items()):
        mean_tka = y.mean(axis=0)
        std_tka = y.std(axis=0)
        color = colors[idx] if colors is not None else PALETTE[idx % len(PALETTE)]

        ax.plot(
            x,
            mean_tka,
            marker="o",
            markersize=4,
            linewidth=2.5,
            label=name,
            color=color,
            clip_on=False,
        )
        ax.fill_between(
            x, mean_tka - std_tka, mean_tka + std_tka, alpha=0.2, color=color
        )

    ax.set_xlabel(xlabel, fontsize=label_fs, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=label_fs)
    if title:
        ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=8)

    # secondary bracket labels underneath x ticks
    if secondary_x_labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{int(v):,}\n[{int(s):,}]" for v, s in zip(x, secondary_x_labels)],
            fontsize=tick_fs,
            rotation=60,
        )
        ax.set_yticklabels(fontsize=tick_fs)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.4, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=tick_fs, direction="out", width=0.8, length=3)
    ax.set_ylim(0.15, 0.6)

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

    fig.tight_layout(pad=0.5)
    fig.subplots_adjust(bottom=0.2)  # ← extra room for bracket labels

    if save_path:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
    plt.show()

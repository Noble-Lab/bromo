import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import binom
import argparse
from matplotlib.ticker import MaxNLocator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--sample_values", type=int, nargs="+", required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--max_runs_majorityvoting", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    return parser.parse_args()


def binom_two_sided_pvalue(k, n, p=0.5):
    """
    Two-sided p-value used in your script:
      if k <= n/2:  P(X <= k)
      else:        P(X >= k)  via 1 - P(X <= k-1)
    """
    if n <= 0:
        return np.nan
    if k <= 0.5 * n:
        return float(binom.cdf(k, n, p))
    else:
        return float(1.0 - binom.cdf(k - 1, n, p))


def aggregate(results, true_labels):
    for i in range(len(results)):
        data = results[i]
        for j in range(len(data)):
            if data[j] != true_labels[i] and data[j] != "remove":
                data[j] = 0
            elif data[j] == true_labels[i]:
                data[j] = 1
            else:
                data[j] = "remove"
    return results


def mean_of_results(results):
    arr = np.array(results, dtype=object)
    arr[arr == "remove"] = np.nan
    arr = arr.astype(float)
    means = np.nanmean(arr, axis=0)  # ignores nan
    return means


def plot_learning_curve(
    subsample_array,
    col_means_majorityvoting,
    col_means_binom,
    xlabel="Number of runs",
    ylabel="Unchanged label proportion",
    title="human-pan",
    tick_fs=12,
    label_fs=12,
    title_fs=12,
    legend_fs=12,
    colors=None,
    save_path=None,
    save_dpi=1200,
):
    PALETTE = ["crimson", "dimgrey"]
    colors = colors or PALETTE

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.plot(
        subsample_array,
        col_means_majorityvoting,
        marker="o",
        markersize=4,
        linewidth=2.5,
        color=colors[0],
        label="Majority voting",
        clip_on=False,
    )
    ax.plot(
        subsample_array[2:],
        col_means_binom[2:],
        marker="o",
        markersize=4,
        linewidth=2.5,
        color=colors[1],
        label="Binomial",
        clip_on=False,
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

    leg = ax.legend(
        fontsize=legend_fs,
        frameon=False,
        loc="upper right",
        handlelength=2.0,
        handletextpad=0.6,
        markerscale=1.8,
    )
    for line in leg.get_lines():
        line.set_linewidth(2.5)

    fig.tight_layout(pad=0.5)

    if save_path:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
    plt.show()


def main():
    args = parse_args()

    print("reading data file")
    df = pd.read_csv(
        args.data_file,
        sep="\t",
    )
    df["peptide_pair"] = df["peptide_a"] + ":" + df["peptide_b"]

    pep = (
        df.sort_values("peptide_pair")
        .drop_duplicates("pair_key")[["peptide_pair", "n_pos", "n_neg", "label"]]
        .to_numpy()
    )

    max_runs_majorityvoting = args.max_runs_majorityvoting
    alpha = args.alpha
    subsample_array = args.sample_values

    total_results_mv = []
    total_results_binom = []
    pairs = []
    true_labels = []

    # pull once
    total_runs = int(df["runs"].max())
    subs = np.asarray(subsample_array, dtype=np.int32)
    rng = np.random.default_rng(0)

    for peptide_pair, n_pos, n_neg, label in tqdm(pep, total=len(pep)):
        print("processing peptide pair: ", peptide_pair)
        n_pos = int(n_pos)
        n_neg = int(n_neg)

        if total_runs <= 0:
            continue

        # same as your j_eff logic
        # (if total_runs is constant, this is just min(j, total_runs))
        mv_labels = [None] * len(subs)
        binom_labels = [None] * len(subs)

        # precompute the loss-range end to avoid doing it repeatedly
        loss_end = n_pos + n_neg

        for t, j in enumerate(subs):
            j_eff = int(j)
            if j_eff > total_runs:
                j_eff = total_runs
            if j_eff <= 0:
                continue

            # rep loop is range(1) in your code; keep it structurally (no-op extra loop removed)
            idx = rng.choice(total_runs, size=j_eff, replace=False)

            # EXACTLY matches wins/losses you were computing from arr[idx]
            # under the intended encoding:
            wins = int(np.count_nonzero(idx < n_pos))
            losses = int(np.count_nonzero((idx >= n_pos) & (idx < loss_end)))
            runs = wins + losses

            if runs <= 0:
                mv_label = "remove"
                binom_label = "remove"
            else:
                ratio = wins / runs

                # majority voting
                mv_label = 1 if (2 * wins >= runs) else 0

                # binomial method
                prob = binom_two_sided_pvalue(wins, runs, p=0.5)

                if runs > max_runs_majorityvoting and prob <= alpha:
                    binom_label = 1 if ratio > 0.5 else 0
                elif runs <= max_runs_majorityvoting:
                    if ratio > 0.5:
                        binom_label = 1
                    elif ratio < 0.5:
                        binom_label = 0
                    else:
                        binom_label = "remove"
                else:
                    binom_label = "remove"

            mv_labels[t] = mv_label
            binom_labels[t] = binom_label

        total_results_mv.append(mv_labels)
        total_results_binom.append(binom_labels)
        pairs.append(peptide_pair)
        true_labels.append(label)

    total_results_mv = aggregate(total_results_mv, true_labels)
    total_results_binom = aggregate(total_results_binom, true_labels)

    col_means_majorityvoting = mean_of_results(total_results_mv)
    col_means_binom = mean_of_results(total_results_binom)

    col_means_majorityvoting = [1 - i for i in col_means_majorityvoting]
    col_means_binom = [1 - i for i in col_means_binom]

    plot_learning_curve(
        subsample_array,
        col_means_majorityvoting,
        col_means_binom,
        title="human-pan",
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()

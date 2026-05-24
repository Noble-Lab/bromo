"""
Rank peptides within each protein from bromo pairwise predictions.

Reads the output of `bromo-model predict` (which must contain both forward A:B
and reverse B:A pairs) and produces a per-protein ranking of peptide forms.

For each peptide the mean pred score is computed as:
  - pred_score  when the peptide appears as peptide_a  (= P(a beats b))
  - 1-pred_score when the peptide appears as peptide_b  (= P(b beats a))
averaged across all pairs it participates in.  Because both directions are
present, this is equivalent to first averaging each canonical pair's forward
and reverse score and then aggregating per peptide.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd


def rank_peptides(predictions_path: str, output_path: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(predictions_path, sep="\t")

    # --- per-peptide mean pred score ---
    # peptide_a side: pred_score = P(a > b)
    contrib_a = (
        df[["protein", "peptide_a", "pred_score"]]
        .rename(columns={"peptide_a": "peptide", "pred_score": "score"})
    )
    # peptide_b side: 1 - pred_score = P(b > a)
    contrib_b = (
        df[["protein", "peptide_b", "pred_score"]]
        .assign(score=lambda d: 1.0 - d["pred_score"])
        [["protein", "peptide_b", "score"]]
        .rename(columns={"peptide_b": "peptide"})
    )

    mean_scores = (
        pd.concat([contrib_a, contrib_b], ignore_index=True)
        .groupby(["protein", "peptide"], sort=False)["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "mean_pred_score"})
    )

    # --- per-peptide win count ---
    # peptide_a wins when pred_label == 1; peptide_b wins when pred_label == 0
    wins_a = (
        df[df["pred_label"] == 1]
        .groupby(["protein", "peptide_a"])
        .size()
        .reset_index(name="wins")
        .rename(columns={"peptide_a": "peptide"})
    )
    wins_b = (
        df[df["pred_label"] == 0]
        .groupby(["protein", "peptide_b"])
        .size()
        .reset_index(name="wins")
        .rename(columns={"peptide_b": "peptide"})
    )
    wins = (
        pd.concat([wins_a, wins_b], ignore_index=True)
        .groupby(["protein", "peptide"], sort=False)["wins"]
        .sum()
        .reset_index()
    )

    # --- merge, rank, output ---
    result = mean_scores.merge(wins, on=["protein", "peptide"], how="left")
    result["wins"] = result["wins"].fillna(0).astype(int)
    result = result.sort_values(
        ["protein", "mean_pred_score"], ascending=[True, False]
    )
    result["rank"] = result.groupby("protein").cumcount() + 1

    fh = open(output_path, "w") if output_path else sys.stdout
    try:
        result.to_csv(fh, sep="\t", index=False)
    finally:
        if output_path:
            fh.close()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bromo-rank",
        description=(
            "Rank peptides within each protein from bromo pairwise predictions.\n"
            "The predictions file must contain both forward (A:B) and reverse (B:A) pairs."
        ),
    )
    parser.add_argument("-i", required=True, metavar="<file>",
                        help="Predictions TSV from bromo-model predict")
    parser.add_argument("-o", metavar="<file>",
                        help="Output TSV file (default: stdout)")
    args = parser.parse_args()

    rank_peptides(args.i, args.o)


if __name__ == "__main__":
    main()

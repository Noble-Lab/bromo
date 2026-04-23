import pandas as pd
from scipy.stats import binom
import numpy as np
import argparse


def reverse_pairs(df, reverse_fraction):
    if reverse_fraction < 1:
        n_to_flip = int(np.floor(len(df) * reverse_fraction))
    else:
        n_to_flip = len(df)
    flip_indices = np.random.choice(df.index, size=n_to_flip, replace=False)

    df_flipped = df.copy()

    df_flipped.loc[flip_indices, ["peptide_a", "peptide_b", "n_pos", "n_neg"]] = (
        df_flipped.loc[
            flip_indices, ["peptide_b", "peptide_a", "n_neg", "n_pos"]
        ].values
    )
    df_flipped.loc[flip_indices, "label"] = 1 - df_flipped.loc[flip_indices, "label"]
    df_flipped.loc[flip_indices, "win_ratio"] = (
        1 - df_flipped.loc[flip_indices, "win_ratio"]
    )

    df_flipped.loc[flip_indices, "peptide_pair"] = (
        df_flipped.loc[flip_indices, "peptide_b"]
        + ":"
        + df_flipped.loc[flip_indices, "peptide_a"]
    )

    return df_flipped.loc[flip_indices]


def cleanup(df):
    ### removing pairs mapped to multiple proteins
    df = df.drop(index=df[df["protein"].str.contains(";")].index)

    ### removing pairs where peptide_a == peptide_b
    df = df.drop(index=df[df["peptide_a"] == df["peptide_b"]].index)

    ### removing duplicate forward/reverse pairs. only 1 pair key per pair!!
    a = df["peptide_a"].astype("string")
    b = df["peptide_b"].astype("string")
    pair_key = np.where(a < b, a + ":" + b, b + ":" + a)
    df["pair_key"] = pair_key
    df = df.drop(index=df[df["pair_key"].duplicated()].index)

    return df


def assign_labels():
    parser = argparse.ArgumentParser(
        description="Assign labels to peptide pairs for training/validation"
    )

    parser.add_argument(
        "--input_file",
        "-i",
        required=True,
        help="TSV file with columns: protein, peptide_pair, peptide_a, peptide_b, n_pos, n_neg, label",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        required=True,
        help="Directory to save the labeled peptide pairs",
    )
    parser.add_argument(
        "--max_runs_majorityvoting",
        "-m",
        default=4,
        help="Maximum number of runs to label pair using majority voting. Higher runs than this will be labeled using binomial model",
    )
    parser.add_argument(
        "--reverse_fraction",
        "-r",
        default=0,
        help="Percentage of peptide pairs where reverse pairs are also added to the dataset",
    )

    args = parser.parse_args()

    print("Assigning labels to peptide pairs for training/validation")
    print("Input file: ", args.input_file)
    print("Output directory: ", args.output_dir)
    print(
        "Maximum number of runs to label pair using majority voting as compared to binomial model: ",
        args.max_runs_majorityvoting,
    )

    df = pd.read_csv(args.input_file, sep="\t")

    df["runs"] = df["n_pos"] + df["n_neg"]

    df["win_ratio"] = df["n_pos"] / df["runs"]

    mask = df["n_pos"] <= 0.5 * df["runs"]

    cdf_case1 = binom.cdf(df["n_pos"], df["runs"], 0.5)
    cdf_case2 = 1 - binom.cdf(df["n_pos"] - 1, df["runs"], 0.5)
    df["prob"] = np.where(mask, cdf_case1, cdf_case2)

    def map_value(prob, ratio, runs):
        if runs == 0:
            return "Remove"
        elif runs > int(args.max_runs_majorityvoting) and prob <= 0.05:
            return 1 if ratio > 0.5 else 0
        elif runs <= int(args.max_runs_majorityvoting):
            if ratio > 0.5:
                return 1
            elif ratio < 0.5:
                return 0
            else:
                # Handle ratio == 0.5 case here
                return "Remove"  # or whatever you want for ratio == 0.5
        else:
            return "Remove"

    df["label"] = df.apply(
        lambda row: map_value(row["prob"], row["win_ratio"], row["runs"]), axis=1
    )

    df_labels = df[df["label"] != "Remove"]

    df_labels = cleanup(df_labels)

    if float(args.reverse_fraction) > 0:
        df_reverse = reverse_pairs(df_labels, float(args.reverse_fraction))
        df_labels = pd.concat([df_labels, df_reverse])
        df_labels.sort_values(by="protein", inplace=True)

    df_labels.to_csv(
        args.output_dir + "consensus_label_corrected.tsv", sep="\t", index=False
    )

    print("Labels assigned to peptide pairs for training/validation")
    print("Number of peptide pairs: ", len(df_labels))
    print("Number of peptide pairs removed: ", len(df[df["label"] == "Remove"]))
    print(
        "Number of peptide pairs labeled as 1: ",
        len(df_labels[df_labels["label"] == 1]),
    )
    print(
        "Number of peptide pairs labeled as 0: ",
        len(df_labels[df_labels["label"] == 0]),
    )


if __name__ == "__main__":
    assign_labels()

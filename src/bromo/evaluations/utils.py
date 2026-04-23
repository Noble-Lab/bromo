import pandas as pd
import numpy as np


def prepare_dfs(bromo_preds_path, xgboost_preds_path):
    bromo_preds = pd.read_csv(bromo_preds_path, sep="\t")
    xgboost_preds = pd.read_csv(xgboost_preds_path, sep="\t")

    bromo_preds["peptide_pair"] = (
        bromo_preds["peptide_a"] + ":" + bromo_preds["peptide_b"]
    )
    xgboost_preds["peptide_pair"] = (
        xgboost_preds["peptide_a"] + ":" + xgboost_preds["peptide_b"]
    )
    bromo_preds.set_index("peptide_pair", inplace=True)
    xgboost_preds.set_index("peptide_pair", inplace=True)
    xgboost_preds.rename(
        columns={"xgboost_pred_prob": "pred_score", "xgboost_pred_label": "pred_label"},
        inplace=True,
    )
    xgboost_preds = xgboost_preds.loc[bromo_preds.index]
    bromo_preds.reset_index(inplace=True)
    xgboost_preds.reset_index(inplace=True)
    return bromo_preds, xgboost_preds


def avg_fwdrev_score(df):
    # Work on a copy so we don't mutate the original df
    d = df.copy()

    # Row position within each pair_key group (0, 1, ...)
    d["_pos"] = d.groupby("pair_key").cumcount()

    # Pivot so each pair_key has columns for pred_score at pos 0 and pos 1
    wide = d.pivot(index="pair_key", columns="_pos", values="pred_score").rename(
        columns={0: "pred0", 1: "pred1"}
    )

    # Average forward + reversed score
    wide["pred_score"] = (wide["pred0"] + (1 - wide["pred1"])) / 2
    wide["pred_label"] = (wide["pred_score"] >= 0.5).astype(int)

    # Get one representative row per pair_key (the "first" row, like your code)
    first_rows = (
        d[d["_pos"] == 0]
        .drop(columns=["pred_score", "_pos"])  # we'll replace pred_score
        .set_index("pair_key")
    )

    # Join averaged score + label back
    out = (
        first_rows.drop(columns=["pred_score", "pred_label"], errors="ignore")
        .join(wide[["pred_score", "pred_label"]], how="inner")
        .reset_index()
    )

    return out


def break_ties(df, columns, df_protein):
    df = df.copy()

    if columns == "true_wins":
        # per-peptide mean of win_ratio / (1 - win_ratio) across appearances
        peptide_to_mean = {}
        for peptide in df["peptide"]:
            a = df_protein.loc[
                df_protein["peptide_a"] == peptide, "win_ratio"
            ].to_numpy()
            b = (
                1.0
                - df_protein.loc[
                    df_protein["peptide_b"] == peptide, "win_ratio"
                ].to_numpy()
            )
            vals = np.concatenate([a, b])
            peptide_to_mean[peptide] = float(np.mean(vals)) if len(vals) else 0.0

        df["mean_win_ratio"] = df["peptide"].map(peptide_to_mean)
        # break ties primarily by true_wins, secondarily by mean_win_ratio
        df["sort_columns"] = df["true_wins"] + df["mean_win_ratio"]
        return df.sort_values(by="sort_columns", ascending=False)

    elif columns == "pred_wins":
        peptide_to_mean = {}
        for peptide in df["peptide"]:
            a = df_protein.loc[
                df_protein["peptide_a"] == peptide, "pred_score"
            ].to_numpy()
            b = (
                1.0
                - df_protein.loc[
                    df_protein["peptide_b"] == peptide, "pred_score"
                ].to_numpy()
            )
            vals = np.concatenate([a, b])
            peptide_to_mean[peptide] = float(np.mean(vals)) if len(vals) else 0.0

        df["mean_pred"] = df["peptide"].map(peptide_to_mean)
        df["sort_columns"] = df["pred_wins"] + df["mean_pred"]
        return df.sort_values(by="sort_columns", ascending=False)


# Convert the list of tuples to a 3-column DataFrame
def convert_to_dataframe(data_list):
    """
    Convert list of tuples with peptide and stats to DataFrame
    """
    peptides = []
    pred_wins = []
    true_wins = []

    for peptide, stats in data_list:
        peptides.append(peptide)
        pred_wins.append(stats["pred_wins"])
        true_wins.append(stats["true_wins"])

    df = pd.DataFrame(
        {"peptide": peptides, "pred_wins": pred_wins, "true_wins": true_wins}
    )

    return df


def invert_num_pairs(m: int) -> int:
    # m = n*(n-1)/2  =>  n = (1 + sqrt(1 + 8m))/2
    return int(round((1.0 + np.sqrt(1.0 + 8.0 * float(m))) / 2.0))

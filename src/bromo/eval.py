import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse
from evaluations.utils import (
    avg_fwdrev_score,
    break_ties,
    convert_to_dataframe,
    invert_num_pairs,
    prepare_dfs,
)
from evaluations.tka import eval_curve_list
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bromo_preds_path", type=str, required=True)
    parser.add_argument("--xgboost_preds_path", type=str, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--labels", type=list, required=True)
    return parser.parse_args()


def eval_bromo_xgboost(args):
    args = parse_args()
    bromo_df, xgboost_df = prepare_dfs(args.bromo_preds_path, args.xgboost_preds_path)
    bromo_uniquepair = avg_fwdrev_score(bromo_df)
    xgboost_uniquepair = avg_fwdrev_score(xgboost_df)
    bromo_uniquepair.to_csv(
        os.path.join(
            os.path.dirname(args.bromo_preds_path), "bromo_uniquepair_preds.csv"
        )
    )
    xgboost_uniquepair.to_csv(
        os.path.join(
            os.path.dirname(args.xgboost_preds_path), "xgboost_uniquepair_preds.csv"
        )
    )
    curve_dicts_q1 = eval_curve_list(
        [bromo_uniquepair, xgboost_uniquepair],
        topk=args.topk,
        metric="q1",
        labels=["Bromo", "XGBoost"],
    )
    return curve_dicts_q1


def main():
    args = parse_args()
    curve_dicts_q1, used_labels_q1 = eval_bromo_xgboost(args)
    tka_results = pd.DataFrame(curve_dicts_q1).transpose()
    tka_results.columns = used_labels_q1
    tka_results.to_csv(
        os.path.join(os.path.dirname(args.bromo_preds_path), "tka_results.csv")
    )
    return tka_results


if __name__ == "__main__":
    main()

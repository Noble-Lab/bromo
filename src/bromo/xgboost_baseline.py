import numpy as np
import xgboost as xgb
import argparse
import pandas as pd
import sys
from sklearn.metrics import roc_auc_score
import itertools
from scipy.sparse import csr_matrix, hstack

#################################################################### Helpers #########################################################################################################


def unique_characters(strings):
    # Use a set to collect unique characters
    unique_chars = set()

    for string in strings:
        # Add each character of the string to the set
        unique_chars.update(string)

    unique_chars_string = "".join(
        sorted(unique_chars)
    )  # Sorting for better readability (optional)

    return unique_chars, unique_chars_string


def _kmer_index(k=2, alphabet="ACDEFGHIKLMNPQRSTUVWY"):
    vocab = ["".join(p) for p in itertools.product(alphabet, repeat=k)]
    return vocab, {kmer: i for i, kmer in enumerate(vocab)}


def _series_to_kmer_csr(series, k, alphabet):
    # series contains raw "SEQ|charge|..." strings → strip to sequence
    seqs = series.str.split("|", n=1, regex=False).str[0].fillna("")
    vocab, idx = _kmer_index(k=k, alphabet=alphabet)
    n = len(seqs)
    m = len(vocab)
    # Build CSR pieces
    indptr = [0]
    indices = []
    data = []
    for s in seqs:
        s = s.upper()
        if len(s) >= k:
            counts = {}
            for i in range(len(s) - k + 1):
                kmer = s[i : i + k]
                j = idx.get(kmer)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
            # append row
            if counts:
                cols, vals = zip(*sorted(counts.items()))
                indices.extend(cols)
                data.extend(vals)
        indptr.append(len(indices))
    X = csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(n, m),
        dtype=np.float32,
    )
    return X, vocab


def build_features(df, mode="dimers", alphabet="ACDEFGHIKLMNPQRSTUVWY"):
    if mode == "dimers":
        k = 2
        Xa, vocab_a = _series_to_kmer_csr(df["peptide_a"], k, alphabet)
        Xb, vocab_b = _series_to_kmer_csr(df["peptide_b"], k, alphabet)
    elif mode == "both":
        Xa_2, vocab_a_2 = _series_to_kmer_csr(df["peptide_a"], 2, alphabet)
        Xb_2, vocab_b_2 = _series_to_kmer_csr(df["peptide_b"], 2, alphabet)
        Xa_1, vocab_a_1 = _series_to_kmer_csr(df["peptide_a"], 1, alphabet)
        Xb_1, vocab_b_1 = _series_to_kmer_csr(df["peptide_b"], 1, alphabet)
        Xa = hstack([Xa_2, Xa_1], format="csr", dtype=np.float32)
        Xb = hstack([Xb_2, Xb_1], format="csr", dtype=np.float32)
        vocab_a = vocab_a_1 + vocab_a_2
        vocab_b = vocab_b_1 + vocab_b_2
    else:
        k = 1
        Xa, vocab_a = _series_to_kmer_csr(df["peptide_a"], k, alphabet)
        Xb, vocab_b = _series_to_kmer_csr(df["peptide_b"], k, alphabet)

    assert vocab_a == vocab_b  # same ordering

    # charges as 2 dense columns (float32)
    charge_a = (
        df["peptide_a"]
        .str.split("|", n=2, regex=False)
        .str[1]
        .astype(np.int16)
        .to_numpy()
    )
    charge_b = (
        df["peptide_b"]
        .str.split("|", n=2, regex=False)
        .str[1]
        .astype(np.int16)
        .to_numpy()
    )
    Xcharge = csr_matrix(np.column_stack([charge_a, charge_b]).astype(np.float32))

    # [Xa | Xb | charges]
    X = hstack([Xa, Xb, Xcharge], format="csr", dtype=np.float32)
    return X  # return CSR directly


def protein_subset(df, group_col="protein", frac=0.10, random_state=42):
    """
    Randomly keep ~frac of entire proteins (all their rows).
    """
    # Unique protein IDs
    proteins = df[group_col].unique()

    # Number to keep
    n_keep = max(1, int(np.ceil(frac * len(proteins))))

    # Sample protein IDs
    keep_proteins = pd.Series(proteins).sample(
        n=n_keep, random_state=random_state, replace=False
    )

    # Keep only those proteins
    return df[df[group_col].isin(keep_proteins)].reset_index(drop=True)


#################################################################### CLI #########################################################################################################


def build_argparser(mode):
    parser = argparse.ArgumentParser(
        description="XGBoost baseline for peptide pair classification"
    )
    if mode == "train":
        parser.add_argument("--train_file", required=True, help="Path to train file")
        parser.add_argument("--val_file", required=True, help="Path to val file")
        parser.add_argument(
            "--output_dir",
            required=True,
            help="Path to output directory to store model",
        )
        parser.add_argument(
            "--mode",
            required=True,
            choices=["dimers", "both", "single"],
            help="Use dimers, both, or single",
        )
    elif mode == "predict":
        parser.add_argument("--test_file", required=True, help="Path to test file")
        parser.add_argument(
            "--xgboost_model_file", required=True, help="Path to xgboost model file"
        )
        parser.add_argument(
            "--svm_model_file", default=None, help="Path to svm model file"
        )
        parser.add_argument(
            "--output_dir",
            required=True,
            help="Path to output directory to store predictions",
        )
        parser.add_argument(
            "--mode",
            required=True,
            choices=["dimers", "both", "single"],
            help="Use dimers, both, or single",
        )

    else:
        raise ValueError(f"Invalid mode: {mode}")
    return parser


def main():
    args = build_argparser(sys.argv[1]).parse_args(sys.argv[2:])

    ################################# Train #################################

    if sys.argv[1] == "train":
        train = pd.read_csv(
            args.train_file,
            sep="\t",
        )

        print("loaded train")

        all_peptides = pd.concat([train["peptide_a"], train["peptide_b"]])
        unique_chars, unique_chars_string = unique_characters(
            set([i[0] for i in all_peptides.str.split("|")])
        )

        print("unique_chars_string", unique_chars_string)

        X_train = build_features(train, mode=args.mode)
        print(X_train.shape)

        # train_svm = protein_subset(train, group_col="protein", frac=0.10)

        # X_train_svm = build_features(train_svm, dimers=args.dimers)
        # print(X_train_svm.shape)

        print("built features")

        ################## XGBoost ##################
        dtrain = xgb.DMatrix(X_train, label=train.label, nthread=8)

        params = {
            "objective": "binary:logistic",  # binary classification
            "eval_metric": "auc",  # loss function to be optimized
        }

        bst = xgb.train(params, dtrain, num_boost_round=100)

        print("trained model XGBoost")

        if args.mode == "dimers":
            bst.save_model(args.output_dir + "/xgboost_model_dimers.json")
        if args.mode == "both":
            bst.save_model(args.output_dir + "/xgboost_model_monomerdimer.json")
        else:
            bst.save_model(args.output_dir + "/xgboost_model.json")

        print("saved model XGBoost")

        ################## SVM ##################
        # base_linear = LinearSVC(C=1.0, dual="auto", random_state=42)

        # clf = make_pipeline(
        #     StandardScaler(with_mean=False),
        #     CalibratedClassifierCV(estimator=base_linear, method="sigmoid", cv=5),
        # )
        # clf.fit(X_train, train.label)
        # print("trained model SVM (linear + calibrated)")

        # if args.dimers:
        #     svm_model_path = os.path.join(args.output_dir, "svm_model_dimers.joblib")
        #     dump(clf, svm_model_path)
        # else:
        #     svm_model_path = os.path.join(args.output_dir, "svm_model.joblib")
        #     dump(clf, svm_model_path)

        # print("saved model svm")

    ################################# Predict #################################

    elif sys.argv[1] == "predict":
        test = pd.read_csv(
            args.test_file,
            sep="\t",
        )

        X_test = build_features(test, mode=args.mode)
        print("built features")

        ################## XGBoost ##################
        bst = xgb.Booster()
        bst.load_model(args.xgboost_model_file)

        print("loaded XGBoost model")

        dtest = xgb.DMatrix(X_test, label=test.label, nthread=8)

        y_pred_prob_xgboost = bst.predict(dtest)  # Predicted probabilities

        test["xgboost_pred_prob"] = y_pred_prob_xgboost
        test["xgboost_pred_label"] = (y_pred_prob_xgboost > 0.5).astype(int)

        print("predicted probabilities XGBoost")

        roc_auc_xgboost = roc_auc_score(test.label, y_pred_prob_xgboost)
        print(f"ROC AUC XGBoost: {roc_auc_xgboost:.4f}")
        print("calculated roc auc xgboost")

        if args.mode == "dimers":
            test.to_csv(
                args.output_dir + "/xgboost_predictions_dimers.tsv",
                sep="\t",
                index=False,
            )
        elif args.mode == "both":
            test.to_csv(
                args.output_dir + "/xgboost_predictions_monomerdimer.tsv",
                sep="\t",
                index=False,
            )
        else:
            test.to_csv(
                args.output_dir + "/xgboost_predictions.tsv", sep="\t", index=False
            )

        print("saved predictions xgboost")

        # ################## SVM ##################

        # if args.svm_model_file:
        #     clf = load(args.svm_model_file)
        #     y_pred_prob_svm = clf.predict_proba(X_test)[:, 1]
        #     test["svm_pred_prob"] = y_pred_prob_svm
        #     test["svm_pred_label"] = (y_pred_prob_svm > 0.5).astype(int)
        #     roc_auc_svm = roc_auc_score(test.label, y_pred_prob_svm)
        #     print(f"ROC AUC SVM: {roc_auc_svm:.4f}")
        #     print("calculated roc auc svm")

        #     if args.dimers:
        #         test.to_csv(
        #             args.output_dir + "/svm_predictions_dimers.tsv",
        #             sep="\t",
        #             index=False,
        #         )
        #     else:
        #         test.to_csv(
        #             args.output_dir + "/svm_predictions.tsv", sep="\t", index=False
        #         )

        #     print("saved predictions svm")


if __name__ == "__main__":
    main()

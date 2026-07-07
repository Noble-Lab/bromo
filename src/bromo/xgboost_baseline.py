import numpy as np
import xgboost as xgb
import argparse
import pandas as pd
import sys
import json
import os
from sklearn.metrics import roc_auc_score
import itertools
from scipy.sparse import csr_matrix, hstack
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

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


def _terminal_onehot_csr(series, alphabet="ACDEFGHIKLMNPQRSTUVWY"):
    """One-hot encode the N-terminal and C-terminal residues of each peptide.
    Output width = 2 * len(alphabet): first half is N-terminal, second is C-terminal.
    """
    aa_idx = {aa: i for i, aa in enumerate(alphabet)}
    A = len(alphabet)
    seqs = series.str.split("|", n=1, regex=False).str[0].fillna("")
    n = len(seqs)
    rows, cols, data = [], [], []
    for r, seq in enumerate(seqs):
        seq = seq.upper()
        if seq:
            n_term = aa_idx.get(seq[0])
            c_term = aa_idx.get(seq[-1])
            if n_term is not None:
                rows.append(r)
                cols.append(n_term)
                data.append(1.0)
            if c_term is not None:
                rows.append(r)
                cols.append(A + c_term)
                data.append(1.0)
    return csr_matrix(
        (np.array(data, dtype=np.float32), (np.array(rows), np.array(cols))),
        shape=(n, 2 * A),
        dtype=np.float32,
    )


def build_features(
    df, mode="dimers", alphabet="ACDEFGHIKLMNPQRSTUVWY", add_terminal=False
):
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

    parts = [Xa, Xb, Xcharge]

    if add_terminal:
        Xterm_a = _terminal_onehot_csr(df["peptide_a"], alphabet)
        Xterm_b = _terminal_onehot_csr(df["peptide_b"], alphabet)
        parts += [Xterm_a, Xterm_b]

    return hstack(parts, format="csr", dtype=np.float32)


def tune_xgboost(dtrain, dval, n_trials=50, seed=42):
    """
    Optuna hyperparameter search for XGBoost. Returns the best params dict
    and best num_boost_round found across n_trials, optimising validation AUC.
    """

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "seed": seed,
            "verbosity": 0,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
        }
        num_boost_round = trial.suggest_int("num_boost_round", 50, 500)
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, "val")],
            verbose_eval=False,
        )
        preds = bst.predict(dval)
        return roc_auc_score(dval.get_label(), preds)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best_num_boost_round = best.pop("num_boost_round")
    best_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "seed": seed,
        **best,
    }
    print(f"Best val AUC: {study.best_value:.4f}")
    print(f"Best params: {best_params}")
    print(f"Best num_boost_round: {best_num_boost_round}")
    return best_params, best_num_boost_round


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
        parser.add_argument(
            "--add_terminal",
            action="store_true",
            help="Add N- and C-terminal one-hot features",
        )
        parser.add_argument(
            "--tune",
            action="store_true",
            help="Run Optuna hyperparameter search before training",
        )
        parser.add_argument(
            "--n_trials",
            type=int,
            default=50,
            help="Number of Optuna trials (default: 50)",
        )
        parser.add_argument(
            "--load_config",
            default=None,
            help="Path to an xgboost_config.json from a previous tuning run; skips Optuna and reuses saved hyperparameters",
        )
    elif mode == "predict":
        parser.add_argument("--test_file", required=True, help="Path to test file")
        parser.add_argument(
            "--xgboost_model_file", required=True, help="Path to xgboost model file"
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
        parser.add_argument(
            "--add_terminal",
            action="store_true",
            help="Add N- and C-terminal one-hot features (must match training)",
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

        X_train = build_features(train, mode=args.mode, add_terminal=args.add_terminal)
        print(X_train.shape)

        print("built features")

        ################## XGBoost ##################
        dtrain = xgb.DMatrix(X_train, label=train.label, nthread=8)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "seed": 42,
        }
        num_boost_round = 100

        if args.load_config:
            with open(args.load_config) as f:
                cfg = json.load(f)
            params = cfg["params"]
            num_boost_round = cfg["num_boost_round"]
            print(f"Loaded XGBoost config from {args.load_config}")
            print(f"  params={params}, num_boost_round={num_boost_round}")
        elif args.tune:
            val = pd.read_csv(args.val_file, sep="\t")
            X_val = build_features(val, mode=args.mode, add_terminal=args.add_terminal)
            dval = xgb.DMatrix(X_val, label=val.label, nthread=8)
            print(f"Running Optuna hyperparameter search ({args.n_trials} trials)...")
            params, num_boost_round = tune_xgboost(
                dtrain, dval, n_trials=args.n_trials, seed=42
            )
            config_path = os.path.join(args.output_dir, "xgboost_config.json")
            with open(config_path, "w") as f:
                json.dump(
                    {"params": params, "num_boost_round": num_boost_round}, f, indent=2
                )
            print(f"Saved XGBoost config → {config_path}")

        bst = xgb.train(params, dtrain, num_boost_round=num_boost_round)

        print("trained model XGBoost")

        if args.mode == "dimers":
            bst.save_model(args.output_dir + "/xgboost_model_dimers.json")
        if args.mode == "both":
            bst.save_model(args.output_dir + "/xgboost_model_monomerdimer.json")
        else:
            bst.save_model(args.output_dir + "/xgboost_model.json")

        print("saved model XGBoost")
    ################################# Predict #################################

    elif sys.argv[1] == "predict":
        test = pd.read_csv(
            args.test_file,
            sep="\t",
        )

        X_test = build_features(test, mode=args.mode, add_terminal=args.add_terminal)
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


if __name__ == "__main__":
    main()

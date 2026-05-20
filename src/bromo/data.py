from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd
import numpy as np


##### Helpers #####
def collate_fn(batch):
    """
    Pads all seq_a to the same length, and all seq_b to the same length.
    Returns: (a_pad, ch_a, b_pad, ch_b, labels)
    """
    seqs_a, ch_a, seqs_b, ch_b, labels = zip(*batch)
    a_pad = pad_sequence(seqs_a, batch_first=True, padding_value=0)
    b_pad = pad_sequence(seqs_b, batch_first=True, padding_value=0)
    return a_pad, torch.stack(ch_a), b_pad, torch.stack(ch_b), torch.stack(labels)


def split_by_protein(
    input_path: str,
    train_path: str,
    val_path: str,
    test_path: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    random_seed: int = 42,
    sep: str = "\t",
):
    """
    Splits a peptide‑pair TSV into train/validation/test by protein.

    Parameters
    ----------
    input_path : str
        Path to the original TSV (must have a 'protein' column).
    train_path : str
        Where to write the 90% training subset.
    test_path : str
        Where to write the 10% test subset.
    train_ratio : float, default=0.9
        Fraction of unique proteins to assign to train.
    random_seed : int, default=42
        Seed for reproducible shuffling.
    sep : str, default='\t'
        Column separator for read/write.
    """
    # 1. Load
    df = pd.read_csv(input_path, sep=sep)

    # 2. Shuffle proteins
    proteins = df["protein"].unique()
    rng = np.random.default_rng(random_seed)
    rng.shuffle(proteins)

    # 3. Split list of proteins
    n_train = int(len(proteins) * train_ratio)
    n_val = int(len(proteins) * val_ratio)
    train_proteins = set(proteins[:n_train])
    val_proteins = set(proteins[n_train : n_train + n_val])
    test_proteins = set(proteins[n_train + n_val :])

    # 4. Filter
    df_train = df[df["protein"].isin(train_proteins)]
    df_val = df[df["protein"].isin(val_proteins)]
    df_test = df[df["protein"].isin(test_proteins)]

    # 5. Write out
    df_train.to_csv(train_path, sep=sep, index=False)
    df_val.to_csv(val_path, sep=sep, index=False)
    df_test.to_csv(test_path, sep=sep, index=False)

    print(
        f"Proteins: total={len(proteins)}, train={len(train_proteins)}, val={len(val_proteins)}, test={len(test_proteins)}"
    )
    print(f"Records: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")


def split_by_pretrained_dataset(
    input_path: str,
    train_path: str,
    val_path: str,
    test_path: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    random_seed: int = 42,
    sep: str = "\t",
    pretrained_datasets: str = None,
):

    df = pd.read_csv(input_path, sep=sep)

    old_train = pd.read_csv(pretrained_datasets + "/train.tsv", sep="\t")
    old_val = pd.read_csv(pretrained_datasets + "/val.tsv", sep="\t")
    old_test = pd.read_csv(pretrained_datasets + "/test.tsv", sep="\t")

    old_train_proteins = old_train.protein.unique()
    old_val_proteins = old_val.protein.unique()
    old_test_proteins = old_test.protein.unique()
    old_proteins_all = np.concatenate(
        [old_train_proteins, old_val_proteins, old_test_proteins]
    )

    df_train = df[df["protein"].isin(old_train_proteins)]
    df_val = df[df["protein"].isin(old_val_proteins)]
    df_test = df[df["protein"].isin(old_test_proteins)]
    df_new_proteins = df[np.invert(df["protein"].isin(old_proteins_all))]

    n_train = int(len(df_new_proteins.protein.unique()) * train_ratio)
    n_val = int(len(df_new_proteins.protein.unique()) * val_ratio)
    train_proteins = set(df_new_proteins.protein.unique()[:n_train])
    val_proteins = set(df_new_proteins.protein.unique()[n_train : n_train + n_val])
    test_proteins = set(df_new_proteins.protein.unique()[n_train + n_val :])

    df_train_new = df[df["protein"].isin(train_proteins)]
    df_val_new = df[df["protein"].isin(val_proteins)]
    df_test_new = df[df["protein"].isin(test_proteins)]

    df_train = pd.concat([df_train, df_train_new])
    df_val = pd.concat([df_val, df_val_new])
    df_test = pd.concat([df_test, df_test_new])

    df_train.to_csv(train_path, sep=sep, index=False)
    df_val.to_csv(val_path, sep=sep, index=False)
    df_test.to_csv(test_path, sep=sep, index=False)

    print(
        f"Proteins: total={len(df.protein.unique())}, train={len(df_train.protein.unique())}, val={len(df_val.protein.unique())}, test={len(df_test.protein.unique())}"
    )
    print(f"Records: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")


def get_records_labels(train_file: str, max_len: int):
    df = pd.read_csv(train_file, sep="\t")
    ## the number of samples for each class (label) in the input file
    print(f"Number of samples in the input file: {len(df)}")
    print(f"Number of positive samples: {len(df[df['label'] == 1])}")
    print(f"Number of negative samples: {len(df[df['label'] == 0])}")
    records = []
    for _, row in df.iterrows():
        seq_a_full = row["peptide_a"]  # e.g. "CEMEGCGTVLAHPR|3"
        seq_b_full = row["peptide_b"]
        try:
            seq_a, ch_a = seq_a_full.split("|")
            seq_b, ch_b = seq_b_full.split("|")
        except ValueError:
            raise ValueError(f"Bad peptide format: {seq_a_full}, {seq_b_full}")
        label = int(row["label"])
        records.append((seq_a, int(ch_a), seq_b, int(ch_b), label))
    print(f"Loaded {len(records)} records!")
    # 4.2 Build vocab (20 AA + unknown + pad=0)
    AA = list("ACDEFGHIKLMNPQRSTVWY")
    aa2idx = {aa: i + 1 for i, aa in enumerate(AA)}
    aa2idx["X"] = len(AA) + 1  # unknown

    # 4.3 Dataset & Train/Test split
    full_dataset = PeptidePairDataset(records, aa2idx, max_len)
    return full_dataset


def get_train_test_datasets(
    train_file,
    val_file=None,
    max_len=30,
    batch_size=2048,
    train_test_split_method="protein",
    out_dir="./",
    n_cpus=8,
    load_pretrained_datasets=None,
):
    val_detections = None
    if train_test_split_method == "protein" and val_file is None:
        # Split by protein

        new_train_file = f"{out_dir}/train.tsv"
        new_val_file = f"{out_dir}/val.tsv"
        new_test_file = f"{out_dir}/test.tsv"

        if load_pretrained_datasets is not None:
            split_by_pretrained_dataset(
                input_path=train_file,
                train_path=new_train_file,
                val_path=new_val_file,
                test_path=new_test_file,
                train_ratio=0.6,
                val_ratio=0.2,
                random_seed=42,
                sep="\t",
                pretrained_datasets=load_pretrained_datasets,
            )
        else:
            split_by_protein(
                input_path=train_file,
                train_path=new_train_file,
                val_path=new_val_file,
                test_path=new_test_file,
                train_ratio=0.6,
                val_ratio=0.2,
                random_seed=42,
                sep="\t",
            )
        train_dataset = get_records_labels(new_train_file, max_len)
        val_dataset = get_records_labels(new_val_file, max_len)
        # check if detection is present in the file
        val_df = pd.read_csv(new_val_file, sep="\t")
        if "detection" in val_df.columns:
            val_detections = val_df["detection"].values
        else:
            val_detections = None
    elif val_file is not None:
        # show test file
        print(f"Using val file: {val_file}")
        # using val for validation
        train_dataset = get_records_labels(train_file, max_len)
        val_dataset = get_records_labels(val_file, max_len)
        # check if detection is present in the file
        val_df = pd.read_csv(val_file, sep="\t")
        if "detection" in val_df.columns:
            val_detections = val_df["detection"].values
        else:
            val_detections = None
    else:
        full_dataset = get_records_labels(train_file, max_len)
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpus,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )
    # don't shuffle val data
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpus,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, val_detections


def filter_peptide_pairs_by_observation(train_file, o_ratio=0.1, out_file=None):
    df = pd.read_csv(train_file, sep="\t")
    # protein	peptide_pair	peptide_a	peptide_b	n_pos	n_neg	label
    if o_ratio > 1:
        # n_pos + n_neg >= o_ratio
        print(f"Filtering peptide pairs by observation times: {o_ratio}")
        n_rows = len(df)
        df = df[(df["n_pos"] + df["n_neg"]) >= o_ratio]
        # save to file
        df.to_csv(out_file, sep="\t", index=False)
        print(f"Filtered {n_rows - len(df)} rows, saved to {out_file}")
    else:
        # TODO
        print("Not implemented yet")
        raise NotImplementedError("o_ratio < 1 not implemented yet")


class PeptidePairDataset(Dataset):
    """
    records: list of tuples (seq_a:str, charge_a:int, seq_b:str, charge_b:int, label:int)
    aa2idx: mapping of amino acids to integer indices
    max_len: maximum sequence length (pad/truncate to this)
    """

    def __init__(self, records, aa2idx, max_len):
        self.records = records
        self.aa2idx = aa2idx
        self.max_len = max_len

    def encode_seq(self, seq):
        # map each AA to idx, truncate, then return torch.LongTensor
        idxs = [self.aa2idx.get(aa, self.aa2idx["X"]) for aa in seq[: self.max_len]]
        return torch.tensor(idxs, dtype=torch.long)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        seq_a, ch_a, seq_b, ch_b, label = self.records[idx]
        return (
            self.encode_seq(seq_a),
            torch.tensor(ch_a, dtype=torch.long),
            self.encode_seq(seq_b),
            torch.tensor(ch_b, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

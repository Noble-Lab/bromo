# bromo

bromo is a deep learning model for prioritizing peptides in targeted mass spectrometry experiments. It ranks peptide precursors within a protein by their predicted MS2 response in DIA experiments using only amino acid sequence and charge state as input.

Unlike existing tools that rely on detectability as a proxy or small synthetic training sets, bromo is trained on millions of peptide pairs derived from large-scale, publicly available DIA data and consistently outperforms existing sequence-based methods across diverse, independent datasets. bromo can also be fine-tuned on experiment-specific data to account for differences in sample preparation, sample matrix, and instrument platform.

The associated preprint *"Prioritizing peptides for targeted mass spectrometry experiments using deep learning"* is available [here](#).

All scripts and notebooks needed to reproduce all results and figures are located in the bromo-manuscript repo, available at: 
[https://github.com/Noble-Lab/bromo-manuscript]. This repo also contains the exact commands used to generate all intermediate datasets used to produce figures in the paper. 

bromo is open source under an [Apache 2.0 license](LICENSE).

<p align="center">
  <img src="docs/images/bromo_icon.png" width="300">
</p>

---

## Repository structure

```text
bromo/
├── src/bromo/
│   ├── generate_pairs.py               # In-silico peptide pair generation from a FASTA file
│   ├── rank_peptides.py                # Rank peptides within each protein from pairwise predictions
│   ├── assign_labels.py                # Assign pair labels via binomial model or majority voting
│   ├── data.py                         # Data loading, train/val/test splitting, PeptidePair dataclass
│   ├── model.py                        # Transformer model architecture
│   ├── model_interface.py              # Training, fine-tuning, and inference
│   ├── eval.py                         # Evaluation metrics on predicted vs. ground-truth labels
│   ├── xgboost_baseline.py             # XGBoost baseline model
│   ├── subsample_runs_experiment.py    # Label stability experiment as a function of number of runs
│   └── evaluations/
│       ├── tka.py                      # Top-k accuracy curve calculation and plotting
│       └── utils.py                    # Evaluation utilities (forward/reverse pair averaging)
```

---

## Installation

Requires Python ≥ 3.9.

We recommend using conda to manage dependencies for bromo. Create a new conda enviornment with:
```bash
conda create --name bromo_env python=3.10
```

This will create an environment called bromo_env with Python 3.10 installed. Activate it by running:
```bash
conda activate bromo_env
```

Finally, you can install bromo and all of its dependencies with:
```bash
pip install bromo
```

For development (changes take effect immediately):

```bash
git clone https://github.com/Noble-Lab/bromo.git
cd bromo
pip install -e .
```

---

## Inference workflow

Use this workflow when you have a FASTA file and a pretrained bromo model and want to score all possible peptide pairs without any DIA data. A script titled `run_inference.sh` is included where given paths to files/checkpoints, can be used to run inference end-to-end for a fasta file with protein sequences. Descriptions of individual calls to bromo that is needed to run inference end to end is provided below:

### Step 1 — Generate pairs

Generate all in-silico peptide pairs from a protein FASTA file using `bromo-pairs`:

```bash
bromo-pairs \
  -db /path/to/proteome.fasta \
  -min_pep_length 7 \
  -max_pep_length 30 \
  -min_pep_charge 2 \
  -max_pep_charge 4 \
  -o pairs.tsv
```

| Argument | Description |
|---|---|
| `-db` | Input FASTA file |
| `-o` | Output TSV file (default: stdout) |
| `-enzyme` | Enzyme ID (default: 1 = Trypsin; see below for all options) |
| `-miss_c` | Max missed cleavages (default: 0) |
| `-min_pep_length` | Min peptide length (default: 7) |
| `-max_pep_length` | Max peptide length (default: 35) |
| `-min_pep_charge` | Min precursor charge (default: 2) |
| `-max_pep_charge` | Max precursor charge (default: 4) |
| `--i2l` | Convert isoleucine (I) to leucine (L) before digestion |
| `--no-clip-m` | Disable N-terminal methionine clipping (enabled by default) |

Enzyme IDs: `0` non-enzyme · `1` Trypsin · `2` Trypsin (no P rule) · `3` Arg-C · `4` Arg-C (no P rule) · `5` Arg-N · `6` Glu-C · `7` Lys-C

The output is a TSV with columns `protein`, `peptide_pair`, `peptide_a`, `peptide_b` — the same format expected by `bromo-model predict`.

### Step 2 — Inference

Run inference using pretrained model on pairs.tsv using `bromo-model predict`:

```bash
bromo-model predict \
    --model /path/to/ModelCheckpoints/Pretrained/bromo/human-astral/peptide_transformer_full_step25700_best.pth \
    --test-file pairs.tsv \
    --max-len 30 \
    --max-charge 4 \
    --out-file ./pairs_predictions.tsv 
```

The table with argument descriptions are described in the `bromo-model predict` portion of the training workflow section


### Step 3 — Rank peptides

Aggregate the pairwise scores into a per-protein peptide ranking using `bromo-rank`:

```bash
bromo-rank \
  -i pairs_predictions.tsv \
  -o peptide_rankings.tsv
```

| Argument | Description |
|---|---|
| `-i` | Predictions TSV from `bromo-model predict` |
| `-o` | Output TSV file (default: stdout) |

Output columns:

| Column | Description |
|---|---|
| `protein` | Protein identifier |
| `peptide` | Peptide form (`SEQUENCE\|CHARGE`) |
| `mean_pred_score` | Mean P(this peptide beats any opponent) across all pairs — higher = more detectable |
| `wins` | Number of pairwise comparisons won |
| `rank` | Rank within protein (1 = most detectable) |

---

## Training workflow

The full bromo training pipeline has four steps:

1. **Generate pairs** — use the [carafe-rank (v2.1.0+)](https://github.com/Noble-Lab/Carafe) Java tool to produce `consensus_label.txt` from a DIA-NN report
2. **Assign labels** — clean pairs and assign binary labels
3. **Model interface: train** — train a bromo model on labeled pair data
4. **Model interface: predict** — run inference on new data

---

### Step 1 — Generate pairs

Use the carafe-rank Java tool to generate peptide pairs from a DIA-NN `report.tsv`:

```bash
java -cp carafe-2.1.0.jar main.java.rank.RankLabelGenerator\
  -i report.tsv \
  -db proteome.fasta \
  -o output_dir/ \
  -min_pep_length 7 \
  -max_pep_length 30 \
  -min_pep_charge 2 \
  -max_pep_charge 4 \
  -n 2
```

This produces `consensus_label.txt`, which is the input to the next step.

---

### Step 2 — Assign labels

Assigns binary labels to each peptide pair using either majority voting (low-run pairs) or a binomial model (high-run pairs). Optionally augments the dataset with reversed pairs.

```bash
bromo-assign-labels \
  --input_file path/to/consensus_label.txt \
  --max_runs_majorityvoting 4 \
  --reverse_fraction 1 \
  --output_dir path/to/output/
```

| Argument | Description |
|---|---|
| `--input_file` | `consensus_label.txt` from carafe-rank |
| `--max_runs_majorityvoting` | Pairs seen in ≤ this many runs are labeled by majority vote; higher-run pairs use the binomial model |
| `--reverse_fraction` | Fraction of pairs to also add in reverse orientation as data augmentation (0 = none, 1 = all) |
| `--output_dir` | Directory to write `consensus_label_corrected.tsv` |

---

### Step 3 — Train

Trains the peptide-pair Transformer from scratch.

```bash
bromo-model train \
  --train_file path/to/consensus_label_corrected.tsv \
  --model_out_dir path/to/checkpoints/ \
  --data_out_dir path/to/data_splits/ \
  --epochs 15 \
  --batch-size 4096 \
  --lr 0.0001 \
  --max-len 30 \
  --max-charge 4 \
  --cpu 4 \
  --weight-decay 1e-2
```

| Argument | Description |
|---|---|
| `--train_file` | Output of `bromo-assign-labels` |
| `--val_file` | Optional held-out validation file (same format as `--train_file`). If `val_file` is provided, `train_file` will not be split into train/val/test sets |
| `--model_out_dir` | Directory to save model checkpoints |
| `--data_out_dir` | Directory to save train/val/test split files |
| `--epochs` | Number of training epochs |
| `--batch-size` | Batch size |
| `--lr` | Learning rate |
| `--max-len` | Maximum peptide sequence length |
| `--max-charge` | Maximum peptide charge state |
| `--cpu` | Number of CPU workers for data loading |
| `--weight-decay` | L2 regularization strength |

---

### Step 4 — Predict

Runs inference with a trained model and outputs per-pair scores.

```bash
bromo-model predict \
  --model path/to/checkpoint.pth \
  --test-file path/to/consensus_label_corrected.tsv \
  --max-len 30 \
  --max-charge 4 \
  --out-file path/to/predictions.tsv 
```

| Argument | Description |
|---|---|
| `--model` | Path to trained `.pth` checkpoint |
| `--test-file` | Input file for inference (same format as training data) |
| `--max_len` | Max peptide length to initialize model |
| `--max-charge` | Max peptide charge to initialize model |
| `--out-file` | Path to write predictions TSV |
---

### Fine-tuning

Fine-tunes a pretrained model on experiment-specific data (e.g. different instrument, sample matrix, or organism). 

```bash
bromo-model train \
  --train_file path/to/finetune_train.tsv \
  --val_file path/to/finetune_val.tsv \
  --load_model path/to/pretrained_checkpoint.pth \
  --model_out_dir path/to/finetuned_checkpoints/ \
  --data_out_dir path/to/data_splits/ \
  --epochs 5 \
  --batch-size 4096 \
  --lr 0.00001 \
  --max-len 30 \
  --max-charge 4 \
  --cpu 4 \
  --weight-decay 1e-2
```

The key difference from training from scratch is `--load_model` (path to the pretrained checkpoint) and a lower learning rate.

---
<!-- 
### Evaluation

Computes evaluation metrics (AUROC, TKA curves) comparing bromo predictions against ground-truth labels.

```bash
bromo-eval \
  --bromo_preds_path path/to/predictions.tsv
```

--- -->

## Input format

The training and inference TSV files produced by carafe-rank and `bromo-assign-labels` have the following columns:

| Column | Description |
|---|---|
| `protein` | Protein identifier |
| `peptide_pair` | `peptide_a:peptide_b` |
| `peptide_a` | First peptide sequence |
| `peptide_b` | Second peptide sequence |
| `n_pos` | Number of runs where peptide_a was detected over peptide_b |
| `n_neg` | Number of runs where peptide_b was detected over peptide_a |
| `label` | Binary label: 1 if peptide_a is preferred, 0 otherwise |

---

## Citation

If you use bromo in your research, please cite:

> [Preprint citation coming soon]

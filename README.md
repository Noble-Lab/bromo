# bromo

bromo is a deep learning model for prioritizing peptides in targeted mass spectrometry experiments. It ranks peptide precursors within a protein by their predicted MS2 response in DIA experiments using only amino acid sequence and charge state as input.

Unlike existing tools that rely on detectability as a proxy or small synthetic training sets, bromo is trained on millions of peptide pairs derived from large-scale, publicly available DIA data and consistently outperforms existing sequence-based methods across diverse, independent datasets. bromo can also be fine-tuned on experiment-specific data to account for differences in sample preparation, sample matrix, and instrument platform.

The associated preprint *"Prioritizing peptides for targeted mass spectrometry experiments using deep learning"* is available [here](#).

All scripts and notebooks needed to reproduce all results and figures are located in the bromo-manuscript repo, available at: 
[https://github.com/Noble-Lab/bromo-manuscript]

bromo is open source under an [Apache 2.0 license](LICENSE).

<p align="center">
  <img src="docs/images/bromo_icon.png" width="300">
</p>

---

## Repository structure

```text
bromo/
├── src/bromo/
│   ├── assign_labels.py                # Assign pair labels via binomial model or majority voting
│   ├── data.py                         # Data loading, train/val/test splitting, PeptidePair dataclass
│   ├── model.py                        # Transformer model architecture
│   ├── model_interface.py                        # Training, fine-tuning, and inference
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
conda activate cascadia_env
```

Finally, you can install Cascadia and all of its dependencies with:
```bash
pip install bromo
```

For development (changes take effect immediately):

```bash
git clone https://github.com/your-org/bromo.git
cd bromo
pip install -e .
```

---

## Workflow

The full bromo pipeline has four steps:

1. **Generate pairs** — use the [carafe-rank](https://github.com/your-org/carafe-rank) Java tool to produce `consensus_label.txt` from a DIA-NN report
2. **Assign labels** — clean pairs and assign binary labels
3. **Train** — train a Transformer model on labeled pairs
4. **Predict** — run inference on new data

---

### Step 1 — Generate pairs

Use the carafe-rank Java tool to generate peptide pairs from a DIA-NN `report.tsv`:

```bash
java -jar carafe-rank.jar \
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
bromo-train train \
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
| `--val_file` | Optional held-out validation file (same format as `--train_file`) |
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
bromo-train predict \
  --model path/to/checkpoint.pth \
  --test-file path/to/consensus_label_corrected.tsv \
  --max-len 30 \
  --max-charge 4 \
  --out-file path/to/predictions.tsv \
  --eval_path path/to/eval_plots/
```

| Argument | Description |
|---|---|
| `--model` | Path to trained `.pth` checkpoint |
| `--test-file` | Input file for inference (same format as training data) |
| `--out-file` | Path to write predictions TSV |
| `--eval_path` | Directory to save ROC curve and evaluation plots |

---

### Fine-tuning

Fine-tunes a pretrained model on experiment-specific data (e.g. different instrument, sample matrix, or organism).

```bash
bromo-train train \
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

### Evaluation

Computes evaluation metrics (AUROC, TKA curves) comparing bromo predictions against ground-truth labels.

```bash
bromo-eval \
  --bromo_preds_path path/to/predictions.tsv
```

---

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

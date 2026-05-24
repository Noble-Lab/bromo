#! /bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bromoenv

## Step 1: Generate peptide pairs
bromo-pairs \
  -db path/to/proteome.fasta \
  -min_pep_length 7 \
  -max_pep_length 30 \
  -min_pep_charge 2 \
  -max_pep_charge 4 \
  -o pairs.tsv

## Step 2: Predict peptide pairs
bromo-model predict \
    --model path/to/model.pth \
    --test-file pairs.tsv \
    --max-len 30 \
    --max-charge 4 \
    --out-file ./pairs_predictions.tsv \
    --eval_path .

## Step 3: Rank peptides
bromo-rank -i pairs_predictions.tsv -o peptide_rankings.tsv
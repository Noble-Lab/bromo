### Activate conda env

```
# qlogin to a gpu node before run the following script, such as n001 or n002.
conda activate /net/noble/vol1/home/bwen1/tools/anaconda3/envs/peptdeep_latest
```

### Training data generation

```shell
# -i DIA-NN report.tsv file
# -db protein database used for DIA-NN analysis
# -o the folder for output files
# -min_pep_length the minimum length of a peptide to be included in the dataset
# -max_pep_length the maximum length of a peptide to be included in the dataset
# -min_pep_charge the minimum charge state of a peptide to be included in the dataset
# -max_pep_charge the maximum charge state of a peptide to be included in the dataset
# -n the minimum number of detected peptides for a protein to consider

ASTRAL_YEAST_REPORT_PATH=/net/noble/vol3/user/ssontha2/yash_noble_rotation/reclone/2024_ssontha2_ms-targeted-dl/data/07.09.2025_generatealldata/astral_yeast/report.tsv

SAVE_PATH_ASTRAL_YEAST=/net/noble/vol3/user/ssontha2/yash_noble_rotation/reclone/2024_ssontha2_ms-targeted-dl/data/2026.02.17_restartfromscratch/generate_pairs/astral_yeast/

java -jar /net/noble/vol1/home/bwen1/project/2024_ssontha2_ms-targeted-dl/code/carafe-rank-1.0.0/carafe-rank-1.0.0.jar \
  -i "$ASTRAL_YEAST_REPORT_PATH" \
  -db "$YEAST_DB_PATH" \
  -o "$SAVE_PATH_ASTRAL_YEAST" \
  -min_pep_length 7 \
  -max_pep_length 30 \
  -min_pep_charge 2 \
  -max_pep_charge 4 \
  -n 2

```

The output file **consensus_label.txt** is the penultimate file needed for model training. An example is available at:

```
/net/noble/vol1/home/bwen1/project/2024_ssontha2_ms-targeted-dl/code/carafe-rank-1.0.0/example/

-rw-r--r-- 1 bwen1 noblelab   3853678 Jul  6 07:16 UP000002311_559292.fasta
-rw-r--r-- 1 bwen1 noblelab 344502199 Jul  6 07:16 report.tsv
-rw-r--r-- 1 bwen1 noblelab       174 Jul  6 07:19 run.sh

```

### Assign labels to data

After all pairs are generated, the **assign_labels.py** script is used to assign labels using either the binomial or majority voting scheme.

```shell
# -input_file consensus_label.txt file from java script
# -max_runs_majorityvoting max number of runs for a pair to use majority voting as opposed to binomial model
# -reverse_percentage fraction of how many pairs to reverse as data augmentation
# -output_dir where to save the novel dataset file
python -u assign_labels.py \
  --input_file "/net/noble/vol3/user/ssontha2/yash_noble_rotation/reclone/2024_ssontha2_ms-targeted-dl/data/2026.02.17_restartfromscratch/preprocessing/generate_pairs/astral_yeast/consensus_label.txt" \
  --max_runs_majorityvoting 4 \
  --reverse_fraction 1 \
  --output_dir "/net/noble/vol3/user/ssontha2/yash_noble_rotation/reclone/2024_ssontha2_ms-targeted-dl/data/2026.02.17_restartfromscratch/preprocessing/assign_labels/astral_yeast/"

```

### Train

```shell
# The inputs to both -i and -t are in the same format with the consensus_label.txt file generated using the above java tool.

# -model_out_dir directory to save the model checkpoints
# -data_out_dir directory to save train/val/test data if only train is specified
# -train_file output of assign_labels.py
# -test_file optional test dataset file in same format as train_file
# -epochs the number of epochs to train for
# -batch_size batch size for one forward pass
# -lr learning rate
# -max-len maximum peptide length
# -max-charge maximum peptide charge
# -cpu number of cpu's
# -weight-decay weight decay strength


python train.py train \
    --model_out_dir /net/noble/vol3/user/ssontha2/yash_noble_rotation/reclone/2024_ssontha2_ms-targeted-dl/results/yash_new/2026.02.17_restartfromscratch/train \
    --data_out_dir /net/noble/vol3/user/ssontha2/yash_noble_rotation/reclone/2024_ssontha2_ms-targeted-dl/data/2026.02.17_restartfromscratch/training/astral_human \
    --train_file /net/noble/vol3/user/ssontha2/yash_noble_rotation/reclone/2024_ssontha2_ms-targeted-dl/data/2026.02.17_restartfromscratch/preprocessing/assign_labels/astral_human/consensus_label_corrected.tsv \
    --epochs 15 \
    --batch-size 4096 \
    --lr 0.0001 \
    --max-len 30 \
    --max-charge 4 \
    --cpu 4 \
    --weight-decay 1e-2
```

### Predict

```shell
# -model peptide_transformer_full.pth for is the trained model file generated using the above training command line.
# -test-file file to use for testing 
# -max-len maximum peptide length 
# -max-charge maximum peptide charge
# -out-file file directory and name to save predictions
# -eval_path directory to save roc curve plot
python train.py predict \
    --model /net/noble/vol3/user/ssontha2/yash_noble_rotation/reclone/2024_ssontha2_ms-targeted-dl/results/yash_new/2026.02.17_restartfromscratch/train/peptide_transformer_full_step9700_best.pth \
    --test-file /net/noble/vol3/user/ssontha2/yash_noble_rotation/reclone/2024_ssontha2_ms-targeted-dl/data/2026.02.17_restartfromscratch/preprocessing/assign_labels/astral_yeast/consensus_label_corrected.tsv \
    --max-len 30 \
    --max-charge 4 \
    --out-file /net/noble/vol3/user/ssontha2/yash_noble_rotation/reclone/2024_ssontha2_ms-targeted-dl/data/2026.02.17_restartfromscratch/make_predictions/astral_yeast/full_predictions.tsv \
    --eval_path /net/noble/vol3/user/ssontha2/yash_noble_rotation/reclone/2024_ssontha2_ms-targeted-dl/results/yash_new/2026.02.17_restartfromscratch/make_predictions/astral_yeast 

```


### Finetune

```shell
# -model_out_dir directory to save the model checkpoints
# -data_out_dir directory to validation set predictions during finetuning
# -train_file output of assign_labels.py for train set of finetuning dataset
# -val_file output of assign_labels.py for val set of finetuning dataset
# -load_model path of pretrained checkpoint to use for finetuning
# -epochs the number of epochs to train for
# -batch_size batch size for one forward pass
# -lr learning rate
# -max-len maximum peptide length
# -max-charge maximum peptide charge
# -cpu number of cpu's
# -weight-decay weight decay strength

python train.py train \
    --model_out_dir /net/noble/vol3/user/ssontha2/yash_noble_rotation/reclone/2024_ssontha2_ms-targeted-dl/results/yash_new/2026.02.17_restartfromscratch/finetune/lumos_human/finetuned \
    --data_out_dir /net/noble/vol3/user/ssontha2/yash_noble_rotation/reclone/2024_ssontha2_ms-targeted-dl/data/2026.02.17_restartfromscratch/finetuning/lumos_human \
    --train_file /net/noble/vol3/user/ssontha2/yash_noble_rotation/reclone/2024_ssontha2_ms-targeted-dl/data/2026.02.17_restartfromscratch/finetuning/lumos_human/train.tsv \
    --val_file /net/noble/vol3/user/ssontha2/yash_noble_rotation/reclone/2024_ssontha2_ms-targeted-dl/data/2026.02.17_restartfromscratch/finetuning/lumos_human/val.tsv \
    --load_model /net/noble/vol3/user/ssontha2/yash_noble_rotation/reclone/2024_ssontha2_ms-targeted-dl/results/yash_new/2026.02.17_restartfromscratch/train/peptide_transformer_state_step25700_best.pth \
    --epochs 5 \
    --batch-size 4096 \
    --lr 0.00001 \
    --max-len 30 \
    --max-charge 4 \
    --cpu 4 \
    --weight-decay 1e-2

```



### Datasets

```
Large cancer cell lines dataset:
/net/noble/vol4/noble/user/bwen1/project/data_common/ProCan-DepMapSanger_DIANN_output.tsv.gz

Small datasets:
/net/noble/vol1/home/bwen1/github/2024_ssontha2_ms-targeted-dl/results/bo/20241221_diann/ :
lumos/human/diann/diann/report.tsv
lumos/yeast/diann/diann/report.tsv
astral/human/diann/diann/report.tsv
astral/yeast/diann/diann/report.tsv
exploris480/human/diann/diann/report.tsv
exploris480/yeast/diann/diann/report.tsv

Databases:
/net/noble/vol1/home/bwen1/github/2024_ssontha2_ms-targeted-dl/results/bo/20241221_diann/databases/

```

### Benchmarking data

#### PREGO
```
## All human proteins prediction:
/net/noble/vol1/home/bwen1/github/2024_ssontha2_ms-targeted-dl/results/bo/20241028/prego_output_UP000005640_9606.txt
## All yeast proteins prediction:
/net/noble/vol1/home/bwen1/github/2024_ssontha2_ms-targeted-dl/results/bo/20241028/prego_output_UP000002311_559292.txt
```

#### PeptideRanger
```
## All human proteins prediction:
/net/noble/vol1/home/bwen1/github/2024_ssontha2_ms-targeted-dl/results/bo/PeptideRanger/human_prioritized_peptides.tsv
## All yeast proteins prediction:
/net/noble/vol1/home/bwen1/github/2024_ssontha2_ms-targeted-dl/results/bo/PeptideRanger/UP000002311_559292_prioritized_peptides.tsv
```

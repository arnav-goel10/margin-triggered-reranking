#!/bin/bash
#SBATCH --job-name=mtr_reader
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --exclude=xgph12
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/scratch/e/e1130141/margin-verifier/logs/reader_%j.out
#SBATCH --error=/mnt/scratch/e/e1130141/margin-verifier/logs/reader_%j.err

# Stage 1. Split SQuAD train by article, fine-tune RoBERTa-base on reader_train
# with the published reader's settings (lr 2e-5, weight decay 0.01, 2 epochs,
# batch 8 x 2 accumulation), pick the epoch on reader_val, then write top-5
# candidates for reader_train, reader_val, heldout and dev.
# The reader never sees heldout or dev; dev candidates are only written here.

source "${MTR_REPO:-/mnt/scratch/e/e1130141/margin-verifier/repo}/slurm/common.sh"
gpu_guard

python scripts/verifier/split_train.py
if [ ! -f "$WORK/reader/model/config.json" ]; then
    python src/fine_tune_roberta.py \
        --model_name roberta-base \
        --train_path data/splits/reader_train.json \
        --dev_path data/splits/reader_val.json \
        --output_dir "$WORK/reader/model" \
        --epochs 2 --learning_rate 2e-5 --weight_decay 0.01 \
        --train_batch 8 --gradient_accumulation 2 \
        --max_train -1 --max_eval -1 --seed 42
    rm -rf "$WORK/reader/model"/checkpoint-*
fi
for split in reader_train reader_val heldout; do
    out="$WORK/candidates/$split.json"
    [ -f "$out" ] || python scripts/verifier/candidates.py --model "$WORK/reader/model" \
        --squad-file "data/splits/$split.json" --out-file "$out" --top-k 5
done
[ -f "$WORK/candidates/dev.json" ] || python scripts/verifier/candidates.py \
    --model "$WORK/reader/model" --squad-file data/dev-v1.1.json \
    --out-file "$WORK/candidates/dev.json" --top-k 5

#!/bin/bash
#SBATCH --job-name=mtr_verifier
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --exclude=xgph12
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/scratch/e/e1130141/margin-verifier/logs/verifier_%j.out
#SBATCH --error=/mnt/scratch/e/e1130141/margin-verifier/logs/verifier_%j.err

# Stage 2. Train the ELECTRA-base answer verifier on reader_train (gold span vs
# the reader's top two wrong spans), select on reader_val, then score the top-5
# candidates of heldout, dev, and the published reader's dev candidates.

source "${MTR_REPO:-/mnt/scratch/e/e1130141/margin-verifier/repo}/slurm/common.sh"
gpu_guard

V="$WORK/verifier"
if [ ! -f "$V/model/config.json" ]; then
    python scripts/verifier/train_verifier.py \
        --train-squad data/splits/reader_train.json \
        --train-candidates "$WORK/candidates/reader_train.json" \
        --val-squad data/splits/reader_val.json \
        --val-candidates "$WORK/candidates/reader_val.json" \
        --out-dir "$V" --negatives 2 --epochs 2 --seed 13
    rm -rf "$V/checkpoints"
fi
score() {  # squad file, candidates, output
    [ -f "$3" ] || python scripts/verifier/score_candidates.py --verifier "$V/model" \
        --squad-file "$1" --candidates "$2" --out-file "$3" --top-k 5
}
score data/splits/heldout.json "$WORK/candidates/heldout.json" "$WORK/logits/heldout.json"
score data/dev-v1.1.json "$WORK/candidates/dev.json" "$WORK/logits/dev.json"
PUB=predictions/candidates/predictions_with_5_acc2.json
if [ -f "$PUB" ]; then
    score data/dev-v1.1.json "$PUB" "$WORK/logits/dev_published.json"
fi
mkdir -p results/verifier
cp "$V/train_summary.json" results/verifier/verifier_train_summary.json

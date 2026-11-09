#!/bin/bash
#SBATCH --job-name=mtr_switch
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/scratch/e/e1130141/margin-verifier/logs/switch_%j.out
#SBATCH --error=/mnt/scratch/e/e1130141/margin-verifier/logs/switch_%j.err

# Stage 3 (CPU). Tune every switch rule on heldout with 5-fold CV, then read
# dev once with the fitted rules. final_eval.py refuses a second dev read.

source "${MTR_REPO:-/mnt/scratch/e/e1130141/margin-verifier/repo}/slurm/common.sh"

python scripts/verifier/fit_switch.py --heldout data/splits/heldout.json \
    --candidates "$WORK/candidates/heldout.json" --logits "$WORK/logits/heldout.json"
pub=()
if [ -f "$WORK/logits/dev_published.json" ]; then
    pub=(--published-candidates predictions/candidates/predictions_with_5_acc2.json
         --published-logits "$WORK/logits/dev_published.json")
fi
python scripts/verifier/final_eval.py --final \
    --heldout-report results/verifier/heldout_report.json \
    --candidates "$WORK/candidates/dev.json" --logits "$WORK/logits/dev.json" "${pub[@]}"

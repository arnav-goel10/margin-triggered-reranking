#!/bin/bash
# Queue the verifier experiment on the SoC cluster: reader -> verifier -> switch.
# Staging, from the laptop:
#   M=/mnt/scratch/e/e1130141/margin-verifier
#   rsync -a --exclude .venv ./ soc:$M/repo/     # includes data/*.json and the
#                                                # published dev candidates
# Then on the login node: bash $M/repo/slurm/submit.sh
# Afterwards copy back results/verifier/ (heldout_report, dev_report,
# verifier_train_summary, split_summary).
set -euo pipefail
R="${MTR_REPO:-/mnt/scratch/e/e1130141/margin-verifier/repo}/slurm"
mkdir -p /mnt/scratch/e/e1130141/margin-verifier/logs
reader=$(sbatch --parsable "$R/reader.sbatch")
verifier=$(sbatch --parsable --dependency=afterok:$reader "$R/verifier.sbatch")
switch=$(sbatch --parsable --dependency=afterok:$verifier "$R/switch.sbatch")
echo "reader $reader, verifier $verifier, switch $switch"

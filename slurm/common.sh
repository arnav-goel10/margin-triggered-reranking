#!/bin/bash
# Shared environment for the verifier jobs. Source it from an sbatch script.
# Everything heavy lives on scratch (the home quota is nearly full):
#   $MTR_ROOT/repo/          this repository, with data/{train,dev}-v1.1.json
#   $MTR_ROOT/work/          reader, candidates, verifier, logits
#   $MTR_ROOT/venv/ cache/   Python environment and every cache
set -euo pipefail

export MTR_ROOT="${MTR_ROOT:-/mnt/scratch/e/e1130141/margin-verifier}"
case "$MTR_ROOT" in
    /mnt/scratch/*) ;;
    *) echo "FATAL: MTR_ROOT=$MTR_ROOT is not on /mnt/scratch" >&2; exit 1 ;;
esac
export REPO="$MTR_ROOT/repo"
export WORK="$MTR_ROOT/work"
export HF_HOME="$MTR_ROOT/cache/hf"
export UV_CACHE_DIR="$MTR_ROOT/cache/uv"
export PIP_CACHE_DIR="$MTR_ROOT/cache/pip"
export XDG_CACHE_HOME="$MTR_ROOT/cache/xdg"
export TMPDIR="${SLURM_TMPDIR:-$MTR_ROOT/cache/tmp}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR" \
    "$WORK"/{reader,candidates,verifier,logits} "$MTR_ROOT/logs"

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "FATAL: uv not found on PATH" >&2; exit 1; }
VENV="$MTR_ROOT/venv"
[ -x "$VENV/bin/python" ] || uv venv -p 3.11 "$VENV"
uv pip install -q -p "$VENV/bin/python" -r "$REPO/requirements-cluster.txt"
source "$VENV/bin/activate"
cd "$REPO"

gpu_guard() {
    python -c "import sys, torch; torch.cuda.is_available() or sys.exit('FATAL: no CUDA device'); print('gpu:', torch.cuda.get_device_name(0))"
}

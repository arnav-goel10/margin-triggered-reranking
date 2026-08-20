#!/usr/bin/env bash
# Fetch SQuAD v1.1 from the official Stanford mirror.
# The dataset is not committed: it is 34MB and unchanged since release.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
BASE="https://rajpurkar.github.io/SQuAD-explorer/dataset"

mkdir -p "$DIR"
for f in train-v1.1.json dev-v1.1.json; do
  if [ -f "$DIR/$f" ]; then
    echo "have $f"
  else
    echo "fetching $f"
    curl -fsSL "$BASE/$f" -o "$DIR/$f"
  fi
done
echo "SQuAD v1.1 ready in $DIR"

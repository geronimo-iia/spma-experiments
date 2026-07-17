#!/usr/bin/env bash
set -euo pipefail

# Train SPMA on full 446k normal sequences (~21 min, 8 cores).
# Model saved to data/model/hdfs_full.json.
# Run once; threshold_full.sh reuses this model for sweeps.

SPMA=/Users/geronimo/dev/projects/libraries/spma/target/release/spma
DIR="$(cd "$(dirname "$0")" && pwd)"
CORPUS="$DIR/data/splits/train_normal.txt"
MODEL="$DIR/data/model/hdfs_full.json"

if [ ! -f "$CORPUS" ]; then
  echo "ERROR: $CORPUS not found — run split.py first" >&2
  exit 1
fi

if [ -f "$MODEL" ]; then
  echo "=== full model already exists, skipping ==="
  echo "    delete data/model/hdfs_full.json to retrain"
  exit 0
fi

mkdir -p "$DIR/data/model"
echo "=== training on full corpus (446k sequences) ==="
time $SPMA train \
  --corpus "$CORPUS" \
  --output "$MODEL" \
  --beam 10

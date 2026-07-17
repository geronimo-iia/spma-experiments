#!/usr/bin/env bash
set -euo pipefail

# Step 1a — Train one model per corpus size.
# Skips sizes where model already exists.
# Run independently from sweep/eval.
#
# Usage: ./corpus_train.sh
# Models saved to: data/model/corpus/hdfs_N.json

SPMA=${SPMA:-spma}
DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN="$DIR/data/splits/train_normal.txt"
MODEL_DIR="$DIR/data/model/corpus"

mkdir -p "$MODEL_DIR"

SIZES=(1000 5000 10000 25000 50000 446579)

for N in "${SIZES[@]}"; do
  MODEL="$MODEL_DIR/hdfs_${N}.json"

  if [ -f "$MODEL" ]; then
    echo "=== N=$N: model exists, skipping ==="
    continue
  fi

  echo "=== training N=$N ==="
  head -"$N" "$TRAIN" > "/tmp/hdfs_corpus_${N}.txt"
  time $SPMA train \
    --corpus "/tmp/hdfs_corpus_${N}.txt" \
    --output "$MODEL" \
    --beam 10
  rm -f "/tmp/hdfs_corpus_${N}.txt"
done

echo ""
echo "Models in $MODEL_DIR:"
ls -lh "$MODEL_DIR"

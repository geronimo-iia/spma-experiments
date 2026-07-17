#!/usr/bin/env bash
set -euo pipefail

# Threshold sweep on full-corpus model (data/model/hdfs_full.json).
# Run train_full.sh first if model does not exist.

SPMA=/Users/geronimo/dev/projects/libraries/spma/target/release/spma
DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="$DIR/data/model/hdfs_full.json"
RESULTS="$DIR/data/results/full"

if [ ! -f "$MODEL" ]; then
  echo "ERROR: $MODEL not found — run train_full.sh first" >&2
  exit 1
fi

mkdir -p "$RESULTS"

for T in 0.0 0.1 0.2 0.3 0.5 0.7 1.0; do
  $SPMA infer --model "$MODEL" \
              --threshold "$T" \
              --input "$DIR/data/splits/test_normal.txt" \
              --json > "$RESULTS/results_normal_t${T}.jsonl" || true

  $SPMA infer --model "$MODEL" \
              --threshold "$T" \
              --input "$DIR/data/splits/test_anomaly.txt" \
              --json > "$RESULTS/results_anomaly_t${T}.jsonl" || true

  echo -n "=== T=$T === " && python "$DIR/eval.py" \
    "$RESULTS/results_normal_t${T}.jsonl" \
    "$RESULTS/results_anomaly_t${T}.jsonl"
done

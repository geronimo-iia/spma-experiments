#!/usr/bin/env bash
set -euo pipefail

SPMA=/Users/geronimo/dev/projects/libraries/spma/target/release/spma
DIR="$(cd "$(dirname "$0")" && pwd)"
CORPUS=/tmp/hdfs_50k.txt
MODEL="$DIR/data/model/hdfs_base.json"
RESULTS="$DIR/data/results/50k"

if [ ! -f "$MODEL" ]; then
  echo "=== training base model (50k) ==="
  $SPMA train --corpus "$CORPUS" --output "$MODEL" --beam 10
else
  echo "=== base model exists, skipping train ==="
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

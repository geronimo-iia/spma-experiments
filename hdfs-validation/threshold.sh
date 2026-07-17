#!/usr/bin/env bash
set -euo pipefail

SPMA=/Users/geronimo/dev/projects/libraries/spma/target/release/spma
DIR="$(cd "$(dirname "$0")" && pwd)"
CORPUS=/tmp/hdfs_50k.txt

if [ ! -f "$DIR/data/hdfs_base.json" ]; then
  echo "=== training base model ==="
  $SPMA train --corpus "$CORPUS" --output "$DIR/data/hdfs_base.json" --beam 10
else
  echo "=== base model exists, skipping train ==="
fi

for T in 0.0 0.1 0.2 0.3 0.5 0.7 1.0; do
  $SPMA infer --model "$DIR/data/hdfs_base.json" \
              --threshold "$T" \
              --input "$DIR/data/test_normal.txt" \
              --json > "$DIR/data/results_normal_t${T}.jsonl" || true

  $SPMA infer --model "$DIR/data/hdfs_base.json" \
              --threshold "$T" \
              --input "$DIR/data/test_anomaly.txt" \
              --json > "$DIR/data/results_anomaly_t${T}.jsonl" || true

  echo "=== T=$T ===" && python "$DIR/eval.py" \
    "$DIR/data/results_normal_t${T}.jsonl" \
    "$DIR/data/results_anomaly_t${T}.jsonl"
done

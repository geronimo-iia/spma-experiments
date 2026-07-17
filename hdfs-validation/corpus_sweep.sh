#!/usr/bin/env bash
set -euo pipefail

# Step 1b — Threshold sweep for each corpus model.
# Requires corpus_train.sh to have run first.
# Skips sizes where results already exist.
# Run independently from train/eval.
#
# Usage: ./corpus_sweep.sh
# Results saved to: data/results/corpus/N/results_{normal,anomaly}_tT.jsonl
# Eval summary:     data/results/corpus/N/summary.txt

SPMA=${SPMA:-spma}
DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$DIR/data/model/corpus"
RESULTS_DIR="$DIR/data/results/corpus"
TEST_NORMAL="$DIR/data/splits/test_normal.txt"
TEST_ANOMALY="$DIR/data/splits/test_anomaly.txt"

SIZES=(1000 5000 10000 25000 50000 446579)
THRESHOLDS=(0.0 0.1 0.2 0.3 0.5 0.7)

for N in "${SIZES[@]}"; do
  MODEL="$MODEL_DIR/hdfs_${N}.json"

  if [ ! -f "$MODEL" ]; then
    echo "=== N=$N: model missing, run corpus_train.sh first — skipping ==="
    continue
  fi

  OUT="$RESULTS_DIR/${N}"
  mkdir -p "$OUT"

  # --- infer pass (skips existing) ---
  echo "=== sweeping N=$N ==="
  for T in "${THRESHOLDS[@]}"; do
    NORM="$OUT/results_normal_t${T}.jsonl"
    ANOM="$OUT/results_anomaly_t${T}.jsonl"

    if [ -f "$NORM" ] && [ -f "$ANOM" ]; then
      echo "  T=$T: results exist, skipping"
      continue
    fi

    $SPMA infer --model "$MODEL" \
                --threshold "$T" \
                --input "$TEST_NORMAL" \
                --json > "$NORM" || true

    $SPMA infer --model "$MODEL" \
                --threshold "$T" \
                --input "$TEST_ANOMALY" \
                --json > "$ANOM" || true

    echo "  T=$T: done"
  done

  # --- eval pass (always regenerate summary) ---
  SUMMARY="$OUT/summary.txt"
  echo "=== N=$N ===" > "$SUMMARY"
  for T in "${THRESHOLDS[@]}"; do
    NORM="$OUT/results_normal_t${T}.jsonl"
    ANOM="$OUT/results_anomaly_t${T}.jsonl"
    if [ -f "$NORM" ] && [ -f "$ANOM" ]; then
      METRICS=$(python "$DIR/eval.py" "$NORM" "$ANOM" | grep "Precision=")
      echo "  T=${T}  ${METRICS}" >> "$SUMMARY"
    fi
  done
  echo "  summary -> $SUMMARY"
  cat "$SUMMARY"
done

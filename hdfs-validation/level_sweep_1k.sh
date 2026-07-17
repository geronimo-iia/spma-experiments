#!/usr/bin/env bash
set -euo pipefail

# Step 5 — Per-level threshold sweep on 1k model.
# Fix global T=0.0 (best for 1k), sweep level-threshold on levels 1 and 2.
#
# Usage: ./level_sweep_1k.sh
# Results saved to: data/results/level_sweep_1k/

SPMA=/Users/geronimo/dev/projects/libraries/spma/target/release/spma
DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="$DIR/data/model/corpus/hdfs_1000.json"
RESULTS="$DIR/data/results/level_sweep_1k"
NORMAL="$DIR/data/splits/test_normal.txt"
ANOMALY="$DIR/data/splits/test_anomaly.txt"
GLOBAL=0.0
LEVELS=(1 2)
THRESHOLDS=(0.0 0.05 0.1 0.2 0.3 0.5)

mkdir -p "$RESULTS"

# baseline (no level threshold)
echo "=== baseline: global T=${GLOBAL} ==="
$SPMA infer --model "$MODEL" --threshold "$GLOBAL" \
    --input "$NORMAL"  --json > "$RESULTS/norm_base.jsonl" || true
$SPMA infer --model "$MODEL" --threshold "$GLOBAL" \
    --input "$ANOMALY" --json > "$RESULTS/anom_base.jsonl" || true
echo -n "  base: " && python "$DIR/eval.py" "$RESULTS/norm_base.jsonl" "$RESULTS/anom_base.jsonl"

echo ""
for LVL in "${LEVELS[@]}"; do
  echo "=== level ${LVL} sweep ==="
  for T in "${THRESHOLDS[@]}"; do
    NORM="$RESULTS/norm_l${LVL}_t${T}.jsonl"
    ANOM="$RESULTS/anom_l${LVL}_t${T}.jsonl"

    if [ -f "$NORM" ] && [ -f "$ANOM" ]; then
      echo -n "  L${LVL} T=${T} (cached): "
    else
      $SPMA infer --model "$MODEL" \
                  --threshold "$GLOBAL" \
                  --level-threshold "${LVL}:${T}" \
                  --input "$NORMAL"  --json > "$NORM" || true
      $SPMA infer --model "$MODEL" \
                  --threshold "$GLOBAL" \
                  --level-threshold "${LVL}:${T}" \
                  --input "$ANOMALY" --json > "$ANOM" || true
      echo -n "  L${LVL} T=${T}: "
    fi
    python "$DIR/eval.py" "$NORM" "$ANOM"
  done
  echo ""
done

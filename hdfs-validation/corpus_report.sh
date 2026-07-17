#!/usr/bin/env bash
set -euo pipefail

# Step 1c — Summarize best F1 per corpus size.
# Reads summary.txt files written by corpus_sweep.sh — no re-eval needed.
# Safe to re-run anytime — read-only.
#
# Usage: ./corpus_report.sh

DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$DIR/data/results/corpus"

SIZES=(1000 5000 10000 25000 50000 446579)
THRESHOLDS=(0.0 0.1 0.2 0.3 0.5 0.7)

printf "%-10s  %-6s  %-10s  %-8s  %-8s  %-6s\n" \
  "corpus_n" "best_T" "precision" "recall" "f1" "status"
echo "--------------------------------------------------------------"

for N in "${SIZES[@]}"; do
  OUT="$RESULTS_DIR/${N}"
  SUMMARY="$OUT/summary.txt"

  if [ ! -f "$SUMMARY" ]; then
    printf "%-10s  missing — run corpus_sweep.sh\n" "$N"
    continue
  fi

  BEST_F1="0"
  BEST_T="-"
  BEST_P="-"
  BEST_R="-"

  for T in "${THRESHOLDS[@]}"; do
    METRICS=$(grep "T=${T}[[:space:]]" "$SUMMARY" 2>/dev/null || true)
    [ -z "$METRICS" ] && continue

    F1=$(echo "$METRICS" | grep -oE 'F1=[0-9.]+' | cut -d= -f2)
    P=$(echo "$METRICS"  | grep -oE 'Precision=[0-9.]+' | cut -d= -f2)
    R=$(echo "$METRICS"  | grep -oE 'Recall=[0-9.]+' | cut -d= -f2)

    if python3 -c "exit(0 if float('${F1:-0}') > float('$BEST_F1') else 1)" 2>/dev/null; then
      BEST_F1="$F1"
      BEST_T="$T"
      BEST_P="$P"
      BEST_R="$R"
    fi
  done

  STATUS="ok"
  [ "$BEST_T" = "-" ] && STATUS="no results"

  printf "%-10s  %-6s  %-10s  %-8s  %-6s  %s\n" \
    "$N" "$BEST_T" "$BEST_P" "$BEST_R" "$BEST_F1" "$STATUS"
done

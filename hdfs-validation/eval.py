import json
import sys

normal_path  = sys.argv[1] if len(sys.argv) > 1 else "data/results_normal.jsonl"
anomaly_path = sys.argv[2] if len(sys.argv) > 2 else "data/results_anomaly.jsonl"

def load(path, true_label):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append({"predicted": r["is_anomaly"], "true": true_label})
    return rows

rows = load(normal_path, False) + load(anomaly_path, True)

tp = sum(1 for r in rows if     r["predicted"] and     r["true"])
fp = sum(1 for r in rows if     r["predicted"] and not r["true"])
fn = sum(1 for r in rows if not r["predicted"] and     r["true"])
tn = sum(1 for r in rows if not r["predicted"] and not r["true"])

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall    = tp / (tp + fn) if (tp + fn) else 0.0
f1 = 2*precision*recall / (precision+recall) if (precision+recall) else 0.0

print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

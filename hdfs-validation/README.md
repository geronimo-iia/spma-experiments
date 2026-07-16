# HDFS Validation Experiment

Validate SPMA anomaly detection against the LogHub HDFS dataset — a benchmark
log dataset with ground-truth anomaly labels. Goal: measure precision, recall,
and F1 against known results from the literature.

## Dataset

**Source**: LogHub HDFS_v1 — https://github.com/logpai/loghub/tree/master/HDFS

Two tiers:

| Tier | Size | Labels | Usable for |
|---|---|---|---|
| 2k sample | 2000 lines, 1994 blocks | None | Pipeline verification only |
| Full dataset | ~11M lines, 575k blocks | Yes (anomaly_label.csv) | P/R/F1 evaluation |

**Why the 2k sample is not enough**: the 2k sample is a slice of the full log.
Most blocks span hundreds of lines, so in 2000 lines each block appears only
1–2 times → sequences of length 1. Useless for grammar induction.

## Download

### 2k sample (no request needed)

```bash
mkdir -p data
curl -L \
  https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log_structured.csv \
  -o data/HDFS_2k.log_structured.csv
```

Verified: 2001 lines (1 header + 2000 rows), 1994 unique blocks, 14 event
templates (E1–E14), avg sequence length 1.0. Use only to verify parse.py works.

### Full dataset (request required)

The full `HDFS.log` (~1.5 GB) requires a request via the LogHub Google Form
linked in the repo README. Once granted, you receive:

```
HDFS.log_structured.csv   — ~11M lines, pre-parsed, EventId column present
anomaly_label.csv          — BlockId,Label  (Normal / Anomaly)
```

Place both in `hdfs-validation/data/` (gitignored).

Full dataset stats:
- 11,175,629 log lines
- 575,061 block sequences
- 30 event templates (E1–E30)
- 16,838 anomalous blocks (~2.9%)
- Typical sequence length: 5–29 events

## Event vocabulary

LogHub pre-parsed the logs via template mining. The `EventId` column is the
token — no regex parsing needed.

2k sample templates (E1–E14):

| EventId | Template |
|---|---|
| E1  | Served block blk_* to /* |
| E2  | Starting thread to transfer block blk_* to * |
| E3  | Got exception while serving blk_* to /* |
| E4  | BLOCK* ask * to delete blk_* |
| E5  | BLOCK* ask * to replicate blk_* to * |
| E6  | BLOCK* NameSystem.addStoredBlock: blockMap updated |
| E7  | BLOCK* NameSystem.allocateBlock |
| E8  | BLOCK* NameSystem.delete: blk_* is added to invalidSet |
| E9  | Deleting block blk_* file /* |
| E10 | PacketResponder * for block blk_* terminating |
| E11 | Received block blk_* of size * from /* |
| E12 | Received block blk_* src * dest * of size * |
| E13 | Receiving block blk_* src * dest * |
| E14 | Verification succeeded for blk_* |

Full dataset adds E15–E30 (replication, errors, timeouts — see HDFS_templates.csv).

## Pipeline overview

```
HDFS_structured.csv ──► parse.py ──► sequences.tsv ──┐
                                                       ├──► spma train ──► model.json
anomaly_label.csv ──► split.py ──► train_normal.txt ──┘
                                   test_normal.txt  ──► spma infer ──► results_normal.jsonl
                                   test_anomaly.txt ──► spma infer ──► results_anomaly.jsonl
                                                                    ──► eval.py ──► P/R/F1
```

## Step 0 — Verify pipeline on 2k sample

Before the full dataset, confirm parse.py works:

```bash
python parse.py data/HDFS_2k.log_structured.csv > data/sequences_2k.tsv
wc -l data/sequences_2k.tsv      # expected: 1994
head -5 data/sequences_2k.tsv    # expected: blk_XXXXX\tE10 ...
```

Expected output (verified 2025-07-17):
```
1994 data/sequences_2k.tsv
blk_38865049064139660     E10
blk_-6952295868487656571  E10
blk_7128370237687728475   E6
blk_8229193803249955061   E10
blk_-6670958622368987959  E10
```

## Step 1 — Parse: structured CSV → block sequences

Script: `parse.py`

```python
import csv
import re
import sys
from collections import defaultdict

BLK_RE = re.compile(r"(blk_-?\d+)")
sequences = defaultdict(list)

with open(sys.argv[1], newline="") as f:
    for row in csv.DictReader(f):
        m = BLK_RE.search(row["Content"])
        if not m:
            continue
        sequences[m.group(1)].append(row["EventId"])

for blk, events in sequences.items():
    print(f"{blk}\t{' '.join(events)}")
```

Run on full dataset:
```bash
python parse.py data/HDFS.log_structured.csv > data/sequences.tsv
wc -l data/sequences.tsv   # expected: ~575,061
```

Output format: `blk_-1608999687919862906\tE22 E5 E6 E11 E9 E26`

## Step 2 — Split: join labels, write train/test files

Script: `split.py`

```python
import csv

labels = {}
with open("data/anomaly_label.csv") as f:
    for row in csv.DictReader(f):
        labels[row["BlockId"]] = row["Label"]

train_normal = open("data/train_normal.txt", "w")
test_normal  = open("data/test_normal.txt",  "w")
test_anomaly = open("data/test_anomaly.txt", "w")
normal_count = 0

with open("data/sequences.tsv") as f:
    for line in f:
        blk, tokens = line.strip().split("\t", 1)
        label = labels.get(blk, "Normal")
        if label == "Anomaly":
            test_anomaly.write(tokens + "\n")
        else:
            normal_count += 1
            if normal_count % 5 == 0:
                test_normal.write(tokens + "\n")
            else:
                train_normal.write(tokens + "\n")

train_normal.close(); test_normal.close(); test_anomaly.close()
print(f"Normal: {normal_count}  train ~{normal_count*4//5}  test ~{normal_count//5}")
```

Run:
```bash
python split.py
# expected: Normal: ~558223  train ~446578  test ~111645
```

## Step 3 — Train

```bash
spma train \
  --corpus data/train_normal.txt \
  --output data/hdfs.json \
  --beam 10
```

Expected:
```
trained: ~446578 sequences, N grammar levels, threshold=0.0000
```

## Step 4 — Infer

```bash
spma infer --model data/hdfs.json \
           --input data/test_normal.txt \
           --json > data/results_normal.jsonl

spma infer --model data/hdfs.json \
           --input data/test_anomaly.txt \
           --json > data/results_anomaly.jsonl || true
```

(`|| true` suppresses the exit-1 from detected anomalies.)

## Step 5 — Evaluate

Script: `eval.py`

```python
import json

def load(path, true_label):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append({"predicted": r["is_anomaly"], "true": true_label})
    return rows

rows = load("data/results_normal.jsonl", False) + \
       load("data/results_anomaly.jsonl", True)

tp = sum(1 for r in rows if     r["predicted"] and     r["true"])
fp = sum(1 for r in rows if     r["predicted"] and not r["true"])
fn = sum(1 for r in rows if not r["predicted"] and     r["true"])
tn = sum(1 for r in rows if not r["predicted"] and not r["true"])

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall    = tp / (tp + fn) if (tp + fn) else 0.0
f1 = 2*precision*recall / (precision+recall) if (precision+recall) else 0.0

print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
```

Run:
```bash
python eval.py
```

## Literature baseline

| Method | F1 |
|---|---|
| DeepLog (2017) | 0.975 |
| LogAnomaly (2019) | 0.958 |
| LogBERT (2021) | 0.980 |
| PCA (classical) | 0.975 |
| Invariant mining | 0.925 |

SPMA is unsupervised, symbolic, no tuning. F1 > 0.80 = competitive.
F1 < 0.50 = grammar not capturing enough structure.

## Threshold tuning

Default threshold = 0.0. Sweep if precision is low:

```bash
for T in 0.0 0.1 0.2 0.3 0.5; do
  spma train --corpus data/train_normal.txt \
             --output data/hdfs_t${T}.json \
             --threshold $T
  spma infer --model data/hdfs_t${T}.json \
             --input data/test_normal.txt \
             --json > data/results_normal_t${T}.jsonl
  spma infer --model data/hdfs_t${T}.json \
             --input data/test_anomaly.txt \
             --json > data/results_anomaly_t${T}.jsonl || true
  echo "=== threshold=$T ===" && python eval.py  # edit paths in eval.py
done
```

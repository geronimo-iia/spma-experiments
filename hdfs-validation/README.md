# HDFS Validation Experiment

Validate SPMA anomaly detection against the LogHub HDFS dataset — a benchmark
log dataset with ground-truth anomaly labels. Goal: measure precision, recall,
and F1 against known results from the literature.

## Dataset

**Source**: LogHub — https://github.com/logpai/loghub (HDFS folder)

**Files needed**:
```
HDFS.log            — raw log lines (~11M lines, ~1.5 GB)
anomaly_label.csv   — block_id,Label (Normal / Anomaly)
```

Download and place both files in `hdfs-validation/data/` (gitignored).

**Stats** (published):
- 11,175,629 log lines
- 575,061 block sequences
- 16,838 anomalous blocks (~2.9%)

## Pipeline overview

```
HDFS.log  ──► parse.py ──► sequences.txt  ──┐
                                             ├──► spma train ──► model.json
anomaly_label.csv ──► split.py ──► train.txt ┘
                                   test.txt  ──► spma infer ──► results.jsonl
                                                               ──► eval.py ──► P/R/F1
```

## Step 1 — Parse: log lines → block sequences

Each log line contains a block ID and an event type. Group by block ID,
collect event types in order → one sequence per block.

**Event extraction**: HDFS log lines follow this template:

```
081109 203615 148 INFO dfs.DataNode$PacketResponder: Received block blk_-1608999687919862906 of size 67108864 from /10.250.10.6
```

Event keywords to extract (covers >99% of HDFS log lines):

| Log component / message fragment | Token |
|---|---|
| `PacketResponder: Received block` | `RECEIVED` |
| `PacketResponder: Transmitted block` | `TRANSMITTED` |
| `FSNamesystem: BLOCK\* NameSystem.allocateBlock` | `ALLOCATE` |
| `FSNamesystem: BLOCK\* NameSystem.addStoredBlock` | `STORED` |
| `FSNamesystem: BLOCK\* NameSystem.delete` | `DELETE` |
| `FSDataset: Deleting block` | `DELETING` |
| `DataBlockScanner: Verification succeeded` | `VERIFIED` |
| `DataBlockScanner: Scanning block` | `SCANNING` |
| `replication.*blk_` | `REPLICATE` |
| `PacketResponder.*Exception` | `ERROR` |
| `terminating` | `TERMINATE` |

Script: `parse.py`

```python
import re
import sys
from collections import defaultdict

PATTERNS = [
    (re.compile(r"PacketResponder.*Exception"),         "ERROR"),
    (re.compile(r"PacketResponder.*Received block"),    "RECEIVED"),
    (re.compile(r"PacketResponder.*Transmitted block"), "TRANSMITTED"),
    (re.compile(r"NameSystem\.allocateBlock"),          "ALLOCATE"),
    (re.compile(r"NameSystem\.addStoredBlock"),         "STORED"),
    (re.compile(r"NameSystem\.delete"),                 "DELETE"),
    (re.compile(r"FSDataset.*Deleting block"),          "DELETING"),
    (re.compile(r"Verification succeeded"),             "VERIFIED"),
    (re.compile(r"Scanning block"),                     "SCANNING"),
    (re.compile(r"replication"),                        "REPLICATE"),
    (re.compile(r"terminating"),                        "TERMINATE"),
]

BLK_RE = re.compile(r"(blk_-?\d+)")

sequences = defaultdict(list)

with open(sys.argv[1]) as f:
    for line in f:
        m = BLK_RE.search(line)
        if not m:
            continue
        blk = m.group(1)
        for pat, token in PATTERNS:
            if pat.search(line):
                sequences[blk].append(token)
                break

# Emit: block_id TAB space-separated tokens
for blk, tokens in sequences.items():
    print(f"{blk}\t{' '.join(tokens)}")
```

Run:
```bash
python parse.py data/HDFS.log > data/sequences.tsv
```

Output format: `blk_-1608999687919862906\tRECEIVED WRITE REPLICATE DELETE`

## Step 2 — Split: join labels, write train/test files

Script: `split.py`

```python
import csv
import sys

# Load labels
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
            # First 80% of normals → train, rest → test
            if normal_count % 5 == 0:
                test_normal.write(tokens + "\n")
            else:
                train_normal.write(tokens + "\n")

train_normal.close()
test_normal.close()
test_anomaly.close()
print(f"Normal sequences: {normal_count}")
print(f"  train: ~{normal_count * 4 // 5}")
print(f"  test:  ~{normal_count // 5}")
```

Run:
```bash
python split.py
```

## Step 3 — Train

```bash
spma train \
  --corpus data/train_normal.txt \
  --output data/hdfs.json \
  --beam 10
```

Expected output:
```
trained: N sequences, M grammar levels, threshold=0.0000
```

## Step 4 — Infer

```bash
spma infer --model data/hdfs.json --input data/test_normal.txt  --json > data/results_normal.jsonl
spma infer --model data/hdfs.json --input data/test_anomaly.txt --json > data/results_anomaly.jsonl
```

Note: `spma infer` exits 1 if any anomaly is detected — ignore exit code here:
```bash
spma infer --model data/hdfs.json --input data/test_anomaly.txt --json > data/results_anomaly.jsonl || true
```

## Step 5 — Evaluate

Script: `eval.py`

```python
import json
import sys

def load(path, true_label):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append({"predicted": r["is_anomaly"], "true": true_label})
    return rows

rows = load("data/results_normal.jsonl", False) + load("data/results_anomaly.jsonl", True)

tp = sum(1 for r in rows if     r["predicted"] and     r["true"])
fp = sum(1 for r in rows if     r["predicted"] and not r["true"])
fn = sum(1 for r in rows if not r["predicted"] and     r["true"])
tn = sum(1 for r in rows if not r["predicted"] and not r["true"])

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall    = tp / (tp + fn) if (tp + fn) else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
```

Run:
```bash
python eval.py
```

## Literature baseline

Published F1 scores on HDFS for log-based anomaly detection methods:

| Method | F1 |
|---|---|
| DeepLog (2017) | 0.975 |
| LogAnomaly (2019) | 0.958 |
| LogBERT (2021) | 0.980 |
| PCA (classical) | 0.975 |
| Invariant mining | 0.925 |

SPMA is an unsupervised symbolic method with no tuning — F1 > 0.80 would be
competitive. F1 < 0.50 signals the grammar is not capturing enough structure
from the normal sequences (corpus too varied, frequency threshold too high, or
event vocabulary needs refinement).

## Threshold tuning

Default threshold = 0.0 (any uncovered symbol = anomaly). This is strict.

If recall is high but precision is low (many false positives), raise the threshold:

```bash
spma train --corpus data/train_normal.txt --output data/hdfs_t05.json --beam 10 --threshold 0.5
spma infer --model data/hdfs_t05.json --input data/test_normal.txt  --json > data/results_normal_t05.jsonl
spma infer --model data/hdfs_t05.json --input data/test_anomaly.txt --json > data/results_anomaly_t05.jsonl || true
python eval.py  # edit paths in eval.py
```

Sweep `--threshold 0.0 0.1 0.2 0.3 0.5` and plot P/R curve.

## .gitignore

Add to `hdfs-validation/.gitignore`:
```
data/
```

Raw HDFS data is ~1.5 GB and must not be committed.

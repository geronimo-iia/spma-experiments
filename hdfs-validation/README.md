# HDFS Validation Experiment

Validate SPMA anomaly detection against the LogHub HDFS dataset — a benchmark
log dataset with ground-truth anomaly labels. Goal: measure precision, recall,
and F1 against known results from the literature.

## Dataset

**Source**: LogHub — https://github.com/logpai/loghub (HDFS folder)

Two tiers available:

### 2k sample (in repo, no request needed)

```
HDFS_2k.log_structured.csv   — 2000 pre-parsed log lines with EventId column
```

Direct download:
```bash
mkdir -p data
curl -L https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log_structured.csv \
     -o data/HDFS_2k.log_structured.csv
```

No anomaly labels for the 2k sample — use it to verify the parse pipeline and
inspect grammar quality only. Not suitable for P/R/F1 evaluation.

### Full dataset (request required)

Per the LogHub README, the full dataset must be requested via Google Form.
Files needed once access is granted:

```
HDFS.log_structured.csv   — ~11M lines, pre-parsed (EventId column present)
anomaly_label.csv          — BlockId,Label (Normal / Anomaly)
```

Place both in `hdfs-validation/data/` (gitignored).

**Stats** (published):
- 11,175,629 log lines
- 575,061 block sequences
- 16,838 anomalous blocks (~2.9%)

## Key insight: no regex parsing needed

LogHub ships pre-parsed structured CSV. The `EventId` column (`E1`–`E30`) is
already the normalized event token. The 30 event templates are:

| EventId | Template (abbreviated) |
|---|---|
| E5  | Receiving block … src … dest |
| E6  | Received block … src … dest … size |
| E9  | Received block … of size … from |
| E10 | PacketResponder … Exception |
| E11 | PacketResponder … for block … terminating |
| E16 | Transmitted block … to |
| E17 | Failed to transfer … got |
| E21 | Deleting block … file |
| E22 | NameSystem … allocateBlock |
| E23 | NameSystem … delete … invalidSet |
| E25 | ask … to replicate … to |
| E26 | NameSystem … addStoredBlock … is added |
| E29 | PendingReplicationMonitor timed out block |
| … | (30 total — see HDFS_templates.csv) |

Token vocabulary for SPMA: `E1 E2 … E30` — 30 symbols, clean, no ambiguity.

## Pipeline overview

```
HDFS_structured.csv ──► parse.py ──► sequences.tsv ──┐
                                                       ├──► spma train ──► model.json
anomaly_label.csv ──► split.py ──► train_normal.txt ──┘
                                   test_normal.txt  ──► spma infer ──► results_normal.jsonl
                                   test_anomaly.txt ──► spma infer ──► results_anomaly.jsonl
                                                                    ──► eval.py ──► P/R/F1
```

## Step 1 — Parse: structured CSV → block sequences

Block ID is embedded in the `Content` column as `blk_XXXXX`. Group rows by
block ID, collect `EventId` values in order.

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
        blk = m.group(1)
        sequences[blk].append(row["EventId"])

# Emit: block_id TAB space-separated EventIds
for blk, events in sequences.items():
    print(f"{blk}\t{' '.join(events)}")
```

Run:
```bash
python parse.py data/HDFS.log_structured.csv > data/sequences.tsv
# or for the 2k sample:
python parse.py data/HDFS_2k.log_structured.csv > data/sequences_2k.tsv
```

Output format: `blk_-1608999687919862906\tE22 E5 E6 E11 E9 E26`

## Step 2 — Split: join labels, write train/test files

Script: `split.py`

```python
import csv

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
            # 80% train, 20% test
            if normal_count % 5 == 0:
                test_normal.write(tokens + "\n")
            else:
                train_normal.write(tokens + "\n")

train_normal.close()
test_normal.close()
test_anomaly.close()
print(f"Normal: {normal_count}  (train ~{normal_count*4//5}, test ~{normal_count//5})")
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

## Step 4 — Infer

```bash
spma infer --model data/hdfs.json --input data/test_normal.txt  --json > data/results_normal.jsonl
spma infer --model data/hdfs.json --input data/test_anomaly.txt --json > data/results_anomaly.jsonl || true
```

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

## Development: verify pipeline on 2k sample

Before requesting the full dataset, verify parse output is correct:

```bash
python parse.py data/HDFS_2k.log_structured.csv > data/sequences_2k.tsv
head -5 data/sequences_2k.tsv
# Expected: blk_XXXXX\tE10 E11 E26 ...
wc -l data/sequences_2k.tsv
# Expected: ~50 unique blocks from 2000 lines
```

Inspect a few sequences to confirm EventIds match the templates table above.

## Literature baseline

Published F1 scores on HDFS:

| Method | F1 |
|---|---|
| DeepLog (2017) | 0.975 |
| LogAnomaly (2019) | 0.958 |
| LogBERT (2021) | 0.980 |
| PCA (classical) | 0.975 |
| Invariant mining | 0.925 |

SPMA is unsupervised, no tuning, symbolic. F1 > 0.80 competitive.
F1 < 0.50 → grammar not capturing enough structure (corpus too varied,
frequency threshold too high, or token vocabulary needs deduplication).

## Threshold tuning

Default threshold = 0.0. If precision low / false positives high, raise:

```bash
spma train --corpus data/train_normal.txt --output data/hdfs_t02.json --threshold 0.2
spma infer --model data/hdfs_t02.json --input data/test_normal.txt  --json > data/results_normal_t02.jsonl
spma infer --model data/hdfs_t02.json --input data/test_anomaly.txt --json > data/results_anomaly_t02.jsonl || true
python eval.py  # edit paths in eval.py
```

Sweep `0.0 0.1 0.2 0.3 0.5` and compare P/R/F1.

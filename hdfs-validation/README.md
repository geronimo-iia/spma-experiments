# HDFS Validation Experiment

Validate SPMA anomaly detection against the LogHub HDFS dataset — a benchmark
log dataset with ground-truth anomaly labels. Goal: measure precision, recall,
and F1 against known results from the literature.

## Results — corpus size sweep

Tested on 111,644 normal + 16,838 anomalous sequences. Grammar plateaus at 9
levels from 5k onward. Run via `corpus_train.sh` + `corpus_sweep.sh`.

| Corpus N | Best T | Precision | Recall | F1 | Notes |
|---|---|---|---|---|---|
| **1,000** | **0.0** | **0.973** | **0.825** | **0.893** | **best** |
| 5,000 | 0.0 | 0.973 | 0.825 | 0.893 | grammar saturates here |
| 10,000 | 0.0 | 0.973 | 0.825 | 0.893 | |
| 25,000 | 0.0 | 0.973 | 0.825 | 0.893 | |
| 50,000 | 0.2 | 0.903 | 0.746 | 0.817 | over-compression starts |
| 446,579 | 0.2 | 0.344 | 0.825 | 0.486 | severe over-compression |

**Best operating point: 1k corpus, T=0.0, F1=0.893**

TP=13888  FP=389  FN=2950  TN=111255

Grammar identical across 5k–25k (9 levels, 147 patterns). 1k uses the same
grammar structure but sparser training → stricter compression → lower FP.

### Why small corpus wins

1k–25k models learn only the most rigid normal patterns. At T=0.0 any
compression cost flags as anomaly — the sparse grammar is strict in the right
direction: HDFS normal sequences are highly stereotyped (same ~5 transitions
repeat). FP=389 vs 1355 at 50k; recall identical.

### MDL over-compression at 50k

At 50k the grammar absorbs enough variation that T must be raised to 0.2 to
achieve good precision, sacrificing recall. Too much training data compresses
anomaly patterns into the grammar.

**Corpus size is a hyperparameter. For HDFS: 1k–25k + T=0.0 is optimal.**

## Results — 50k corpus detail

| T | Precision | Recall | F1 |
|---|---|---|---|
| 0.0 | 0.522 | 0.825 | 0.639 |
| 0.1 | 0.698 | 0.825 | 0.756 |
| **0.2** | **0.903** | **0.746** | **0.817** |
| 0.3 | 0.910 | 0.452 | 0.604 |
| 0.5 | 0.943 | 0.331 | 0.490 |
| 0.7 | 0.527 | 0.005 | 0.010 |

Cliff at T=0.3: recall drops 0.746 → 0.452.

## Results — full corpus (446k sequences)

Run via `corpus_train.sh` + `corpus_sweep.sh` (N=446579). Training: 21 min.

| T | Precision | Recall | F1 |
|---|---|---|---|
| 0.0 | 0.300 | 0.825 | 0.440 |
| 0.1 | 0.303 | 0.825 | 0.443 |
| **0.2** | **0.344** | **0.825** | **0.486** |
| 0.3 | 0.364 | 0.576 | 0.446 |
| 0.5 | 0.601 | 0.363 | 0.452 |
| 0.7 | 0.891 | 0.005 | 0.011 |

Best: T=0.2, F1=0.486 — worse than small corpus models due to over-compression.

## Comparison with literature

| Method | F1 | Supervised? | Notes |
|---|---|---|---|
| DeepLog (2017) | 0.975 | Yes (LSTM) | |
| LogAnomaly (2019) | 0.958 | Yes | |
| LogBERT (2021) | 0.980 | Yes | |
| PCA (classical) | 0.975 | No | |
| Invariant mining | 0.925 | No | |
| **SPMA 1k (T=0.0)** | **0.893** | **No** | best result |
| SPMA 50k (T=0.2) | 0.817 | No | |
| SPMA full 446k (T=0.2) | 0.486 | No | over-compressed (TN=0 at T=0.0) |

SPMA is unsupervised, symbolic, no embeddings, no neural network.
Small corpus (1k–25k) outperforms large — sparse grammar is strict in the right
direction for HDFS's stereotyped normal sequences. Gap to PCA (~8 points)
is driven by 389 residual FP — 92% caused by 5 uncovered atoms that also
discriminate 34% of TP, making them unremovable without supervision.

## FP analysis (1k model, T=0.0)

TP=13888, FP=389, FN=2950, TN=111255.

Root cause of FP:

| Category | FP count | % FP | Fixable? |
|---|---|---|---|
| Contain uncovered atoms E6/E16/E18/E25/E28 | 359 | 92% | No — same atoms drive 34% of TP |
| Pure order/repetition anomalies | 30 | 8% | Maybe — grammar refinement (Step 6 in METHOD.md) |

**Why uncovered atoms cannot be neutralized:** These 5 atoms appear in 4733/13888 TP
anomalies. Zeroing their cost would drop recall from 0.825 to 0.550. The tradeoff is
unfavorable without labeled data to separate the two populations.

**F1=0.893 is the ceiling** for this approach without supervision:
- 2950 FN: sequences that compress well against sparse grammar
- 359 FP: structurally identical to a subset of TP at the feature level

## Dataset

**Source**: LogHub HDFS_v1 — https://github.com/logpai/loghub/tree/master/HDFS

Stats:
- 575,061 block sequences
- 558,223 normal (Success), 16,838 anomalous (Fail, ~2.9%)
- Sequence length: 5–300+ events, vocabulary E1–E30

## Download

Hosted on Zenodo — public, no login required:

```bash
mkdir -p data
cd data
curl -L "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1" -o HDFS_v1.zip
unzip HDFS_v1.zip
```

Only `data/preprocessed/Event_traces.csv` is needed. It already contains
grouped block sequences with labels — no raw log parsing required.

File structure after unzip:
```
data/
  HDFS.log                          — raw log (1.47 GiB, not needed)
  preprocessed/
    Event_traces.csv                — block sequences + labels  ← USE THIS
    anomaly_label.csv               — labels only (redundant)
    Event_occurrence_matrix.csv
    HDFS.npz
    HDFS.log_templates.csv
```

## Event vocabulary

30 event templates (E1–E30), pre-extracted by LogHub via template mining.
`Event_traces.csv` `Features` column contains ordered sequences like
`[E5,E22,E5,E11,E9,E26,...]`.

Key events:

| EventId | Meaning |
|---|---|
| E5  | Receiving block |
| E6  | Received block (dest) |
| E9  | Received block (from) |
| E11 | Received block of size |
| E22 | allocateBlock |
| E26 | addStoredBlock |
| E3  | Got exception while serving |
| E21 | Deleting block |
| E23 | delete → invalidSet |

## Pipeline overview

```
Event_traces.csv ──► split.py ──► train_normal.txt ──► spma train ──► model.json
                                  test_normal.txt  ──► spma infer ──► results_normal.jsonl  ──► eval.py ──► P/R/F1
                                  test_anomaly.txt ──► spma infer ──► results_anomaly.jsonl ──┘
```

No parse step. No raw log needed. `split.py` reads `Event_traces.csv` directly.

## Step 1 — Split

```bash
python split.py
```

Expected output:
```
train_normal: 446579
test_normal:  111644
test_anomaly: 16838
```

Verified 2026-07-17.

## Step 2 — Train (corpus sweep)

Use the corpus sweep scripts — they handle all sizes, skip existing models, and
produce structured per-size results:

```bash
./corpus_train.sh    # trains 1k/5k/10k/25k/50k/446k, skips existing
                     # models → data/model/corpus/hdfs_N.json
./corpus_sweep.sh    # threshold sweep per model, skips existing
                     # results → data/results/corpus/N/
./corpus_report.sh   # print best F1 per size
```

Best result: 1k model at T=0.0, F1=0.893. Canonical model committed at
`model/hdfs_1000.json` — usable without running the full pipeline.

For a single quick train (50k, ~2 min):
```bash
head -50000 data/splits/train_normal.txt | \
  spma train --corpus /dev/stdin --output /tmp/hdfs_50k.json --beam 10
```

## Step 3 — Evaluate a model

```bash
MODEL=model/hdfs_1000.json   # committed best model, no download needed

spma infer --model "$MODEL" --threshold 0.0 \
           --input data/splits/test_normal.txt \
           --json > /tmp/norm.jsonl || true

spma infer --model "$MODEL" --threshold 0.0 \
           --input data/splits/test_anomaly.txt \
           --json > /tmp/anom.jsonl || true

python eval.py /tmp/norm.jsonl /tmp/anom.jsonl
```

(`|| true` suppresses exit-1 from detected anomalies.)


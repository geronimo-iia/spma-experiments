# HDFS Validation Experiment

Validate SPMA anomaly detection against the LogHub HDFS dataset — a benchmark
log dataset with ground-truth anomaly labels. Measures precision, recall, and F1
against known results from the literature.

SPMA is unsupervised, symbolic, no embeddings, no neural network.

## Results

Tested on 111,644 normal + 16,838 anomalous sequences.

| Corpus N | Best T | Precision | Recall | F1 | Notes |
|---|---|---|---|---|---|
| **1,000** | **0.0** | **0.973** | **0.825** | **0.893** | **best** |
| 5,000 | 0.0 | 0.973 | 0.825 | 0.893 | grammar saturates here |
| 10,000 | 0.0 | 0.973 | 0.825 | 0.893 | |
| 25,000 | 0.0 | 0.973 | 0.825 | 0.893 | |
| 50,000 | 0.2 | 0.903 | 0.746 | 0.817 | over-compression starts |
| 446,579 | 0.2 | 0.344 | 0.825 | 0.486 | severe over-compression |

**Best: 1k corpus, T=0.0** — TP=13888, FP=389, FN=2950, TN=111255

Grammar saturates at 5k (9 levels, 147 patterns). 1k uses the same structure
but sparser training → stricter compression → lower FP. Corpus size is a
hyperparameter: too large and MDL absorbs anomaly patterns into the grammar.

F1=0.893 is the ceiling without labeled supervision — 92% of FP are caused by
5 uncovered atoms that also drive 34% of TP (cannot be removed without labels).
See METHOD.md for full analysis.

## Note on recalibrate

`spma recalibrate` refits `e_distribution` on a new corpus without retraining
the grammar — useful for production deployments where you train on a small
corpus for speed then refit thresholds on a full normal set.

Tested here: 1k grammar recalibrated on all 446k normal training sequences,
then swept T=0.0–0.3. Result: **no change** vs the original 1k model at any
threshold. 99.9% of normal sequences score exactly 0.0 regardless of corpus
size — the grammar compresses normal patterns perfectly, leaving no room for
percentile-based threshold improvement. The feature is correct; HDFS just
has a clean binary score distribution that doesn't benefit from it.

## Comparison with literature

| Method | F1 | Supervised? |
|---|---|---|
| LogBERT (2021) | 0.980 | Yes |
| DeepLog (2017) | 0.975 | Yes |
| PCA (classical) | 0.975 | No |
| LogAnomaly (2019) | 0.958 | Yes |
| Invariant mining | 0.925 | No |
| **SPMA 1k (T=0.0)** | **0.893** | **No** |
| SPMA 50k (T=0.2) | 0.817 | No |
| SPMA full 446k (T=0.2) | 0.486 | No |

F1=0.893 unsupervised, no feature engineering, no embeddings. Competitive with
invariant mining (0.925) and within 8 points of PCA (0.975). This table is not
a comprehensive survey — other unsupervised methods on HDFS may exist.

## Quickstart

Best model committed at `model/hdfs_1000.json` — no dataset download needed:

```bash
spma grammar --model model/hdfs_1000.json
```

To reproduce fully or run experiments: see **METHOD.md**.

## Dataset

**Source**: LogHub HDFS_v1 — https://github.com/logpai/loghub/tree/master/HDFS

- 575,061 block sequences; 558,223 normal, 16,838 anomalous (~2.9%)
- Sequence length: 5–300+ events, vocabulary E1–E30

Download (Zenodo, public, no login):

```bash
mkdir -p data && cd data
curl -L "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1" -o HDFS_v1.zip
unzip HDFS_v1.zip
```

Only `data/preprocessed/Event_traces.csv` is needed — grouped block sequences
with labels, no raw log parsing required.

## Event vocabulary

30 event templates (E1–E30), pre-extracted by LogHub.

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

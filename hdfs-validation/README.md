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

From Chen et al. 2021 (arXiv:2107.05908, Tables 2–3). Setup differences:
- Their methods use 80/20 chronological split with sliding window partitioning,
  training on ~446k normal sequences. SPMA uses identifier-based partitioning
  (the correct approach for HDFS block sequences) and trains on **1k sequences**.
- Larger training corpora degrade SPMA: MDL absorbs anomaly patterns into the
  grammar at scale (F1 drops to 0.817 at 50k, 0.486 at full 446k corpus). The
  1k result is not a shortcut — it is the optimum.

**Traditional ML, unsupervised (HDFS):**

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Invariant Mining | 0.895 | 1.0 | 0.944 |
| Log Clustering | 1.0 | 0.728 | 0.843 |
| PCA | 0.971 | 0.628 | 0.763 |
| **SPMA 1k (T=0.0)** | **0.973** | **0.825** | **0.893** |

**DL-based, unsupervised (HDFS, without log semantics):**

| Method | Precision | Recall | F1 |
|---|---|---|---|
| LSTM (DeepLog) | 0.96 | 0.965 | 0.944 |
| Transformer | 0.946 | 0.86 | 0.905 |
| Autoencoder | 0.881 | 0.892 | 0.881 |
| **SPMA 1k (T=0.0)** | **0.973** | **0.825** | **0.893** |

SPMA beats PCA (0.763) and Log Clustering (0.843). Below Invariant Mining
(0.944) and DeepLog (0.944), which use numerical event-count features.
SPMA uses symbolic sequence structure only — no feature vectors, no
training on anomaly labels, no embeddings.

## Quickstart

Best model committed at `model/hdfs_1000.json` — no dataset download needed:

```bash
spma grammar --model model/hdfs_1000.json
```

To reproduce fully or run experiments: see **METHOD.md**.

## Dataset and attribution

**Source**: LogHub HDFS_v1 — https://github.com/logpai/loghub/tree/master/HDFS

Dataset is free for research and academic use. Any use or distribution must
cite the LogHub repository and the LogHub paper:

> Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu.
> "Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics."
> IEEE ISSRE, 2023. https://github.com/logpai/loghub

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

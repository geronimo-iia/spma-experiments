# SPMA Validation Method

Systematic procedure for building, calibrating, and evaluating an SPMA anomaly
detector on a labelled log dataset. Grounded in SP/MDL theory.

## Overview

SPMA learns a grammar from normal sequences via MDL compression. Anomaly score
`e_norm` measures how poorly a new sequence compresses against that grammar.
The grammar is hierarchical — atoms at level 0, pattern sequences at level 1–N.
Each level can discriminate different classes of anomaly.

Three independent hyperparameters:
1. **Corpus size** — how many normal sequences to train on
2. **Global threshold** — `e_norm` cutoff at level 0
3. **Per-level thresholds** — `e_norm` cutoffs at levels 1–N (opt-in, OR logic)

Over-training (too large corpus) causes MDL over-compression: grammar absorbs
rare/noisy patterns → anomalies compress too well → F1 degrades.

## Scripts

| Script | Purpose | Requires |
|---|---|---|
| `split.py` | Generate train/test splits from `Event_traces.csv` | dataset |
| `corpus_train.sh` | Train one model per corpus size, skip existing | splits |
| `corpus_sweep.sh` | Threshold sweep per model, skip existing results | corpus_train |
| `corpus_report.sh` | Print best F1 per corpus size (read-only) | corpus_sweep |
| `level_sweep_1k.sh` | Per-level threshold sweep on 1k model | corpus_train |
| `spma grammar` | Grammar summary: human-readable or `--json` for LLM pruning | model |
| `spma recalibrate` | Replay corpus on pruned grammar, refit e_distribution | pruned model + corpus |

Archived (superseded or abandoned): `archive/grammar_summary.py`, `archive/level_sweep_50k.sh`,
`archive/train_full.sh`, `archive/threshold_full.sh`.

## Step 1 — Corpus size exploration

Grammar saturates at some corpus size — adding more training data yields the
same levels and patterns. Find the saturation point, then find the F1 peak.
Use the **smallest corpus that saturates** as baseline.

```bash
./corpus_train.sh   # trains 1k/5k/10k/25k/50k/446k, skips existing
./corpus_sweep.sh   # threshold sweep per model, skips existing results
./corpus_report.sh  # prints best F1 per size
```

On HDFS: grammar saturates at 5k (9 levels, 147 patterns). Optimal F1 at 1k
(F1=0.893) — sparse grammar is strict in the right direction for HDFS's
stereotyped normal sequences.

## Step 2 — Grammar inspection

Understand which levels carry signal and which atoms are uncovered (always
contribute to `e_cost` regardless of threshold). Use output to decide which
levels are worth sweeping in Step 5.

```bash
spma grammar --model model/hdfs_1000.json
spma grammar --model model/hdfs_1000.json --json   # LLM-ready structured output
```

**Active levels**: only levels where anomaly p99 > 0 and mean_anomaly > 2×mean_normal.
On HDFS 1k: only level 0 discriminates. Levels 3–7 are zero.

**Uncovered atoms** on 1k–25k models: `E6, E16, E18, E25, E28` — any sequence
containing these scores `e_cost > 0`. Cannot be neutralized (see FP analysis).

## Step 3 — Global threshold sweep

`corpus_sweep.sh` covers this for all corpus sizes. Results in
`data/results/corpus/N/summary.txt`, best per size via `corpus_report.sh`.

Find the threshold where F1 peaks. Watch for recall cliffs — stay below them.

On HDFS 1k: T=0.0 optimal (F1=0.893). On 50k: cliff at T=0.3, use T=0.2.

## Step 4 — LLM-guided grammar pruning (optional)

Export grammar as JSON, ask LLM to flag domain-invalid patterns, prune them
from the model JSON, then recalibrate without re-inducing the grammar.

**When useful:** MDL over-generates spurious patterns from rare co-occurrences.
On HDFS the grammar is clean — more relevant on noisier datasets.

### 4a — Export and prompt

```bash
spma grammar --model model/hdfs_1000.json --json > /tmp/grammar_for_llm.json
```

Prompt the LLM:
> "This is an event log grammar learned by MDL. Each pattern is a sequence of
> event IDs with frequencies. Flag any pattern IDs at level 0 that represent
> transitions physically impossible in normal operation, or that correspond to
> known failure signatures. Return a JSON list of pattern indices to remove."

### 4b — Prune model JSON

Pattern indices in `grammar.levels[N].patterns` are positional:

```python
import json
m = json.load(open("model/hdfs_1000.json"))
to_remove = {K1, K2}  # indices returned by LLM
m["grammar"]["levels"][0]["patterns"] = [
    p for i, p in enumerate(m["grammar"]["levels"][0]["patterns"])
    if i not in to_remove
]
json.dump(m, open("/tmp/hdfs_pruned.json", "w"))
```

### 4c — Recalibrate and re-sweep

Grammar is frozen. Replay corpus to refit anomaly score distributions, then
re-run `corpus_sweep.sh` pointing at the pruned model. Keep only if F1 improves.

```bash
spma recalibrate \
    --model /tmp/hdfs_pruned.json \
    --corpus data/splits/train_normal.txt \
    --output /tmp/hdfs_pruned.json
```

## Step 5 — Per-level threshold sweep

OR logic: level gate can only add anomalies, never reduce FP. Only useful to
raise recall on over-compressed models where level-0 misses order violations.

Run `level_sweep_1k.sh` for the 1k model. For other models, pass
`--level-threshold N:T` to `spma infer` directly.

On HDFS 1k: level sweep adds no value — precision is the bottleneck, not recall.
Conclusion: global threshold alone is sufficient for 1k–25k models.

## Step 6 — FP-driven grammar refinement (optional)

FP sequences are normal sequences the grammar failed to compress. Appending them
to the training corpus forces MDL to absorb the missing patterns.

Only valid when ground-truth labels are reliable — label noise corrupts the grammar.

Extract FPs from infer output, append to training corpus, retrain, repeat Steps
3–5. Stop when FP delta < 5% between iterations.

```python
import json
seqs = open("data/splits/test_normal.txt").readlines()
results = [json.loads(l) for l in open("data/results/final_normal.jsonl")]
fps = [s for s, r in zip(seqs, results) if r["is_anomaly"]]
open("data/splits/fp_sequences.txt", "w").writelines(fps)
print(f"FP: {len(fps)}")
```

## Summary table

| Model | Corpus | T_global | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| **1k (best)** | **1k** | **0.0** | **13888** | **389** | **2950** | **0.973** | **0.825** | **0.893** |
| 5k–25k | 5k | 0.0 | 13888 | 389 | 2950 | 0.973 | 0.825 | 0.893 |
| 50k | 50k | 0.2 | 12561 | 1355 | 4277 | 0.903 | 0.746 | 0.817 |
| full 446k | 446k | 0.2 | 13888 | 26540 | 2950 | 0.344 | 0.825 | 0.486 |

F1=0.893 is the ceiling without labeled supervision (see FP analysis below).

## Key findings (HDFS)

### Corpus size

- Grammar saturates at 9 levels / 147 patterns at 5k — 10k/25k/50k identical structure
- 1k–25k: F1=0.893 at T=0.0 (sparse grammar strict in right direction for HDFS stereotypes)
- 50k: F1=0.817 at T=0.2 — grammar absorbs some variation, requires raised threshold
- 446k: F1=0.486 — MDL over-compression, 10 uncovered atoms, degraded precision
- **Corpus size is a hyperparameter. Optimal for HDFS: 1k–25k, T=0.0**

### Per-level threshold gate

- OR logic: level gate can only add anomalies, never reduce FP
- Cannot improve precision — only useful to raise recall on over-compressed models
- 1k model: level sweep adds no value (precision already bottleneck)
- Global threshold alone sufficient for HDFS 1k–25k models

### FP root-cause analysis (1k model, T=0.0)

FP=389 total. Root causes:

| Category | Count | % FP |
|---|---|---|
| Contain uncovered atoms (E6/E16/E18/E25/E28) | 359 | 92% |
| Pure order/repetition anomalies (no uncovered atom) | 30 | 8% |

**Why uncovered atoms cannot be zeroed:** The same 5 atoms appear in 34% of TP
(4733 / 13888 anomalies). Setting their cost to 0 turns 4733 TP into FN —
recall collapses. The approach was analyzed and rejected.

**F1=0.893 is the ceiling** for this feature set without labeled supervision:
- 2950 FN: compress well against sparse grammar, not recoverable via threshold
- 359 FP: uncovered-atom sequences, not separable from TP without labels
- 30 FP: order/repetition anomalies — potentially reducible via grammar refinement (Step 6)

### Grammar quality

- 5 uncovered atoms on 1k–25k models: `E6, E16, E18, E25, E28`
- Rare in training (below MDL threshold to build patterns), appear in normal test sequences
- Full 446k model: 10 uncovered atoms — over-compression worsens the problem

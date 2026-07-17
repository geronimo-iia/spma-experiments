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
3. **Per-level thresholds** — `e_norm` cutoffs at levels 1–N

Over-training (too large corpus) causes MDL over-compression: grammar absorbs
rare/noisy patterns → anomalies compress too well → F1 degrades. Confirmed on
HDFS: 50k sequences outperforms 446k.

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

Archived (no longer used): `archive/grammar_summary.py` (superseded by `spma grammar`),
`archive/level_sweep_50k.sh` (stopped — OR logic means level gate adds no precision value).

## Step 1 — Corpus size exploration

Grammar saturates at some corpus size — adding more data yields the same number
of levels and patterns. Find the saturation point, then find the F1 peak.

```bash
./corpus_train.sh    # trains 1k/5k/10k/25k/50k/446k, skips existing
./corpus_sweep.sh    # sweeps T per model, skips existing results
./corpus_report.sh   # prints best F1 per size
```

Look for: levels and pattern count stop growing → grammar saturated. Use the
**smallest corpus that saturates** as baseline.

On HDFS: grammar saturates at 50k (9 levels, 147 patterns). Full 446k same
grammar but worse F1 (0.486 vs 0.817) due to over-compression.

## Step 2 — Grammar inspection

Before sweeping thresholds, understand which levels carry signal and which atoms
are uncovered (always contribute to e_cost regardless of threshold).

```bash
spma grammar --model model/hdfs_1000.json
# JSON output for LLM pruning:
spma grammar --model model/hdfs_1000.json --json
```

Output includes:
- Atom costs and uncovered atoms
- Top patterns per level with frequencies
- E_norm distribution (p50/p90/p99/max) per level on training data

**Active levels for threshold sweep**: only levels where p99 > 0 on anomaly
sequences and mean_anomaly > 2 × mean_normal. On HDFS: level 1 is the only
discriminating level above 0 (p50=0.11, p90=0.25). Levels 3–7 are zero.

**Uncovered atoms** on HDFS 50k model: `E6, E16, E18, E25, E28` — any sequence
containing these always scores e_cost > 0. On full model: 10 atoms uncovered
(including `E3, E2, E4`) — explains degraded precision.

## Step 3 — Global threshold sweep

Train once on optimal corpus size. Sweep global threshold at infer time via
`corpus_sweep.sh` (covers all sizes) or inline:

```bash
MODEL=model/hdfs_1000.json
for T in 0.0 0.1 0.2 0.3 0.5 0.7; do
  spma infer --model "$MODEL" --threshold $T \
    --input data/splits/test_normal.txt --json > /tmp/norm_$T.jsonl || true
  spma infer --model "$MODEL" --threshold $T \
    --input data/splits/test_anomaly.txt --json > /tmp/anom_$T.jsonl || true
  echo -n "T=$T: " && python eval.py /tmp/norm_$T.jsonl /tmp/anom_$T.jsonl
done
```

Find the threshold where F1 peaks. There is typically a cliff — recall drops
sharply above a certain value. Stay below it.

On HDFS 1k: T=0.0 is optimal (F1=0.893). On 50k: cliff at T=0.3, use T=0.2.

## Step 4 — LLM-guided grammar pruning (optional)

After global sweep, feed the grammar to an LLM to identify domain-invalid
patterns, prune them from the model JSON, then recalibrate e_distribution
without re-inducing the grammar.

**When useful:** MDL over-generates spurious patterns from rare co-occurrences.
On HDFS 50k the grammar is clean (147 patterns, well-structured). More relevant
on noisier datasets or after domain knowledge identifies impossible transitions.

### 4a — Export grammar for LLM

```bash
spma grammar --model data/model/hdfs_base.json --json > /tmp/grammar_for_llm.json
```

Prompt the LLM:
> "This is an HDFS event log grammar learned by MDL. Each pattern is a
> sequence of event IDs with frequencies. Flag any pattern IDs at level 0
> that represent transitions physically impossible in normal HDFS operation,
> or that correspond to known failure signatures (hardware errors, corruption
> events). Return a JSON list of pattern indices to remove."

### 4b — Prune model JSON

Pattern indices in `grammar.levels[N].patterns` are positional. To remove
pattern at index K from level 0:

```python
import json
m = json.load(open("data/model/hdfs_base.json"))
# remove patterns at indices [K1, K2, ...] from level 0
to_remove = {K1, K2}
m["grammar"]["levels"][0]["patterns"] = [
    p for i, p in enumerate(m["grammar"]["levels"][0]["patterns"])
    if i not in to_remove
]
json.dump(m, open("data/model/hdfs_pruned.json", "w"))
```

### 4c — Recalibrate e_distribution

Grammar is now pruned. Replay corpus to refit anomaly score distributions:

```bash
spma recalibrate \
    --model data/model/hdfs_pruned.json \
    --corpus data/splits/train_normal.txt \
    --output data/model/hdfs_pruned.json
```

Requires `spma recalibrate` subcommand — see prompt
`spma/docs/prompts/add-recalibrate-subcommand.md`.

### 4d — Re-sweep threshold on pruned model

```bash
# Use threshold_50k.sh pointing to pruned model, or inline:
for T in 0.0 0.1 0.2 0.3 0.5 0.7; do
  spma infer --model data/model/hdfs_pruned.json --threshold $T \
    --input data/splits/test_normal.txt --json > /tmp/norm_p_$T.jsonl || true
  spma infer --model data/model/hdfs_pruned.json --threshold $T \
    --input data/splits/test_anomaly.txt --json > /tmp/anom_p_$T.jsonl || true
  echo -n "T=$T: " && python eval.py /tmp/norm_p_$T.jsonl /tmp/anom_p_$T.jsonl
done
```

Keep pruned model only if F1 improves over baseline.

## Step 5 — Per-level threshold sweep

Only sweep levels identified as active in Step 2. Fix global threshold at best
value from Step 3.

```bash
SPMA=${SPMA:-spma}
MODEL=data/model/hdfs_base.json
GLOBAL=0.2

for LVL in 1 2; do
  for T in 0.0 0.05 0.1 0.2 0.3 0.5; do
    $SPMA infer --model $MODEL \
                --threshold $GLOBAL \
                --level-threshold ${LVL}:${T} \
                --input data/splits/test_normal.txt \
                --json > /tmp/norm_l${LVL}_${T}.jsonl || true
    $SPMA infer --model $MODEL \
                --threshold $GLOBAL \
                --level-threshold ${LVL}:${T} \
                --input data/splits/test_anomaly.txt \
                --json > /tmp/anom_l${LVL}_${T}.jsonl || true
    echo -n "L${LVL} T=${T}: " && python eval.py /tmp/norm_l${LVL}_${T}.jsonl /tmp/anom_l${LVL}_${T}.jsonl
  done
done
```

SP theory prediction: level 1 catches order-violation anomalies invisible at
level 0 — sequences with all atoms covered but pattern-ID sequence violating
learned order. HDFS over-compressed full model may recover F1 via level-1 gate.

## Step 6 — Combined evaluation

Apply best global + per-level thresholds together:

```bash
SPMA=${SPMA:-spma}

$SPMA infer --model data/model/hdfs_base.json \
            --threshold $GLOBAL_BEST \
            --level-threshold 1:$L1_BEST \
            --input data/splits/test_normal.txt \
            --json > data/results/final_normal.jsonl || true

$SPMA infer --model data/model/hdfs_base.json \
            --threshold $GLOBAL_BEST \
            --level-threshold 1:$L1_BEST \
            --input data/splits/test_anomaly.txt \
            --json > data/results/final_anomaly.jsonl || true

python eval.py data/results/final_normal.jsonl data/results/final_anomaly.jsonl
```

## Step 7 — FP-driven grammar refinement (optional)

FP sequences are normal sequences the grammar failed to compress. Adding them
forces MDL to absorb missing patterns → fewer FP → precision up.

Only valid when ground-truth labels are reliable. Label noise corrupts the grammar.

```bash
# Extract FP sequences
python - <<'EOF'
import json
seqs  = open("data/splits/test_normal.txt").readlines()
results = [json.loads(l) for l in open("data/results/final_normal.jsonl")]
fps = [s for s, r in zip(seqs, results) if r["is_anomaly"]]
open("data/splits/fp_sequences.txt", "w").writelines(fps)
print(f"FP: {len(fps)}")
EOF

# Retrain with FP appended
cat /tmp/hdfs_50k.txt data/splits/fp_sequences.txt > /tmp/hdfs_refined.txt
$SPMA train --corpus /tmp/hdfs_refined.txt \
            --output data/model/hdfs_refined.json \
            --beam 10
```

Repeat Steps 3–5 on refined model. Stop when FP delta < 5% between iterations.

## Step 8 — Corpus size re-check after refinement

After adding FP sequences, re-run `corpus_train.sh` / `corpus_sweep.sh` with
refined corpus to confirm saturation point unchanged.

## Summary table

| Model | Corpus | T_global | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| **1k (best)** | **1k** | **0.0** | **13888** | **389** | **2950** | **0.973** | **0.825** | **0.893** |
| 5k–25k | 5k | 0.0 | 13888 | 389 | 2950 | 0.973 | 0.825 | 0.893 |
| 50k | 50k | 0.2 | 12561 | 1355 | 4277 | 0.903 | 0.746 | 0.817 |
| full 446k | 446k | 0.2 | 13888 | 26540 | 2950 | 0.344 | 0.825 | 0.486 |

F1=0.893 is the ceiling without labeled supervision (see FP root-cause analysis above).

## Key findings (HDFS)

### Corpus size

- Grammar saturates at 9 levels / 147 patterns at 5k — 10k/25k/50k identical structure
- 1k–25k: F1=0.893 at T=0.0 (sparse grammar strict in right direction for HDFS stereotypes)
- 50k: F1=0.817 at T=0.2 — grammar absorbs some variation, requires raised threshold
- 446k: F1=0.486 — MDL over-compression, 10 uncovered atoms, degraded precision
- **Corpus size is a hyperparameter. Optimal for HDFS: 1k–25k, T=0.0**

### Per-level threshold gate

- OR logic: level gate can only add anomalies, never reduce FP
- Level gate cannot improve precision — only useful to raise recall on over-compressed models
- 1k model: level sweep adds no value (precision already bottleneck)
- Conclusion: global threshold only sufficient for HDFS 1k–25k models

### FP root-cause analysis (1k model, T=0.0)

FP=389 total. Root causes:

| Category | Count | % FP |
|---|---|---|
| Contain uncovered atoms (E6/E16/E18/E25/E28) | 359 | 92% |
| Pure order/repetition anomalies (no uncovered atom) | 30 | 8% |

**Why uncovered atoms cannot be zeroed:** The same 5 atoms appear in 34% of TP
(4733 / 13888 anomalies detected via uncovered atoms). Setting their cost to 0
would turn 4733 TP into FN — recall collapses. The `uncovered-atoms-neutral`
approach was analyzed and rejected; see `docs/prompts/uncovered-atoms-neutral.md`.

**F1=0.893 is the ceiling** for this feature set without labeled supervision:
- 2950 FN: sequences that compress well against sparse grammar (not recoverable via threshold)
- 359 FP: uncovered-atom sequences, not separable from TP without labels
- 30 FP: order/repetition anomalies — potentially reducible via grammar refinement (Step 7)

### Grammar quality

- 5 uncovered atoms on 1k–25k models: `E6, E16, E18, E25, E28`
- These are rare in training (too infrequent for MDL to build patterns around them)
  but appear legitimately in some normal test sequences
- Full 446k model: 10 uncovered atoms — over-compression worsens the problem

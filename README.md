# sp_experiments

Experiments run against the `spma` crate. Each experiment is a standalone Rust binary.

## Layout

```
v8/   — original harness (invalid, adapted to compile, bugs preserved)
v9/   — random embeddings null baseline (reconstructed from V10, i have lose the code ...)
v10/  — MiniLM-L6-v2 semantic embeddings via ONNX (final run)
```

Each is a standalone Rust crate with `spma` path dependency.

## Files

| Path | Content |
|---|---|
| `v10/src/main.rs` | V10 source — MiniLM ONNX |
| `v10/attention_comparison.csv` | V10 results (mean Jaccard 0.030) |
| `v9/src/main.rs` | V9 source — random embeddings |
| `v9/compare_attention_results.csv` | V9 results (mean Jaccard 0.027) |
| `v8/src/main.rs` | V8 source — original harness, both bugs visible |

## Reproduce

### V10 (MiniLM embeddings)

Requires `all-MiniLM-L6-v2` ONNX model at the path in `MODEL_DIR` constant (`v10/src/main.rs:69`).

```bash
cd v10 && cargo run --release
# outputs attention_comparison.csv
```

### V9 (random embeddings — null baseline)

```bash
cd v9 && cargo run --release
# outputs compare_attention_results.csv
# expected: mean Jaccard ≈ 0.027
```

### V8 (invalid — bugs preserved, for reference)

```bash
cd v8 && cargo run --release
# compiles and runs; result is meaningless (see bug notes in src/main.rs)
```

### V8a / V8b (`v8/src/main.rs`)

Original source from git commit `31eab51`. Both bugs visible in the same file:

- **V8a bug** (key space): `old_pattern_indices` are grammar pattern IDs, not sentence indices. Jaccard over different-typed sets is noise.
- **V8b bug** (cardinality): after fixing key space, SPMA `old_pattern_indices` still returns up to K alignments × patterns each, yielding ~120 indices vs attention's top-5. Set sizes too unequal for meaningful Jaccard.

Not a valid comparison in either form. Compiles against current `spma` crate (API updated); algorithm and bugs unchanged.

## Experiment — Transformer attention vs SPMA

### Hypothesis

Transformer attention is a continuous, differentiable approximation of SPMA multiple alignment. Both perform associative retrieval: a probe pattern matched against a stored set, scored by correspondence. If true, SPMA becomes a symbolic reference implementation for a single attention head.

### Results

| Attempt | Valid? | Mean Jaccard | Issue |
|---|---|---|---|
| V8a | No | 0.068 | Key space mismatch |
| V8b | No | 0.042 | Cardinality mismatch |
| V9 | Yes | 0.027 | Random embeddings — null baseline |
| V10 | Yes | 0.022 | MiniLM-L6-v2 semantic embeddings |

**Conclusion**: hypothesis not supported by either valid experiment.

### V8a — invalid (key space mismatch)

SPMA returned token-level Old pattern IDs. Attention returned sentence indices. Jaccard over different-typed objects is noise.

### V8b — invalid (cardinality mismatch)

Fixed key space: both sides return sentence indices. New problem: SPMA returns ~120 train sentences per query (broad recall via subsequence), attention returns exactly 5 (sharp top-K). Jaccard ≈ 0.04 by construction — a set of 5 cannot overlap much with a set of 120.

### V9 — valid, genuine negative

Fix: rank SPMA-retrieved sentences by pattern specificity (`-log2(freq/total)`), keep top-5. Both sides now return exactly 5 indices. Jaccard is meaningful.

Mean Jaccard: **0.027**  Std: 0.047  Min: 0.000  Max: 0.111

Spot-checks:
- "a cat chased the dog" → Jaccard 0.000; SPMA: VP retrieval by `chased`; attention: noise
- "the cat sat a cat" → Jaccard 0.111; one index overlap by chance
- "a cat climbed the mat" → Jaccard 0.000; SPMA: VP retrieval by `climbed`; attention: noise

SPMA retrieval is structurally plausible. Attention with random embeddings is noise. Result is a genuine null: **random embeddings have no grammar signal**.

### V10 — valid, MiniLM semantic embeddings

Model: all-MiniLM-L6-v2 via ONNX. Embeddings: 384-dim, L2-normalised, mean-pooled over tokens.

Mean Jaccard: **0.022**  Std: 0.044  Min: 0.000  Max: 0.111
Delta vs random baseline: **-0.004** (not significant)

| Test sentence | SPMA top-5 | Attn top-5 | Jaccard |
|---|---|---|---|
| a cat chased the dog | all share verb `chased` — VP retrieval | near-duplicates of exact sentence | 0.111 |
| the cat sat a cat | all have `climbed` — wrong verb pattern | exact `sat`+`cat` matches | 0.000 |
| a cat climbed the mat | all share `climbed` — correct VP | near-duplicates of exact sentence | 0.000 |

Why MiniLM doesn't help: CFG vocabulary is 11 tokens. MiniLM collapses to lexical overlap and retrieves near-duplicates of the surface form. SPMA retrieves by highest-specificity VP constituent. Different retrieval objectives. -0.004 delta (below random baseline) is noise.

### Why Jaccard stays low

- **SPMA**: highest-specificity matched substructure — grammatical constituents (VP, NP)
- **Attention (random)**: token ID coincidence — no signal
- **Attention (MiniLM)**: lexical surface similarity — near-duplicates on tiny vocabulary

The analogy would require attention weights trained with a syntactic objective (constituency parsing, dependency parsing), not semantic similarity.

### V11 — closed, not pursued

No syntax-trained ONNX model available. Training one is out of scope.

**Hypothesis status: rejected.** -0.004 delta (MiniLM scores below random baseline) is noise. Closed.

### Implementation note

Grammar trained on 150-sentence CFG split (seed fixed for reproducibility).
SPMA: beam search K=5, specificity-ranked top-5 sentence retrieval.
Attention: mean-pooled embeddings, scaled dot-product, top-5 by score.

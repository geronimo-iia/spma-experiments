# spma-experiments

Experiments run against the [`spma`](https://github.com/geronimo-iia/spma) crate.

## Experiments

| Directory | Topic | Status |
|---|---|---|
| [`hdfs-validation/`](hdfs-validation/README.md) | Anomaly detection on LogHub HDFS dataset | Active — F1=0.893 |
| [`attn-spma-minilm/`](attn-spma-minilm/README.md) | SPMA vs MiniLM-L6-v2 semantic attention | Closed — hypothesis rejected |
| [`attn-spma-random/`](attn-spma-random/README.md) | SPMA vs random embeddings (null baseline) | Closed |

See each directory's README for results and reproduction steps.

## Experiment — Transformer attention vs SPMA

**Hypothesis**: transformer attention is a continuous approximation of SPMA
multiple alignment. Both perform associative retrieval — a probe pattern matched
against a stored set, scored by correspondence. If true, SPMA becomes a symbolic
reference implementation for a single attention head.

**Results**:

| Experiment | Valid? | Mean Jaccard | Notes |
|---|---|---|---|
| attn-spma-random | Yes | 0.027 | Null baseline — random embeddings |
| attn-spma-minilm | Yes | 0.022 | MiniLM-L6-v2 — below random baseline |

**Conclusion: hypothesis rejected.** MiniLM scores −0.004 below the random
baseline (not significant). CFG vocabulary is 11 tokens — MiniLM collapses to
lexical surface overlap; SPMA retrieves by highest-specificity VP constituent.
Different retrieval objectives. The analogy would require attention weights
trained with a syntactic objective, not semantic similarity.

A syntax-trained ONNX model was not available. Closed.

## Implementation notes (attn-spma experiments)

Grammar trained on 150-sentence CFG corpus (seed fixed for reproducibility).
SPMA: beam search K=5, specificity-ranked top-5 sentence retrieval (`-log2(freq/total)`).
Attention: mean-pooled embeddings, scaled dot-product, top-5 by score.

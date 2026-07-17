# V9 — SPMA vs random embeddings (null baseline)

## Purpose

Null baseline experiment. Reconstructed from V10.

Fixes V8b cardinality mismatch: SPMA-retrieved sentences now ranked by pattern specificity (`-log2(freq/total)`), top-5 kept. Both SPMA and attention return exactly 5 sentence indices — Jaccard is meaningful.

Hypothesis: random embeddings have no grammar signal, so attention retrieval is noise. If true, Jaccard ≈ 0.

## Run

```bash
cargo run --release
# outputs compare_attention_results.csv
# expected: mean Jaccard ≈ 0.026
```

## Result

```
Mean Jaccard: 0.027  Std: 0.047  Min: 0.000  Max: 0.111
```

Spot-checks:
- "a cat chased the dog" → Jaccard 0.000; SPMA: VP retrieval by `chased`; attention: noise
- "the cat sat a cat" → Jaccard 0.111; one index overlap by chance
- "a cat climbed the mat" → Jaccard 0.000; SPMA: VP retrieval by `climbed`; attention: noise

## Conclusion

SPMA retrieval is structurally plausible. Attention with random embeddings is noise. Genuine null: random embeddings carry no grammar signal. Result validates the fixed comparison framework before V10 introduces semantic embeddings.

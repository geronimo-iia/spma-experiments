# attn-spma-random — Null baseline (random embeddings)

Fixes the cardinality bug from attn-spma-bugged. Both SPMA and attention return
exactly top-5 sentence indices — Jaccard is meaningful.

## Hypothesis

Random embeddings carry no grammar signal → attention retrieval is noise →
Jaccard ≈ 0. Establishes the floor before testing semantic embeddings (attn-spma-minilm).

## Result

```
Mean Jaccard: 0.027  Std: 0.047  Min: 0.000  Max: 0.111
```

Spot-checks:
- "a cat chased the dog" → 0.000; SPMA: VP by `chased`; attention: noise
- "the cat sat a cat" → 0.111; one overlap by chance
- "a cat climbed the mat" → 0.000; SPMA: VP by `climbed`; attention: noise

## Conclusion

Confirmed null. Random embeddings produce near-zero Jaccard. SPMA retrieval is
structurally plausible — retrieves by highest-specificity VP constituent, not
surface overlap. Baseline for attn-spma-minilm: must exceed 0.027 to show
semantic embeddings add signal.

## Reproduce

```bash
cargo run --release
# outputs compare_attention_results.csv
```

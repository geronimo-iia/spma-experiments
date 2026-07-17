# attn-spma-bugged — Invalid baseline (do not use)

**Superseded by attn-spma-random.** Preserved as artifact only — results are meaningless.

## Hypothesis

SPMA beam search and scaled dot-product attention retrieve the same training
sentences for a given query. Measured via Jaccard similarity.

## Result

```
Mean Jaccard: 0.068  Std: 0.112
```

## Why invalid

**Bug 1 — key space mismatch**: SPMA returns grammar pattern IDs; attention
returns sentence indices. Jaccard over different-typed sets is noise.

**Bug 2 — cardinality mismatch**: SPMA `old_pattern_indices` spans all beam
alignments (~120 indices); attention returns exactly 5. A set of 5 cannot
overlap meaningfully with a set of 120 — Jaccard ≈ 0.04 by construction.

Both bugs fixed in attn-spma-random.

## Reproduce

```bash
cargo run --release
# outputs v8_results.csv
```

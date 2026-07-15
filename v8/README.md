# V8 — Original harness (invalid)

## Purpose

First attempt at comparing SPMA beam search retrieval with scaled dot-product attention.
Hypothesis: both sides retrieve the same training sentences for a given query.

Contains two bugs that make the comparison meaningless — preserved as an artifact.

## Run

```bash
cargo run --release
# outputs v8_results.csv
```

## Result

```
Mean Jaccard: 0.068  Std: 0.112
```

## Bugs

**V8a — key space mismatch**: SPMA returns grammar pattern IDs; attention returns sentence indices.
Jaccard over different-typed sets is noise. Inflated Jaccard (0.068) is an artifact.

**V8b — cardinality mismatch**: After fixing key space, SPMA `old_pattern_indices` spans all beam
alignments (~120 indices); attention returns exactly 5. Jaccard ≈ 0.04 by construction — a set
of 5 cannot overlap much with a set of 120.

## Conclusion

Invalid experiment. Both bugs fixed in V9.

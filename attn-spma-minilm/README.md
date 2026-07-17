# attn-spma-minilm — Semantic embedding baseline (MiniLM-L6-v2)

Replaces random embeddings with `all-MiniLM-L6-v2` (384-dim, L2-normalised,
mean-pooled via ONNX Runtime). Tests whether real semantic embeddings improve
SPMA-attention overlap vs the random null baseline (0.027).

## Hypothesis

Semantic embeddings should push Jaccard significantly above 0.027 if the
SPMA-attention analogy holds.

## Result

```
Mean Jaccard: 0.022  Std: 0.044  Min: 0.000  Max: 0.111
Delta vs random baseline: -0.004
```

Spot-checks:
- "a cat chased the dog" → 0.111; one overlap (train[42])
- "the cat sat a cat" → 0.000; SPMA retrieves `climbed` VP; attention retrieves surface `sat`+`cat`
- "a cat climbed the mat" → 0.000; same retrieval divergence

## Conclusion

MiniLM scores **below** the random baseline (−0.004, not significant). CFG
vocabulary is 11 tokens — MiniLM collapses to lexical surface overlap, retrieving
near-duplicates. SPMA retrieves by highest-specificity VP constituent. Different
retrieval objectives — no evidence for the SPMA-attention analogy.

## Reproduce

Edit `src/main.rs` line 69 to point at a local `all-MiniLM-L6-v2` directory
containing `tokenizer.json` and `onnx/model.onnx`:

```bash
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
    --local-dir /path/to/all-MiniLM-L6-v2 \
    --exclude "*.msgpack" "*.h5" "flax_model*" "tf_model*"

cargo run --release
# outputs attention_comparison.csv
```

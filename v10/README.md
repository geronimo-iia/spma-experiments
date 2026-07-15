# V10 — SPMA vs MiniLM-L6-v2 semantic embeddings

## Purpose

Semantic embedding experiment. Replaces V9's random embeddings with `all-MiniLM-L6-v2` (384-dim, L2-normalised, mean-pooled) via ONNX Runtime. Tests whether real semantic embeddings improve attention/SPMA overlap vs the random null baseline.

If the SPMA-attention analogy holds, semantic embeddings should push Jaccard significantly above the V9 baseline (0.027).

## Model setup

### Locate the model

Edit `src/main.rs` line 69:

```rust
const MODEL_DIR: &str = "/path/to/your/all-MiniLM-L6-v2";
```

The directory must contain:
- `tokenizer.json`
- `onnx/model.onnx`

### Download the model

```bash
pip install huggingface_hub

python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir="/path/to/your/all-MiniLM-L6-v2",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
)
EOF
```

Or with the CLI:

```bash
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
    --local-dir /path/to/your/all-MiniLM-L6-v2 \
    --exclude "*.msgpack" "*.h5" "flax_model*" "tf_model*"
```

Only `tokenizer.json` and `onnx/model.onnx` are required at runtime.

## Run

```bash
# update MODEL_DIR in src/main.rs first
cargo run --release
# outputs attention_comparison.csv
# expected: mean Jaccard ≈ 0.022
```

## Result

```
Mean Jaccard: 0.022  Std: 0.044  Min: 0.000  Max: 0.111
Delta vs V9 random baseline: -0.004
```

Spot-checks:
- "a cat chased the dog" → Jaccard 0.111; one overlap (train[42])
- "the cat sat a cat" → Jaccard 0.000; SPMA retrieves `climbed` VP; attention retrieves exact `sat`+`cat` surface matches
- "a cat climbed the mat" → Jaccard 0.000; same retrieval divergence

## Conclusion

MiniLM scores below random baseline (-0.004, not significant). CFG vocabulary is 11 tokens — MiniLM collapses to lexical surface overlap, retrieving near-duplicates. SPMA retrieves by highest-specificity VP constituent. Different retrieval objectives; no evidence for the analogy.

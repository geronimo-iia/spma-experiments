//! V10: Compare SPMA beam search retrieval with scaled dot-product attention.
//!
//! Embeddings: all-MiniLM-L6-v2 (384-dim) via ONNX Runtime.
//! Both sides retrieve exactly 5 train sentence indices per query.
//! SPMA: top-5 by max specificity of matched Old patterns.
//! Attention: top-5 by dot-product score / sqrt(384).

use anyhow::{Context, Result};
use ndarray::Array2;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use spma::{beam_search, Pattern, Symbol, SpmaEngine as SP71Comprehensive};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use tokenizers::Tokenizer;

// ---------------------------------------------------------------------------
// LCG (for dataset generation only)
// ---------------------------------------------------------------------------

fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

// ---------------------------------------------------------------------------
// Grammar sentence generation
// S → NP VP, NP → det noun, VP → verb NP
// det={the,a} noun={cat,dog,mat,bird,tree} verb={sat,chased,saw,climbed}
// ---------------------------------------------------------------------------

const DETS: &[&str] = &["the", "a"];
const NOUNS: &[&str] = &["cat", "dog", "mat", "bird", "tree"];
const VERBS: &[&str] = &["sat", "chased", "saw", "climbed"];

fn generate_sentences(n: usize) -> Vec<Vec<String>> {
    let mut state: u64 = 42;
    let mut sentences = Vec::with_capacity(n);
    for _ in 0..n {
        state = lcg_next(state);
        let det1 = DETS[(state % 2) as usize];
        state = lcg_next(state);
        let noun1 = NOUNS[(state % 5) as usize];
        state = lcg_next(state);
        let verb = VERBS[(state % 4) as usize];
        state = lcg_next(state);
        let det2 = DETS[(state % 2) as usize];
        state = lcg_next(state);
        let noun2 = NOUNS[(state % 5) as usize];
        sentences.push(vec![
            det1.to_owned(),
            noun1.to_owned(),
            verb.to_owned(),
            det2.to_owned(),
            noun2.to_owned(),
        ]);
    }
    sentences
}

// ---------------------------------------------------------------------------
// MiniLM embedding pipeline
// ---------------------------------------------------------------------------

const MODEL_DIR: &str = "/Users/geronimo/dev/mancie/models/all-MiniLM-L6-v2";
const EMB_DIM: usize = 384;

struct MiniLM {
    tokenizer: Tokenizer,
    session: std::cell::RefCell<Session>,
}

impl MiniLM {
    fn load() -> Result<Self> {
        let tok_path = format!("{MODEL_DIR}/tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;

        let model_path = format!("{MODEL_DIR}/onnx/model.onnx");
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("ort builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("opt level: {e}"))?
            .commit_from_file(&model_path)
            .map_err(|e| anyhow::anyhow!("load onnx: {e}"))?;

        Ok(Self { tokenizer, session: std::cell::RefCell::new(session) })
    }

    fn embed(&self, text: &str) -> Result<[f32; EMB_DIM]> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let type_ids: Vec<i64> = encoding
            .get_type_ids()
            .iter()
            .map(|&x| x as i64)
            .collect();

        let seq_len = ids.len();

        let ids_tensor = Tensor::<i64>::from_array(([1, seq_len], ids))
            .context("ids tensor")?;
        let mask_tensor = Tensor::<i64>::from_array(([1, seq_len], mask.clone()))
            .context("mask tensor")?;
        let type_ids_tensor = Tensor::<i64>::from_array(([1, seq_len], type_ids))
            .context("type_ids tensor")?;

        let mut sess = self.session.borrow_mut();
        let outputs = sess
            .run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
                "token_type_ids" => type_ids_tensor
            ])
            .map_err(|e| anyhow::anyhow!("onnx run: {e}"))?;

        // last_hidden_state: [1, seq_len, 384]
        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("extract tensor: {e}"))?;
        // data is flat [1 * seq_len * 384]
        let token_embs_2d = Array2::from_shape_vec((seq_len, EMB_DIM), data.to_vec())
            .context("reshape token embeddings")?;

        let pooled = mean_pool(&token_embs_2d, &mask);
        Ok(l2_normalize(pooled))
    }
}

fn mean_pool(token_embeddings: &Array2<f32>, attention_mask: &[i64]) -> Vec<f32> {
    let mask_f: Vec<f32> = attention_mask.iter().map(|&m| m as f32).collect();
    let sum_mask = mask_f.iter().sum::<f32>().max(1e-9);
    let dim = token_embeddings.ncols();
    let mut result = vec![0.0f32; dim];
    for (i, row) in token_embeddings.rows().into_iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[j] += val * mask_f[i];
        }
    }
    result.iter_mut().for_each(|x| *x /= sum_mask);
    result
}

fn l2_normalize(mut v: Vec<f32>) -> [f32; EMB_DIM] {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter_mut().for_each(|x| *x /= norm);
    let mut arr = [0.0f32; EMB_DIM];
    arr.copy_from_slice(&v);
    arr
}

fn dot_f32(a: &[f32; EMB_DIM], b: &[f32; EMB_DIM]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x as f64 * y as f64).sum()
}

// ---------------------------------------------------------------------------
// Jaccard
// ---------------------------------------------------------------------------

fn jaccard(a: &HashSet<usize>, b: &HashSet<usize>) -> f64 {
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 { 1.0 } else { intersection as f64 / union as f64 }
}

// ---------------------------------------------------------------------------
// Subsequence check
// ---------------------------------------------------------------------------

fn is_subsequence(pattern: &[u32], sentence: &[u32]) -> bool {
    let mut pat_idx = 0;
    for &sym in sentence {
        if pat_idx < pattern.len() && sym == pattern[pat_idx] {
            pat_idx += 1;
        }
    }
    pat_idx == pattern.len()
}

// ---------------------------------------------------------------------------
// SPMA top-5 by pattern specificity
// ---------------------------------------------------------------------------

fn spma_top5_by_specificity(
    test_ids: &[u32],
    train_id_vecs: &[Vec<u32>],
    grammar_id_vecs: &[Vec<u32>],
    costs: &[f64],
    k: usize,
) -> Vec<usize> {
    let beam_results = beam_search(test_ids, grammar_id_vecs, k, costs);
    let best = match beam_results.into_iter().next() {
        Some(a) => a,
        None => return vec![],
    };

    let matched_pattern_ids = best.matched_old_pattern_ids();

    let mut sentence_scores: HashMap<usize, f64> = HashMap::new();
    for pid in matched_pattern_ids {
        let pattern_syms = &grammar_id_vecs[pid];
        let specificity: f64 = pattern_syms.iter().map(|&id| costs[id as usize]).sum();
        for (train_idx, train_sent) in train_id_vecs.iter().enumerate() {
            if is_subsequence(pattern_syms, train_sent) {
                let entry = sentence_scores.entry(train_idx).or_insert(0.0);
                if specificity > *entry {
                    *entry = specificity;
                }
            }
        }
    }

    let mut ranked: Vec<(usize, f64)> = sentence_scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.into_iter().take(5).map(|(idx, _)| idx).collect()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    // 1. Load MiniLM
    println!("Loading all-MiniLM-L6-v2...");
    let minilm = MiniLM::load().context("failed to load MiniLM")?;
    println!("Embedding: all-MiniLM-L6-v2 (384-dim, ONNX)");

    // Verify first embedding is unit vector
    let probe = minilm.embed("the cat sat a dog")?;
    let norm: f32 = probe.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("Probe L2 norm: {norm:.6} (should be ≈1.0)");
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "L2 norm not ≈1.0: {norm}"
    );

    // 2. Generate dataset
    let all_sentences = generate_sentences(200);
    let train_sentences = &all_sentences[..150];
    let test_sentences = &all_sentences[150..];

    // 3. Build SP71
    let mut sp = SP71Comprehensive::new();

    let train_patterns: Vec<Pattern> = train_sentences
        .iter()
        .enumerate()
        .map(|(i, words)| {
            let symbols: Vec<Symbol> = words
                .iter()
                .map(|w| Symbol::new(sp.interner.intern(w)))
                .collect();
            Pattern::new(symbols, i as u32)
        })
        .collect();

    for words in test_sentences {
        for w in words {
            sp.interner.intern(w);
        }
    }

    let _results = sp.learn(train_patterns)?;

    // 4. Costs
    let needed = sp.interner.len();
    let mut costs = vec![1.0f64; needed];
    for p in &sp.old_patterns {
        for s in &p.symbols {
            if (s.name as usize) < costs.len() {
                costs[s.name as usize] = s.bit_cost;
            }
        }
    }

    // 5. ID vecs
    let train_id_vecs: Vec<Vec<u32>> = train_sentences
        .iter()
        .map(|words| words.iter().map(|w| sp.interner.intern(w)).collect())
        .collect();

    let grammar_id_vecs: Vec<Vec<u32>> = sp
        .old_patterns
        .iter()
        .filter(|p| p.symbols.len() >= 2)
        .map(|p| p.symbols.iter().map(|s| s.name).collect())
        .collect();

    println!("Grammar patterns (multi-symbol): {}", grammar_id_vecs.len());

    // 6. Compute MiniLM embeddings for all train sentences
    println!("Computing train embeddings...");
    let train_embs: Vec<[f32; EMB_DIM]> = train_sentences
        .iter()
        .map(|words| minilm.embed(&words.join(" ")))
        .collect::<Result<Vec<_>>>()?;

    let k = 5usize;
    let scale = (EMB_DIM as f64).sqrt();

    // 7. Evaluate
    let mut rows: Vec<(usize, Vec<usize>, Vec<usize>, f64)> = Vec::new();

    for (qi, words) in test_sentences.iter().enumerate() {
        let test_ids: Vec<u32> = words.iter().map(|w| sp.interner.intern(w)).collect();
        let text = words.join(" ");

        let spma_top5 = spma_top5_by_specificity(
            &test_ids,
            &train_id_vecs,
            &grammar_id_vecs,
            &costs,
            k,
        );
        let spma_hs: HashSet<usize> = spma_top5.iter().copied().collect();

        let query_emb = minilm.embed(&text)?;
        let mut scores: Vec<(usize, f64)> = train_embs
            .iter()
            .enumerate()
            .map(|(j, key_emb)| (j, dot_f32(&query_emb, key_emb) / scale))
            .collect();
        scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        let attn_top5: Vec<usize> = scores.iter().take(k).map(|&(j, _)| j).collect();
        let attn_hs: HashSet<usize> = attn_top5.iter().copied().collect();

        let j = jaccard(&spma_hs, &attn_hs);
        let mut spma_sorted = spma_top5.clone();
        spma_sorted.sort_unstable();
        rows.push((qi, spma_sorted, attn_top5, j));
    }

    // 8. Write CSV
    let mut f = File::create("attention_comparison.csv")?;
    writeln!(f, "test_idx,spma_top5,attn_top5,jaccard")?;
    for (qi, spma, attn, j) in &rows {
        let spma_str: Vec<String> = spma.iter().map(|x| x.to_string()).collect();
        let attn_str: Vec<String> = attn.iter().map(|x| x.to_string()).collect();
        writeln!(
            f,
            "{},\"{}\",\"{}\",{:.6}",
            qi,
            spma_str.join(","),
            attn_str.join(","),
            j
        )?;
    }

    // 9. Statistics
    let jaccards: Vec<f64> = rows.iter().map(|(_, _, _, j)| *j).collect();
    let n = jaccards.len() as f64;
    let mean = jaccards.iter().sum::<f64>() / n;
    let variance = jaccards.iter().map(|j| (j - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    let min_j = jaccards.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_j = jaccards.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let interpretation = if mean > 0.5 {
        "strong evidence for SPMA-attention analogy"
    } else if mean >= 0.2 {
        "partial evidence for SPMA-attention analogy"
    } else {
        "no evidence for SPMA-attention analogy (against)"
    };

    println!("\nMean Jaccard: {mean:.3}  Std: {std_dev:.3}  Min: {min_j:.3}  Max: {max_j:.3}");
    println!("Interpretation: {interpretation}");
    println!("Baseline (random embeddings V9): 0.026");
    println!("Delta: {:+.3}", mean - 0.026);

    // 10. Spot-checks
    println!("\n=== Spot-checks: 3 test sentences ===");
    for qi in [0, 1, 2] {
        let (_, spma, attn, j) = &rows[qi];
        let test_words = &test_sentences[qi];
        println!("\nTest[{qi}]: {}", test_words.join(" "));
        println!("  SPMA top-5:");
        for &sidx in spma {
            println!("    train[{sidx:3}]: {}", train_sentences[sidx].join(" "));
        }
        println!("  Attn top-5:");
        for &sidx in attn {
            println!("    train[{sidx:3}]: {}", train_sentences[sidx].join(" "));
        }
        println!("  Jaccard: {j:.3}");
    }

    let spma_exact5 = rows.iter().filter(|(_, spma, _, _)| spma.len() == 5).count();
    println!("\nSPMA cardinality: {spma_exact5}/50 queries returned exactly 5 sentences");

    Ok(())
}

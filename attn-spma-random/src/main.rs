//! V9: Compare SPMA beam search retrieval with scaled dot-product attention.
//!
//! Null baseline: embeddings are random (deterministic LCG per token ID).
//! Both sides retrieve exactly 5 train sentence indices per query.
//! SPMA: top-5 by max specificity of matched Old patterns.
//! Attention: top-5 by dot-product score / sqrt(EMB_DIM).
//!
//! Expected result: mean Jaccard ≈ 0.026 (random embeddings carry no signal).

use anyhow::Result;
use spma::{beam_search, Pattern, Symbol, SpmaEngine as SP71Comprehensive};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;

// ---------------------------------------------------------------------------
// LCG
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
// Random embeddings (deterministic per sentence index — null baseline)
// ---------------------------------------------------------------------------

const EMB_DIM: usize = 64;

fn random_emb(seed: u64) -> [f32; EMB_DIM] {
    let mut s = seed.wrapping_add(1).wrapping_mul(6_364_136_223_846_793_005);
    let mut arr = [0.0f32; EMB_DIM];
    for x in &mut arr {
        s = lcg_next(s);
        *x = (s >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0;
    }
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
    // 1. Generate dataset
    let all_sentences = generate_sentences(200);
    let train_sentences = &all_sentences[..150];
    let test_sentences = &all_sentences[150..];

    // 2. Train SPMA
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

    // 3. Costs
    let needed = sp.interner.len();
    let mut costs = vec![1.0f64; needed];
    for p in &sp.old_patterns {
        for s in &p.symbols {
            if (s.name as usize) < costs.len() {
                costs[s.name as usize] = s.bit_cost;
            }
        }
    }

    // 4. ID vecs
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
    println!("Embedding: random LCG {}d (null baseline)", EMB_DIM);

    // 5. Random embeddings for train sentences (seed = sentence index)
    let train_embs: Vec<[f32; EMB_DIM]> = (0..train_sentences.len())
        .map(|i| random_emb(i as u64))
        .collect();

    let k = 5usize;
    let scale = (EMB_DIM as f64).sqrt();

    // 6. Evaluate
    let mut rows: Vec<(usize, Vec<usize>, Vec<usize>, f64)> = Vec::new();

    for (qi, words) in test_sentences.iter().enumerate() {
        let test_ids: Vec<u32> = words.iter().map(|w| sp.interner.intern(w)).collect();

        let spma_top5 = spma_top5_by_specificity(
            &test_ids,
            &train_id_vecs,
            &grammar_id_vecs,
            &costs,
            k,
        );
        let spma_hs: HashSet<usize> = spma_top5.iter().copied().collect();

        // Query embedding: random seed = 150 + qi (offset from train seeds)
        let query_emb = random_emb(150 + qi as u64);
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

    // 7. Write CSV
    let mut f = File::create("compare_attention_results.csv")?;
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

    // 8. Statistics
    let jaccards: Vec<f64> = rows.iter().map(|(_, _, _, j)| *j).collect();
    let n = jaccards.len() as f64;
    let mean = jaccards.iter().sum::<f64>() / n;
    let variance = jaccards.iter().map(|j| (j - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    let min_j = jaccards.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_j = jaccards.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("\nMean Jaccard: {mean:.3}  Std: {std_dev:.3}  Min: {min_j:.3}  Max: {max_j:.3}");
    println!("Interpretation: no evidence for SPMA-attention analogy (random baseline)");

    // 9. Spot-checks
    println!("\n=== Spot-checks: 3 test sentences ===");
    for qi in [0, 1, 2] {
        let (_, spma, attn, j) = &rows[qi];
        let test_words = &test_sentences[qi];
        println!("\nTest[{qi}]: {}", test_words.join(" "));
        println!("  SPMA top-5: {spma:?}");
        println!("  Attn top-5: {attn:?}");
        println!("  Jaccard: {j:.3}");
    }

    Ok(())
}

//! V8: Original compare_attention harness (INVALID — two bugs, archived).
//!
//! Bug V8a: SPMA returns grammar pattern IDs (not sentence indices).
//!          Attention returns sentence indices. Jaccard over different-typed sets is noise.
//!
//! Bug V8b: After fixing key space — SPMA still returns old_pattern_indices from all beam
//!          alignments, yielding ~120 indices vs attention's top-5.
//!          Set cardinality too unequal for meaningful Jaccard.
//!
//! Adapted from git commit 31eab51 (sp71_rust). API updated to current spma crate.
//! Algorithm and bugs are unchanged.

use anyhow::Result;
use spma::{beam_search, Pattern, Symbol, SpmaEngine as SP71Comprehensive};
use std::collections::HashSet;
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
// Grammar sentence generation (smaller vocabulary than V9/V10)
// det={the,a} noun={cat,dog,mat} verb={sat,chased}
// ---------------------------------------------------------------------------

const DETS: &[&str] = &["the", "a"];
const NOUNS: &[&str] = &["cat", "dog", "mat"];
const VERBS: &[&str] = &["sat", "chased"];

fn generate_sentences(n: usize) -> Vec<Vec<String>> {
    let mut state: u64 = 42;
    let mut sentences = Vec::with_capacity(n);
    for _ in 0..n {
        state = lcg_next(state);
        let det1 = DETS[(state % 2) as usize];
        state = lcg_next(state);
        let noun1 = NOUNS[(state % 3) as usize];
        state = lcg_next(state);
        let verb = VERBS[(state % 2) as usize];
        state = lcg_next(state);
        let det2 = DETS[(state % 2) as usize];
        state = lcg_next(state);
        let noun2 = NOUNS[(state % 3) as usize];
        sentences.push(vec![
            det1.to_owned(),
            noun1.to_owned(),
            verb.to_owned(),
            det2.to_owned(),
            noun2.to_owned(),
        ])
    }
    sentences
}

// ---------------------------------------------------------------------------
// Embedding helpers (random, deterministic per token ID)
// ---------------------------------------------------------------------------

const DIM: usize = 16;

fn token_embedding(token_id: u32) -> [f64; DIM] {
    let mut state = (token_id as u64)
        .wrapping_add(1)
        .wrapping_mul(6_364_136_223_846_793_005);
    let mut emb = [0.0f64; DIM];
    for e in &mut emb {
        state = lcg_next(state);
        *e = (state as f64) / (u64::MAX as f64) * 2.0 - 1.0;
    }
    emb
}

fn mean_embedding(token_ids: &[u32]) -> [f64; DIM] {
    let mut result = [0.0f64; DIM];
    for &id in token_ids {
        let emb = token_embedding(id);
        for (r, e) in result.iter_mut().zip(emb.iter()) {
            *r += e;
        }
    }
    let n = token_ids.len() as f64;
    for r in &mut result {
        *r /= n;
    }
    result
}

fn dot_product(a: &[f64; DIM], b: &[f64; DIM]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// Jaccard
// ---------------------------------------------------------------------------

fn jaccard(a: &HashSet<usize>, b: &HashSet<usize>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
}

// ---------------------------------------------------------------------------
// Build costs vector from a trained engine
// ---------------------------------------------------------------------------

fn build_costs(sp: &SP71Comprehensive) -> Vec<f64> {
    let max_id = sp.interner.len();
    let mut costs = vec![1.0f64; max_id];
    for p in &sp.old_patterns {
        for s in &p.symbols {
            if (s.name as usize) < costs.len() {
                costs[s.name as usize] = s.bit_cost;
            }
        }
    }
    costs
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let all_sentences = generate_sentences(200);
    let train_sentences = &all_sentences[..150];
    let test_sentences = &all_sentences[150..];

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

    let costs = build_costs(&sp);

    let train_id_vecs: Vec<Vec<u32>> = train_sentences
        .iter()
        .map(|words| words.iter().map(|w| sp.interner.intern(w)).collect())
        .collect();

    let train_key_embs: Vec<[f64; DIM]> = train_id_vecs
        .iter()
        .map(|ids| mean_embedding(ids))
        .collect();

    let mut rows: Vec<(usize, Vec<usize>, Vec<usize>, f64)> = Vec::new();
    let k = 5usize;
    let scale = (DIM as f64).sqrt();

    for (qi, words) in test_sentences.iter().enumerate() {
        let test_ids: Vec<u32> = words.iter().map(|w| sp.interner.intern(w)).collect();

        let needed = sp.interner.len();
        let mut local_costs = costs.clone();
        while local_costs.len() < needed {
            local_costs.push(1.0);
        }

        // BUG V8a/V8b: old_pattern_indices are grammar pattern IDs, not sentence indices.
        // Cardinality also uncontrolled — may return far more than k entries.
        let beam_results = beam_search(&test_ids, &train_id_vecs, k, &local_costs);

        let mut spma_set: Vec<usize> = Vec::new();
        let mut seen: HashSet<usize> = HashSet::new();
        for alignment in &beam_results {
            for &idx in &alignment.old_pattern_indices {
                if seen.insert(idx) {
                    spma_set.push(idx);
                }
                if spma_set.len() >= k {
                    break;
                }
            }
            if spma_set.len() >= k {
                break;
            }
        }

        let query_emb = mean_embedding(&test_ids);
        let mut scores: Vec<(usize, f64)> = train_key_embs
            .iter()
            .enumerate()
            .map(|(j, key_emb)| (j, dot_product(&query_emb, key_emb) / scale))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let attn_set: Vec<usize> = scores.iter().take(k).map(|&(j, _)| j).collect();

        let spma_hs: HashSet<usize> = spma_set.iter().copied().collect();
        let attn_hs: HashSet<usize> = attn_set.iter().copied().collect();
        let j = jaccard(&spma_hs, &attn_hs);

        rows.push((qi, spma_set, attn_set, j));
    }

    let mut f = File::create("v8_results.csv")?;
    writeln!(f, "query_id,spma_retrievals,attn_retrievals,jaccard")?;
    for (qi, spma, attn, j) in &rows {
        let spma_str: Vec<String> = spma.iter().map(|x| x.to_string()).collect();
        let attn_str: Vec<String> = attn.iter().map(|x| x.to_string()).collect();
        writeln!(f, "{},{},{},{:.6}", qi, spma_str.join(";"), attn_str.join(";"), j)?;
    }

    let jaccards: Vec<f64> = rows.iter().map(|(_, _, _, j)| *j).collect();
    let n = jaccards.len() as f64;
    let mean = jaccards.iter().sum::<f64>() / n;
    let variance = jaccards.iter().map(|j| (j - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    println!("Mean Jaccard: {mean:.3}  Std: {std_dev:.3}");
    println!("NOTE: result is meaningless — see module-level doc for bug description.");

    Ok(())
}

#!/usr/bin/env python3
"""
Print a human-readable summary of an SPMA grammar for LLM analysis.

Usage:
    python grammar_summary.py data/model/hdfs_base.json
    python grammar_summary.py data/model/hdfs_base.json data/model/hdfs_full.json
"""

import json
import sys
from collections import defaultdict

HDFS_EVENTS = {
    "E1":  "PacketResponder start",
    "E2":  "PacketResponder termination",
    "E3":  "Got exception while serving",
    "E4":  "Exception in receiveBlock",
    "E5":  "Receiving block",
    "E6":  "Received block (dest side)",
    "E7":  "Served block",
    "E8":  "Replicating block",
    "E9":  "Received block (src side)",
    "E10": "Asking to recover block",
    "E11": "Received block of size",
    "E12": "DataXceiver error",
    "E13": "Deleting block (local)",
    "E14": "Verification succeeded",
    "E15": "Exception closing socket",
    "E16": "Exception writing",
    "E17": "IOException",
    "E18": "Receiving empty packet",
    "E19": "Connection reset",
    "E20": "BlockReport",
    "E21": "Deleting block (invalid)",
    "E22": "allocateBlock",
    "E23": "addStoredBlock (reportedBlock)",
    "E24": "addStoredBlock (received)",
    "E25": "PendingReplicationMonitor timeout",
    "E26": "addStoredBlock (stored)",
    "E27": "Unexpected exception",
    "E28": "DataStreamer exception",
    "E29": "Recovering block",
    "E30": "Block is corrupt",
}


def sym_name(sym, names):
    if "Atom" in sym:
        idx = sym["Atom"]
        label = names[idx] if idx < len(names) else f"#{idx}"
        return f"{label}({HDFS_EVENTS.get(label, '?')})"
    elif "Pattern" in sym:
        return f"P{sym['Pattern']}"
    return str(sym)


def gap_str(gaps, idx):
    if not gaps or idx >= len(gaps):
        return "→"
    g = gaps[idx]
    if g["min"] == 0 and g["max"] == 0:
        return "→"
    return f"~[{g['min']},{g['max']}]→"


def render_pattern(pat, names):
    syms = pat["symbols"]
    gaps = pat.get("gaps", [])
    parts = []
    for i, sym in enumerate(syms):
        parts.append(sym_name(sym, names))
        if i < len(syms) - 1:
            parts.append(gap_str(gaps, i))
    return " ".join(parts)


def summarize(path):
    g = json.load(open(path))
    names = g["grammar"]["interner"]["names"]
    levels = g["grammar"]["levels"]
    atom_costs = g.get("atom_costs", [])
    beam_k = g.get("beam_k", "?")
    max_gap = g.get("max_induced_gap", "?")

    print(f"{'='*70}")
    print(f"Model: {path}")
    print(f"Beam: {beam_k}  MaxGap: {max_gap}")
    print(f"Vocabulary ({len(names)} atoms): {', '.join(names)}")
    print()

    # Atom costs
    if atom_costs:
        print("Atom costs (lower = more frequent = cheaper to cover):")
        atom_cost_pairs = [(names[i], atom_costs[i]) for i in range(min(len(names), len(atom_costs)))]
        atom_cost_pairs.sort(key=lambda x: x[1])
        for name, cost in atom_cost_pairs:
            bar = "█" * max(1, int(cost * 4))
            desc = HDFS_EVENTS.get(name, "?")
            print(f"  {name:4s} {cost:6.3f} {bar:20s} {desc}")
        print()

    # Atoms never covered (not in any pattern)
    covered_atoms = set()
    for level in levels:
        for pat in level["patterns"]:
            for sym in pat["symbols"]:
                if "Atom" in sym:
                    covered_atoms.add(sym["Atom"])
    uncovered = [names[i] for i in range(len(names)) if i not in covered_atoms]
    if uncovered:
        print(f"Uncovered atoms (always contribute to e_cost): {', '.join(uncovered)}")
        for e in uncovered:
            print(f"  {e}: {HDFS_EVENTS.get(e, '?')}")
        print()

    total_patterns = sum(len(lvl["patterns"]) for lvl in levels)
    print(f"Grammar: {len(levels)} levels, {total_patterns} total patterns")
    print()

    for lvl_idx, level in enumerate(levels):
        pats = level["patterns"]
        if not pats:
            continue
        total_freq = sum(p["frequency"] for p in pats)
        top_n = sorted(pats, key=lambda p: p["frequency"], reverse=True)[:10]
        gap_count = sum(1 for p in pats if any(
            g["min"] > 0 or g["max"] > 0 for g in p.get("gaps", [])))

        print(f"Level {lvl_idx}: {len(pats)} patterns, {gap_count} gap patterns, total_freq={total_freq}")
        print(f"  Top patterns by frequency:")
        for pat in top_n:
            rendered = render_pattern(pat, names)
            freq_pct = pat["frequency"] / max(total_freq, 1) * 100
            print(f"    [freq={pat['frequency']:7d} {freq_pct:5.1f}%] {rendered}")
        print()

    # Per-level e_norm distribution summary
    dist = g["grammar"].get("e_distribution", {})
    level_dists = dist.get("level_sorted_e_norms", [])
    if level_dists:
        print("E_norm distribution per level (training sequences):")
        print(f"  {'level':>6}  {'n':>7}  {'p50':>7}  {'p90':>7}  {'p99':>7}  {'max':>7}")
        for i, vals in enumerate(level_dists):
            if not vals:
                continue
            n = len(vals)
            p50 = vals[int(n * 0.50)]
            p90 = vals[int(n * 0.90)]
            p99 = vals[int(n * 0.99)]
            mx  = vals[-1]
            print(f"  {i:>6}  {n:>7}  {p50:>7.4f}  {p90:>7.4f}  {p99:>7.4f}  {mx:>7.4f}")
        print()


def compare(path_a, path_b):
    ga = json.load(open(path_a))
    gb = json.load(open(path_b))
    names_a = set(ga["grammar"]["interner"]["names"])
    names_b = set(gb["grammar"]["interner"]["names"])

    print(f"\n{'='*70}")
    print(f"Comparison: {path_a}  vs  {path_b}")
    print(f"  Vocab A: {sorted(names_a)}")
    print(f"  Vocab B: {sorted(names_b)}")
    print(f"  Only in A: {sorted(names_a - names_b)}")
    print(f"  Only in B: {sorted(names_b - names_a)}")

    levels_a = ga["grammar"]["levels"]
    levels_b = gb["grammar"]["levels"]
    print(f"  Levels A: {len(levels_a)}  Levels B: {len(levels_b)}")
    pats_a = sum(len(l["patterns"]) for l in levels_a)
    pats_b = sum(len(l["patterns"]) for l in levels_b)
    print(f"  Patterns A: {pats_a}  Patterns B: {pats_b}  (diff: {pats_b - pats_a:+d})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    for path in sys.argv[1:]:
        summarize(path)

    if len(sys.argv) == 3:
        compare(sys.argv[1], sys.argv[2])

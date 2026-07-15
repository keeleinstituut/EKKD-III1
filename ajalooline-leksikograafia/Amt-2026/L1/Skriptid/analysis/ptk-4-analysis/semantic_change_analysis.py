#!/usr/bin/env python3
# Created: 2026-06-15 17-58-45
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
"""Semantic-change candidate finder for AMT-Master.

Scans the German equivalents (`*-de` columns) of every lemma across the six
historical dictionaries (Stahl 1637 → Hupel 1780) and flags lemmas whose gloss
shifts meaning over time. The core signal is an "endpoint shift": the earliest
and latest attesting source share no German head-noun.

Heuristic OVER-generates — it also catches orthographic modernisation
(münch → Mönch) and synonym swaps (Hirte → Viehhüter); the printed list is a
candidate pool for manual triage, not a final result. See
`Semantic-change_candidates_20260615.md` for the curated, hand-sorted set.

Usage:
    uv run python Katus-DRAFTS/Ptk-4/semantic_change_analysis.py [--min-sources N] [--detail ID ...]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Source JSON lives in Ptk-3, one folder up from this script's Ptk-4 location.
SRC = Path(__file__).resolve().parent.parent / "Ptk-3" / "AMT-Master_annotated.json"

# (display name, year, JSON-key prefix) in chronological order.
SOURCES = [
    ("Stahl",    1637, "Stahl-1637"),
    ("Gutslaff", 1648, "Gutslaff-1648"),
    ("Göseken",  1660, "Göseken-1660"),
    ("Vestring", 1715, "Vestring-17XX"),   # 17XX -> 1715 only for ordering
    ("Helle",    1732, "Helle-1732"),
    ("Hupel",    1780, "Hupel-1780-est-ger"),
]

# Placeholders that do NOT denote a real attestation (see Ptk-3 CSC correction).
PLACEHOLDERS = {"", "---", "NULL", "???"}

# German articles / noise tokens dropped before taking a phrase's head noun.
STOPWORDS = {"der", "die", "das", "ein", "eine", "den", "dem", "des", "st", "x"}


def head_nouns(gloss: str) -> set[str]:
    """Return the set of head nouns in a German gloss.

    Splits on clause/synonym separators, drops parenthetical/bracketed notes
    and articles, and keeps the LAST word of each phrase (the German head).
    """
    g = gloss.lower()
    g = re.sub(r"\([^)]*\)", " ", g)   # (r.), (d.) dialect notes
    g = re.sub(r"\[[^]]*\]", " ", g)   # [cross-references]
    g = re.sub(r"[^a-zäöüß, ;/.]", " ", g)
    heads: set[str] = set()
    for part in re.split(r"[,;/.]", g):
        words = [w for w in part.split() if w and w not in STOPWORDS]
        if words:
            heads.add(words[-1])
    return heads


def fuzzy_disjoint(a: set[str], b: set[str], prefix: int = 5) -> bool:
    """True if no head noun in `a` matches one in `b`.

    A shared 5-char prefix counts as a match, to suppress some orthographic
    noise (Schmid/Schmidt). Vowel-shift variants (münch/Mönch) still slip
    through — that residue is what manual triage removes.
    """
    for x in a:
        for y in b:
            if x == y or (len(x) >= prefix and len(y) >= prefix and x[:prefix] == y[:prefix]):
                return False
    return True


def attested_sequence(entry: dict) -> list[tuple[str, int, str, set[str]]]:
    """Chronological [(name, year, gloss, head_nouns)] of non-placeholder glosses."""
    seq = []
    for name, year, key in SOURCES:
        gloss = (entry.get(f"{key}-de") or "").strip()
        if gloss not in PLACEHOLDERS:
            seq.append((name, year, gloss, head_nouns(gloss)))
    return seq


def analyse(rows: list[dict], min_sources: int) -> list[dict]:
    """Return endpoint-shift candidates sorted by source count then divergence."""
    candidates = []
    for r in rows:
        seq = attested_sequence(r)
        if len(seq) < 2:
            continue
        early, late = seq[0], seq[-1]
        if not (early[3] and late[3] and fuzzy_disjoint(early[3], late[3])):
            continue
        all_heads = set().union(*(s[3] for s in seq))
        core = set.intersection(*(s[3] for s in seq))
        candidates.append({
            "id": r["Amt-Master-ID"],
            "n_sources": len(seq),
            "csc": r.get("Cross-source count"),
            "amt_cat": r.get("Amt-Cat"),
            "sem_cat": r.get("Sem-Cat"),
            "distinct_heads": len(all_heads),
            "core_shared": sorted(core),
            "early": f"{early[0]} {early[1]}: {early[2]}",
            "late": f"{late[0]} {late[1]}: {late[2]}",
            "trajectory": [(s[0], s[1], s[2]) for s in seq],
        })
    candidates.sort(key=lambda c: (-c["n_sources"], -c["distinct_heads"]))
    return [c for c in candidates if c["n_sources"] >= min_sources]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-sources", type=int, default=4,
                    help="minimum attesting sources for the candidate pool (default 4)")
    ap.add_argument("--detail", nargs="*", metavar="ID", default=[],
                    help="print the full chronological gloss trajectory for these lemma IDs")
    ap.add_argument("--json", action="store_true",
                    help="emit the candidate pool as JSON instead of a text table")
    args = ap.parse_args()

    rows = json.loads(SRC.read_text(encoding="utf-8"))["AMT-Master"]

    if args.detail:
        by_id = {r["Amt-Master-ID"]: r for r in rows}
        for lid in args.detail:
            r = by_id.get(lid)
            if r is None:
                print(f"## {lid}: NOT FOUND\n")
                continue
            print(f"## {lid}  [{r.get('Amt-Cat')} | {r.get('Sem-Cat')}]")
            print(f"   en: {r.get('DEF_en')}")
            for name, year, gloss, _ in attested_sequence(r):
                print(f"   {year} {name:9}: {gloss}")
            print()
        return

    pool = analyse(rows, args.min_sources)

    if args.json:
        # Drop the set-typed field so the result is JSON-serialisable.
        print(json.dumps(pool, ensure_ascii=False, indent=2))
        return

    total_multi = sum(1 for r in rows if len(attested_sequence(r)) >= 2)
    print(f"Multi-source lemmas (>=2): {total_multi}")
    print(f"Endpoint-shift pool (>= {args.min_sources} sources): {len(pool)}\n")
    for c in pool:
        print(f"# {c['id']}  (n={c['n_sources']}, CSC={c['csc']}, {c['amt_cat']}/{c['sem_cat']})")
        print(f"   EARLY  {c['early']}")
        print(f"   LATE   {c['late']}")
        print()


if __name__ == "__main__":
    main()

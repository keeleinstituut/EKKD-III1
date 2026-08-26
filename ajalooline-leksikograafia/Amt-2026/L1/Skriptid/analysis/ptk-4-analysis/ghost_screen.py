#!/usr/bin/env python3
# Created: 2026-06-16 11-49-31
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
# Co-Authored-By: Claude Fable 5
"""Ghost-profession screen for sub-chapter 4.3.

Tests every master lemma against a modern Estonian frequency list to find
"ghost professions" — roles recorded in the 1637–1780 dictionaries that have
since dropped out of (or shifted in) the language. The master `Amt-Master-ID`s
are already modern normalised lemmas, so they match the frequency list directly.

Output `lookup-needed.csv` is the worklist for the Sõnaveeb / ÜS API step:
every lemma that is absent, rare, or multi-word gets flagged `needs_lookup=Y`.
After the API returns, fold the verdicts back into the tagged JSON's `Hääbunud`
field (Y = obsolete / NIHE = meaning shifted / N = current).

Buckets (single-word lemmas):
    absent   not in the modern list at all      -> ghost candidate
    rare     present but freq < --rare-threshold -> possible shift / rarity
    current  freq >= threshold                   -> still in use
Multi-word lemmas are screened on their head noun (last word) and always
flagged for lookup (a frequency list cannot confirm a phrase).

Usage:
    uv run python Katus-DRAFTS/Ptk-4/ghost_screen.py [--rare-threshold 500]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Run from the repo root. 2026-08-10: repointed from the old Ptk-4 draft-folder
# copies to the canonical master; the frequency list stays in the archive.
MASTER = Path("Katus-ALUSANDMED/json-all/AMT-Master_annotated.json")
FREQ_FILE = Path("Katuse-failide-arhiiv/Ptk-4/lemmafreq-jan24.txt")
OUT_CSV = Path("scripts/ptk-4-analysis/lookup-needed.csv")


def load_rows() -> list[dict]:
    return json.loads(MASTER.read_text(encoding="utf-8"))["AMT-Master"]


def primary_form(amt_id: str) -> str:
    """First comma-variant of an ID (e.g. 'vabadik, vabandetu' -> 'vabadik')."""
    return amt_id.split(",")[0].strip().lower()


def build_freq(needed: set[str]) -> dict[str, int]:
    """Single pass over the ~20M-line frequency file; keep max freq per needed lemma."""
    freq: dict[str, int] = {}
    with FREQ_FILE.open(encoding="utf-8") as fh:
        next(fh, None)  # header: Lexeme<TAB>Frequency
        for line in fh:
            try:
                lex, f = line.rstrip("\n").split("\t")
            except ValueError:
                continue
            lemma = lex.rsplit("-", 1)[0].lower()  # strip -POS tag
            if lemma in needed:
                n = int(f)
                if n > freq.get(lemma, 0):
                    freq[lemma] = n
    return freq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rare-threshold", type=int, default=500,
                    help="freq below this (but present) counts as 'rare' -> lookup (default 500)")
    args = ap.parse_args()

    rows = load_rows()

    # Build the set of forms to look up: single-word primaries + multi-word head nouns.
    records = []
    needed: set[str] = set()
    for r in rows:
        amt_id = r["Amt-Master-ID"]
        primary = primary_form(amt_id)
        is_multi = " " in primary
        head = primary.split()[-1] if is_multi else primary
        needed.add(head)
        records.append({
            "id": amt_id, "primary": primary, "is_multi": is_multi, "head": head,
            "teema": r.get("Teema", ""), "def_en": r.get("DEF_en", ""),
        })

    freq = build_freq(needed)

    def bucket(rec) -> tuple[str, int | str]:
        f = freq.get(rec["head"])
        if f is None:
            return "absent", ""
        if f < args.rare_threshold:
            return "rare", f
        return "current", f

    out = []
    counts = {"absent": 0, "rare": 0, "current": 0, "multi": 0}
    for rec in records:
        b, f = bucket(rec)
        needs = (b in ("absent", "rare")) or rec["is_multi"]
        if rec["is_multi"]:
            counts["multi"] += 1
        else:
            counts[b] += 1
        out.append({
            "Amt-Master-ID": rec["id"],
            "type": "multi" if rec["is_multi"] else "single",
            "head_noun": rec["head"] if rec["is_multi"] else "",
            "modern_freq": f,
            "bucket": b,
            "needs_lookup": "Y" if needs else "N",
            "Teema": rec["teema"],
            "DEF_en": rec["def_en"],
        })

    # absent first, then rare (asc freq), then the rest — most interesting on top.
    order = {"absent": 0, "rare": 1, "current": 2}
    out.sort(key=lambda x: (x["needs_lookup"] != "Y", order.get(x["bucket"], 3),
                            x["modern_freq"] if isinstance(x["modern_freq"], int) else 0))

    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    total = len(out)
    lookup = sum(1 for x in out if x["needs_lookup"] == "Y")
    print(f"Screened {total} master lemmas against {FREQ_FILE.name}")
    print(f"  single-word: absent={counts['absent']}  rare(<{args.rare_threshold})={counts['rare']}  "
          f"current={counts['current']}")
    print(f"  multi-word (head-noun screen): {counts['multi']}")
    print(f"\n  needs_lookup = Y : {lookup}  ({lookup/total*100:.0f}%)  -> {OUT_CSV.name}")
    print(f"  needs_lookup = N : {total - lookup}  (current single-word lemmas)")
    print("\nNext: run lookup-needed.csv through Sõnaveeb/ÜS, then fold verdicts into")
    print("the tagged JSON 'Hääbunud' field (Y=obsolete / NIHE=shifted / N=current).")


if __name__ == "__main__":
    main()

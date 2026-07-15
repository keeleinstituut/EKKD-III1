#!/usr/bin/env python3
# Created: 2026-07-06 18-42-32
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
"""Build a review work file for the semantic-synonym sweep.

For every TSV row in tiers {comphead, fuzzy, verb, head, unmatched} (all letters,
including A-range verb rows the user left for us), emit:
  amt_lemma, tier, current candidates, DEF_et, DEF_en, def_tokens_in_YS
where def_tokens_in_YS = words of DEF_et/DEF_en(et tokens only) found in the ÜS
lemma list (homonym numbers stripped) — candidate pool for the synonym choice.
"""
import json
import re
from pathlib import Path

BASE = Path("/Users/q/dev/gen/Katus-DEV/Katus-ALUSANDMED/YS-Master-semcat-diff/Lemmas")
ys = set()
for ln in (BASE / "YS-lemmad_20260506.txt").open(encoding="utf-8"):
    w = re.sub(r" \d+$", "", ln.strip())
    if w:
        ys.add(w)
ys_lower = {w.lower() for w in ys}

amt = json.load((BASE / "AMT-Master_annotated_REVIEW.json").open(encoding="utf-8"))["AMT-Master"]
defs = {e["Amt-Master-ID"]: (e.get("DEF_et") or "", e.get("DEF_en") or "") for e in amt}

rows = [ln.rstrip("\n").split("\t") for ln in (BASE / "AMT-YS-match-report_20260706.tsv").open(encoding="utf-8")]
header, rows = rows[0], rows[1:]

PENDING = {"comphead", "fuzzy", "verb", "head", "unmatched"}
out = BASE / "semsyn-workfile.tsv"
n = 0
with out.open("w", encoding="utf-8") as f:
    f.write("amt_lemma\ttier\tcurrent\tDEF_et\tDEF_en\tdef_tokens_in_YS\n")
    for r in rows:
        lemma, tier = r[0], r[1]
        cur = r[2] if len(r) > 2 else ""
        if tier not in PENDING:
            continue
        det, den = defs.get(lemma, ("<NO ENTRY>", ""))
        toks = re.findall(r"[a-zA-ZäöüõšžÄÖÜÕŠŽ-]{4,}", det)
        hits = []
        for t in toks:
            tl = t.lower()
            if tl in ys_lower and tl not in hits:
                hits.append(tl)
        f.write("\t".join([lemma, tier, cur, det, den, ", ".join(hits)]) + "\n")
        n += 1
print(f"{n} pending rows -> {out}")
missing = [r[0] for r in rows if r[0] not in defs]
print(f"lemmas missing from REVIEW json: {len(missing)}", missing[:10])

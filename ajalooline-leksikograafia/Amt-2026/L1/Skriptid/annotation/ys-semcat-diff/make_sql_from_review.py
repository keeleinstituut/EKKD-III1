#!/usr/bin/env python3
# Created: 2026-07-06 19-29-27
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
"""Regenerate the semcat SQL query from the user-reviewed mapping CSV.

Input:  AMT-YS-match-report_20260706_read.csv (comma-separated, BOM, candidates
        "; "-separated) — this CSV is now the authoritative AMT→ÜS mapping.
Output: AMT-YS-semcat-query_20260706.sql with WHERE w.value IN (<all forms>).
Candidates are validated against the ÜS lemma list (homonym numbers stripped);
anything absent is reported and left out of the query.
"""
import csv
import json
import re
from pathlib import Path

BASE = Path("/Users/q/dev/gen/Katus-DEV/Katus-ALUSANDMED/YS-Master-semcat-diff/Lemmas")
CSV_IN = BASE / "AMT-YS-match-report_20260706_read.csv"

ys_set = set()
for ln in (BASE / "YS-lemmad_20260506.txt").open(encoding="utf-8"):
    w = re.sub(r" \d+$", "", ln.strip())
    if w:
        ys_set.add(w)
ys_ci = {}
for w in ys_set:
    ys_ci.setdefault(w.lower(), w)

forms, not_in_ys, blank = set(), [], []
with CSV_IN.open(encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        lemma = row["amt_lemma"].strip()
        cands = [c.strip() for c in (row["ys_candidates"] or "").split(";") if c.strip()]
        if not cands:
            blank.append((lemma, row["tier"]))
            continue
        for c in cands:
            if c in ys_set:
                forms.add(c)
            elif c.lower() in ys_ci:
                forms.add(ys_ci[c.lower()])
            else:
                not_in_ys.append((lemma, c))

example = BASE / "_SELECT_w_id_AS_word_id_w_value_AS_word_w_value_prese_AS_word_pr_202607061807.json"
sql_template = next(iter(json.load(example.open(encoding="utf-8"))))
in_list = ",\n  ".join("'" + f.replace("'", "''") + "'" for f in sorted(forms))
sql = sql_template.replace(
    "WHERE w.value IN ('abikaasa')",
    f"WHERE w.value IN (\n  {in_list}\n) AND w.lang = 'est'",
)
(BASE / "AMT-YS-semcat-query_20260706.sql").write_text(sql + ";\n", encoding="utf-8")

print(f"{len(forms)} distinct ÜS forms -> AMT-YS-semcat-query_20260706.sql")
if blank:
    print(f"\n-- {len(blank)} rows with no candidate (excluded) --")
    for l, t in blank:
        print(f"  {l} [{t}]")
if not_in_ys:
    print(f"\n-- candidates NOT in ÜS list (excluded from query, please check) --")
    for l, c in not_in_ys:
        print(f"  {l}: {c}")

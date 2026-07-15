#!/usr/bin/env python3
# Created: 2026-07-06 19-37-27
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
"""Copy ÜS semantic types onto AMT-Master entries.

Pipeline (rerunnable; see AMT-YS-match-report / ys-semcat workflow):
  1. mapping CSV : amt_lemma -> ÜS word(s) ("; "-separated), user-reviewed
  2. ÜS export   : DBeaver JSON {sql: [rows]}, one row per word/lexeme/meaning,
                   with `semantic_types` already ", "-aggregated per meaning
  3. target JSON : {"AMT-Master": [entries]} — two fields are inserted after
                   "Sem-Cat" into every entry:
                     YS-lemma   : the ÜS word(s) the semcats came from
                     YS-Sem-Cat : all distinct ÜS semantic types, ", "-joined,
                                  in first-seen order across candidates/meanings
                   "NULL" when the mapping has no candidate or ÜS has no
                   semantic types for any candidate (AMT placeholder convention).

Usage: uv run python join_ys_semcat.py [mapping.csv] [ys-export.json] [target.json]
(defaults below; the target file is read as base and overwritten in place)
"""
import csv
import json
import sys
from pathlib import Path

BASE = Path("/Users/q/dev/gen/Katus-DEV/Katus-ALUSANDMED/YS-Master-semcat-diff")
MAPPING_CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "Lemmas/AMT-YS-match-report_20260706_read.csv"
YS_JSON = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE / "JSON-YS/_SELECT_w_id_AS_word_id_w_value_AS_word_w_value_prese_AS_word_pr_202607061931.json"
TARGET_JSON = Path(sys.argv[3]) if len(sys.argv) > 3 else BASE / "Sem-Cat-YS-to-AMT/AMT-Master_annotated_with_YS-semcat.json"

# 1. mapping: amt_lemma -> [ÜS words]
mapping = {}
with MAPPING_CSV.open(encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        cands = [c.strip() for c in (row["ys_candidates"] or "").split(";") if c.strip()]
        mapping[row["amt_lemma"].strip()] = cands

# 2. ÜS export: word -> ordered distinct semantic types over all its meanings
ys_raw = json.load(YS_JSON.open(encoding="utf-8"))
ys_rows = next(iter(ys_raw.values()))
semcats = {}
for r in ys_rows:
    types = [t.strip() for t in (r.get("semantic_types") or "").split(",") if t.strip()]
    bucket = semcats.setdefault(r["word"], [])
    for t in types:
        if t not in bucket:
            bucket.append(t)

# 3. annotate target entries
data = json.load(TARGET_JSON.open(encoding="utf-8"))
entries = data["AMT-Master"]

n_filled, no_mapping, no_semcat, missing_words = 0, [], [], set()
for e in entries:
    # some Amt-Master-IDs carry stray whitespace ("maavalitseja ") — strip for lookup
    lemma = e["Amt-Master-ID"].strip()
    cands = mapping.get(lemma, [])
    cats = []
    for c in cands:
        if c not in semcats:
            missing_words.add(c)
        for t in semcats.get(c, []):
            if t not in cats:
                cats.append(t)
    ys_lemma = "; ".join(cands) if cands else "NULL"
    ys_semcat = ", ".join(cats) if cats else "NULL"
    if not cands:
        no_mapping.append(lemma)
    elif not cats:
        no_semcat.append(lemma)
    else:
        n_filled += 1
    # rebuild the entry with the two YS fields right after "Sem-Cat"
    new = {}
    for k, v in e.items():
        if k in ("YS-lemma", "YS-Sem-Cat"):
            continue
        new[k] = v
        if k == "Sem-Cat":
            new["YS-lemma"] = ys_lemma
            new["YS-Sem-Cat"] = ys_semcat
    if "Sem-Cat" not in new:  # entry without Sem-Cat field: append at end
        new["YS-lemma"] = ys_lemma
        new["YS-Sem-Cat"] = ys_semcat
    e.clear()
    e.update(new)

TARGET_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(f"{len(entries)} entries -> {TARGET_JSON.name}")
print(f"  YS-Sem-Cat filled : {n_filled}")
print(f"  no mapping row/candidate ({len(no_mapping)}): {', '.join(no_mapping) or '-'}")
print(f"  candidate(s) had no semantic_types in ÜS ({len(no_semcat)}):")
for l in no_semcat:
    print(f"    {l} <- {'; '.join(mapping.get(l, []))}")
if missing_words:
    print(f"  ÜS words in mapping but absent from export ({len(missing_words)}): {', '.join(sorted(missing_words))}")

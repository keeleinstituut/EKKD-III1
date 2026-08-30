# Created: 2026-07-12 13-26-44
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
"""Fold ÜS (EKI ühendsõnastik / Ekilex) semantic types into AMT-Master.

Replaces the value of the "Sem-Cat" field of every entry in
AMT-Master_annotated_with_YS-semcat.json with person-denoting ÜS semantic
types. Nothing else in the JSON is touched: no fields added or removed, no
entry lost, all other values byte-identical.

Sources, in priority order per entry:
  1. decisions TSV (semcat-72-review.tsv) — manually verified decisions
     (Madis Jürviste, 2026-07-12) for the 72 entries whose mapped ÜS word
     has no person-type semantic types (45 with no ÜS types at all, 27 with
     only non-person types) plus "jooks" (user decision: kuller's category).
  2. ÜS export via the reviewed mapping CSV — the entry's ÜS candidate
     word(s) contribute their semantic types, filtered to the person
     whitelist, ", "-joined in first-seen order across candidates/meanings.

Person whitelist (agreed 2026-07-12): in_elukutse, in_roll, in_omadus,
in_tegija, inimene, in_sugulane, in_rahvas, in_rahvas_keel, in_müt,
esitus_tiitel.

Special cases:
  - "alv rahvas" was renamed from "alw rahvas" after the mapping CSV was
    produced; the lookup falls back to the old spelling.

The script is fail-fast: any entry that would end up with an empty Sem-Cat,
or any decision tag outside the whitelist, aborts before writing. A
timestamped backup of the target is written next to it before overwrite.
Built-in validation re-reads the written file and compares it field by
field against the pre-run snapshot.

Usage: uv run python 15-fold_ys_semcat_into_master.py
"""
import copy
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Run from the original working-repo root (see README: these paths refer to
# the private data layout and are not present in the public repository).
BASE = Path("Katus-ALUSANDMED/YS-Master-semcat-diff")
MAPPING_CSV = BASE / "Lemmas/AMT-YS-match-report_20260706_read.csv"
YS_JSON = BASE / "JSON-YS/_SELECT_w_id_AS_word_id_w_value_AS_word_w_value_prese_AS_word_pr_202607061931.json"
DECISIONS_TSV = BASE / "Sem-Cat-YS-to-AMT/semcat-72-review.tsv"
TARGET_JSON = BASE / "Sem-Cat-YS-to-AMT/AMT-Master_annotated_with_YS-semcat.json"

PERSON_WHITELIST = {
    "in_elukutse", "in_roll", "in_omadus", "in_tegija", "inimene",
    "in_sugulane", "in_rahvas", "in_rahvas_keel", "in_müt", "esitus_tiitel",
}
RENAMED = {"alv rahvas": "alw rahvas"}  # current lemma -> lemma in mapping CSV

# 1. mapping: amt_lemma -> [ÜS candidate words]
mapping = {}
with MAPPING_CSV.open(encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        cands = [c.strip() for c in (row["ys_candidates"] or "").split(";") if c.strip()]
        mapping[row["amt_lemma"].strip()] = cands

# 2. ÜS export: word -> ordered distinct semantic types over all its meanings
ys_rows = next(iter(json.load(YS_JSON.open(encoding="utf-8")).values()))
semcats = {}
for r in ys_rows:
    types = [t.strip() for t in (r.get("semantic_types") or "").split(",") if t.strip()]
    bucket = semcats.setdefault(r["word"], [])
    for t in types:
        if t not in bucket:
            bucket.append(t)

# 3. manual decisions: lemma -> ", "-joined person tags
decisions = {}
with DECISIONS_TSV.open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        decisions[row["lemma"].strip()] = row["proposal"].strip()

# 4. annotate
data = json.load(TARGET_JSON.open(encoding="utf-8"))
entries = data["AMT-Master"]
snapshot = copy.deepcopy(entries)

n_decision, n_ys, errors = 0, 0, []
decisions_used = set()
for e in entries:
    lemma = e["Amt-Master-ID"].strip()
    if lemma in decisions:
        new_val = decisions[lemma]
        decisions_used.add(lemma)
        n_decision += 1
    else:
        cands = mapping.get(lemma) or mapping.get(RENAMED.get(lemma, ""), [])
        cats = []
        for c in cands:
            for t in semcats.get(c, []):
                if t in PERSON_WHITELIST and t not in cats:
                    cats.append(t)
        new_val = ", ".join(cats)
        n_ys += 1
    toks = {t.strip() for t in new_val.split(",") if t.strip()}
    if not toks:
        errors.append(f"EMPTY Sem-Cat would result for: {lemma}")
    elif not toks <= PERSON_WHITELIST:
        errors.append(f"non-whitelist tag(s) {toks - PERSON_WHITELIST} for: {lemma}")
    e["Sem-Cat"] = new_val

unused = set(decisions) - decisions_used
if unused:
    errors.append(f"decisions TSV rows not matched to any entry: {sorted(unused)}")
if errors:
    sys.exit("ABORT, nothing written:\n  " + "\n  ".join(errors))

# 5. backup, write
backup = TARGET_JSON.with_name(
    TARGET_JSON.stem + "_BCKP-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json"
)
shutil.copy2(TARGET_JSON, backup)
TARGET_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# 6. validate the written file against the pre-run snapshot
check = json.load(TARGET_JSON.open(encoding="utf-8"))["AMT-Master"]
assert len(check) == len(snapshot) == 882, f"entry count changed: {len(check)}"
touched = set()
for old, new in zip(snapshot, check):
    assert list(old.keys()) == list(new.keys()), f"field set/order changed in {old['Amt-Master-ID']}"
    for k in old:
        if old[k] != new[k]:
            assert k == "Sem-Cat", f"field {k!r} changed in {old['Amt-Master-ID']}"
            touched.add(new["Amt-Master-ID"].strip())
    toks = {t.strip() for t in new["Sem-Cat"].split(",")}
    assert toks and toks <= PERSON_WHITELIST, f"bad Sem-Cat in {new['Amt-Master-ID']}: {new['Sem-Cat']!r}"

print(f"{len(check)} entries -> {TARGET_JSON.name}   (backup: {backup.name})")
print(f"  Sem-Cat from ÜS via mapping : {n_ys}")
print(f"  Sem-Cat from decisions TSV  : {n_decision}")
print(f"  Sem-Cat values changed      : {len(touched)}  (unchanged: {len(check) - len(touched)})")
print("  validation: entry count, field sets/order, non-Sem-Cat values, whitelist — all OK")

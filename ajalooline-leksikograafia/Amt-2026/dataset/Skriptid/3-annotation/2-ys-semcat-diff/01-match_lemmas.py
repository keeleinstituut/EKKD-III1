#!/usr/bin/env python3
# Created: 2026-07-06 18-24-53
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
"""Match AMT-Master lemmas against the ÜS (Sõnaveeb/eki) lemma list.

Tiers (precision-first):
  1. exact        — identical string (after whitespace strip)
  2. casefold     — identical after lowercasing
  3. orthographic — identical after archaic-orthography normalization (w→v, dd→d, ...)
  4. compound     — MWU joined without spaces exists in ÜS ("lahti mees" → "lahtimees")
  5. comphead     — compound whose head (longest ÜS suffix, ≥5 chars, prefix ≥3) exists
                    in ÜS ("alevisulane" → "sulane"); semantic type comes from the head
  6. fuzzy        — high-similarity candidates (difflib), for manual review
  7. verb         — -ja agent noun whose base verb (-ma) exists in ÜS ("aitaja" → "aitama");
                    NB: yields verb semantic types, not person categories
  8. head         — MWU head word (last token) exists in ÜS, as a fallback suggestion
  unmatched       — nothing plausible found

Outputs:
  AMT-YS-match-report_20260706.tsv  — one row per AMT lemma: tier, ÜS form(s)
  AMT-YS-semcat-query_20260706.sql  — the example export query with WHERE w.value IN
                                      (every distinct ÜS candidate from all tiers);
                                      extra candidates are harmless — the join back to
                                      AMT-Master uses only the reviewed mapping TSV
"""
import difflib
import re
import unicodedata
from pathlib import Path

# Original working-repo data layout (not included in the public repository).
BASE = Path("Katus-ALUSANDMED/YS-Master-semcat-diff/Lemmas")
YS_FILE = BASE / "YS-lemmad_20260506.txt"
AMT_FILE = BASE / "AMT-Master-lemmas_20260706.txt"
OUT = BASE / "AMT-YS-match-report_20260706.tsv"

ys = [ln.rstrip("\n") for ln in YS_FILE.open(encoding="utf-8")]
# The ÜS export appends homonym numbers ("kangur 2"); the DB value is the bare lemma.
ys = [re.sub(r" \d+$", "", w) for w in (w.strip() for w in ys) if w]
ys = sorted(set(ys))
ys_set = set(ys)
ys_lower = {}
for w in ys:
    ys_lower.setdefault(w.lower(), []).append(w)

amt_raw = [ln.rstrip("\n") for ln in AMT_FILE.open(encoding="utf-8")]
amt = [w for w in (w.strip() for w in amt_raw) if w]


def norm_orth(s: str) -> str:
    """Normalize archaic Estonian orthography toward modern forms."""
    s = unicodedata.normalize("NFC", s.lower())
    s = s.replace("w", "v")
    # collapse double consonants that old orthography wrote long: dd->d, bb->b, gg->g
    s = re.sub(r"dd", "d", s)
    s = re.sub(r"bb", "b", s)
    s = re.sub(r"gg", "g", s)
    # old 'ä/õ' variance is too risky to normalize; leave as-is
    return s


ys_orth = {}
for w in ys:
    ys_orth.setdefault(norm_orth(w), []).append(w)

ys_lower_list = list(ys_lower.keys())

def compound_head(lemma: str):
    """Longest ÜS lemma that is a proper suffix of `lemma` (head of a compound)."""
    s = norm_orth(lemma)
    for i in range(3, len(s) - 4):  # prefix >=3 chars, suffix >=5 chars
        suf = s[i:]
        if suf in ys_lower:
            return ys_lower[suf]
    return None


def agent_verb(lemma: str):
    """-ja agent noun -> -ma verb ("aitaja" -> "aitama")."""
    s = norm_orth(lemma)
    if s.endswith("ja") and len(s) > 4:
        verb = s[:-2] + "ma"
        if verb in ys_lower:
            return ys_lower[verb]
    return None


rows = []
counts = {}
for lemma in amt:
    tier, matches = None, []
    if lemma in ys_set:
        tier, matches = "exact", [lemma]
    elif lemma.lower() in ys_lower:
        tier, matches = "casefold", ys_lower[lemma.lower()]
    elif norm_orth(lemma) in ys_orth:
        tier, matches = "orthographic", ys_orth[norm_orth(lemma)]
    elif " " in lemma:
        joined = lemma.replace(" ", "").lower()
        if joined in ys_lower:
            tier, matches = "compound", ys_lower[joined]
    if tier is None and " " not in lemma:
        ch = compound_head(lemma)
        if ch:
            tier, matches = "comphead", ch
    if tier is None:
        cand = difflib.get_close_matches(norm_orth(lemma), ys_lower_list, n=3, cutoff=0.87)
        if cand:
            tier = "fuzzy"
            matches = sorted({m for c in cand for m in ys_lower[c]})
    if tier is None:
        av = agent_verb(lemma)
        if av:
            tier, matches = "verb", av
        elif " " in lemma:
            head = lemma.split()[-1].lower()
            if head in ys_lower:
                tier, matches = "head", ys_lower[head]
    if tier is None:
        tier = "unmatched"
    counts[tier] = counts.get(tier, 0) + 1
    rows.append((lemma, tier, "; ".join(matches)))

with OUT.open("w", encoding="utf-8") as f:
    f.write("amt_lemma\ttier\tys_candidates\n")
    for r in rows:
        f.write("\t".join(r) + "\n")

total = len(amt)
print(f"{total} AMT lemmas")
for t in ["exact", "casefold", "orthographic", "compound", "comphead", "fuzzy", "verb", "head", "unmatched"]:
    if t in counts:
        print(f"  {t:13s} {counts[t]:4d}")
print(f"\nreport: {OUT}")

# ---- SQL generation -------------------------------------------------------
SQL_OUT = BASE / "AMT-YS-semcat-query_20260706.sql"
example = BASE / "_SELECT_w_id_AS_word_id_w_value_AS_word_w_value_prese_AS_word_pr_202607061807.json"
import json

sql_template = next(iter(json.load(example.open(encoding="utf-8"))))

all_forms = sorted({m for _, _, ms in rows if ms for m in ms.split("; ")})
in_list = ",\n  ".join("'" + f.replace("'", "''") + "'" for f in all_forms)
sql = sql_template.replace(
    "WHERE w.value IN ('abikaasa')",
    f"WHERE w.value IN (\n  {in_list}\n) AND w.lang = 'est'",
)
SQL_OUT.write_text(sql + ";\n", encoding="utf-8")
print(f"SQL ({len(all_forms)} distinct ÜS forms): {SQL_OUT}")

#!/usr/bin/env python3
# Created: 2026-07-13 14-44-29
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
"""Diff AMT-Master (json-all) against the APTSK dictionary (Eesti vanema
piiblitõlke sõnastik 1600–1739) to measure how many Master lemmas are also
attested in APTSK, maximising recall with tiered matching.

Match tiers (best first, one tier per Master lemma):
  T1 EXACT     – Master lemma == an APTSK headword form (NFC, lowercased).
  T2 SPACING   – equal after removing spaces & hyphens on both sides
                 (catches "alv rahvas" ↔ "alvrahvas", "avanisa-amm" ↔ ...).
  T3 COMPOUND  – Master is a compound whose parts are BOTH APTSK headwords
                 (multi-token split on space/hyphen, or a solid compound
                 decomposed greedily against APTSK's own vocabulary).
  T4 HEAD      – only the head element (last token / a ≥4-char suffix) is an
                 APTSK headword; the modifier is not attested.
  T0 NONE      – no correspondence found.

"Same word" recall  = T1 + T2.
"Broad" recall      = T1 + T2 + T3 (+T4 as weak).

Outputs (into Katus-ALUSANDMED/):
  APTSK-Master_diff_report.md   – human-readable report
  APTSK-Master_diff_detail.csv  – one row per Master lemma with evidence
"""
import json, glob, csv, unicodedata, difflib
from pathlib import Path

MASTER = Path("/Users/q/dev/gen/Katus-DEV/Katus-ALUSANDMED/json-all/AMT-Master_annotated.json")
APT_DIR = Path("/Users/q/dev/gen/APTSK-analysis/json")
OUT_DIR = Path("/Users/q/dev/gen/Katus-DEV/Katus-ALUSANDMED")

def norm(s):
    return unicodedata.normalize("NFC", s).strip().lower() if s else s

def strip_sep(s):
    return s.replace(" ", "").replace("-", "") if s else s

# ---- load APTSK -----------------------------------------------------------
apt = []
for f in sorted(glob.glob(str(APT_DIR / "entries_*.json"))):
    apt.extend(json.load(open(f))["entries"])

# form -> list of entries; and a stripped-form index for T2
forms = {}          # norm(form) -> [entries]
stripped = {}       # strip_sep(norm(form)) -> set(norm(form))
for e in apt:
    hs = set()
    if e.get("headword"): hs.add(e["headword"])
    for h in e.get("headwords") or []: hs.add(h)
    for hd in e.get("headword_details") or []:
        if hd.get("form"): hs.add(hd["form"])
    for h in hs:
        n = norm(h)
        forms.setdefault(n, []).append(e)
        stripped.setdefault(strip_sep(n), set()).add(n)

formset = set(forms)

def entry_brief(n):
    """Compact evidence for the first non-reference entry under form n (else first)."""
    es = forms.get(n, [])
    chosen = next((e for e in es if not e.get("reference_only")), es[0] if es else None)
    if not chosen:
        return {}
    return {
        "apt_id": chosen.get("id"),
        "apt_headword": chosen.get("headword"),
        "homonym": chosen.get("homonym"),
        "reference_only": bool(chosen.get("reference_only")),
        "gloss": (chosen.get("gloss") or "")[:120],
        "pages": ",".join(str(p) for p in (chosen.get("pages") or [])),
        "n_entries": len(es),
    }

def decompose(tok):
    """Return (modifier, head) if tok is a solid compound whose parts are both
    APTSK headwords (len>=3 each). Prefer the split with the longest head."""
    best = None
    for i in range(3, len(tok) - 2):
        mod, head = tok[:i], tok[i:]
        if mod in formset and head in formset:
            if best is None or len(head) > len(best[1]):
                best = (mod, head)
    return best

def longest_suffix_head(tok):
    """Longest APTSK headword (>=4 chars) that is a suffix of tok, else None."""
    best = None
    for i in range(len(tok) - 4, -1, -1):
        cand = tok[i:]
        if len(cand) >= 4 and cand in formset:
            if best is None or len(cand) > len(best):
                best = cand
    return best

# ---- match Master ---------------------------------------------------------
master = json.load(open(MASTER))["AMT-Master"]
records = []
for r in master:
    raw = r["Amt-Master-ID"]
    n = norm(raw)
    rec = {"master_id": raw, "amt_cat": r.get("Amt-Cat"), "sem_cat": r.get("Sem-Cat"),
           "tier": "T0", "match_form": "", "evidence": {}, "note": ""}

    if n in formset:                                    # T1
        rec["tier"] = "T1"; rec["match_form"] = n
        rec["evidence"] = entry_brief(n)
    else:
        ns = strip_sep(n)
        cand = stripped.get(ns, set()) - {n}
        if cand:                                        # T2
            m = sorted(cand)[0]
            rec["tier"] = "T2"; rec["match_form"] = m
            rec["evidence"] = entry_brief(m)
            rec["note"] = "spacing/hyphen variant"
        else:
            # split into elements on space/hyphen
            toks = [t for t in n.replace("-", " ").split() if t]
            parts_attested = [t for t in toks if t in formset]
            head_tok = toks[-1] if toks else n
            if len(toks) > 1:
                # multi-token compound
                if head_tok in formset and all(t in formset for t in toks):
                    rec["tier"] = "T3"; rec["match_form"] = head_tok
                    rec["evidence"] = entry_brief(head_tok)
                    rec["note"] = "all parts attested: " + "+".join(toks)
                elif head_tok in formset:
                    rec["tier"] = "T4"; rec["match_form"] = head_tok
                    rec["evidence"] = entry_brief(head_tok)
                    rec["note"] = "head attested; modifier(s) not: " + "+".join(
                        t for t in toks if t not in formset)
                elif parts_attested:
                    rec["tier"] = "T4"; rec["match_form"] = parts_attested[0]
                    rec["evidence"] = entry_brief(parts_attested[0])
                    rec["note"] = "only modifier attested: " + "+".join(parts_attested)
            else:
                # solid single token: try decomposition
                dec = decompose(n)
                if dec:
                    rec["tier"] = "T3"; rec["match_form"] = dec[1]
                    rec["evidence"] = entry_brief(dec[1])
                    rec["note"] = f"compound both parts attested: {dec[0]}+{dec[1]}"
                else:
                    suf = longest_suffix_head(n)
                    if suf:
                        rec["tier"] = "T4"; rec["match_form"] = suf
                        rec["evidence"] = entry_brief(suf)
                        rec["note"] = f"head suffix attested: -{suf}"
    records.append(rec)

# ---- tallies --------------------------------------------------------------
from collections import Counter
tc = Counter(r["tier"] for r in records)
tot = len(records)
same = tc["T1"] + tc["T2"]
broad = same + tc["T3"]
broad_w = broad + tc["T4"]

# ---- CSV ------------------------------------------------------------------
OUT_DIR.mkdir(exist_ok=True)
with open(OUT_DIR / "APTSK-Master_diff_detail.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["master_id","tier","match_form","amt_cat","sem_cat","note",
                "apt_id","apt_headword","homonym","reference_only","gloss","pages","apt_n_entries"])
    for r in records:
        ev = r["evidence"]
        w.writerow([r["master_id"], r["tier"], r["match_form"], r["amt_cat"], r["sem_cat"],
                    r["note"], ev.get("apt_id",""), ev.get("apt_headword",""),
                    ev.get("homonym",""), ev.get("reference_only",""), ev.get("gloss",""),
                    ev.get("pages",""), ev.get("n_entries","")])

# ---- Markdown report ------------------------------------------------------
def pct(x): return f"{100*x/tot:.1f}%"
def sample(tier, k=25):
    return [r["master_id"] for r in records if r["tier"] == tier][:k]

lines = []
lines.append("# AMT-Master ↔ APTSK correspondence report\n")
lines.append(f"Generated by `scripts/aptsk_master_diff.py`.\n")
lines.append("**Master:** `Katus-ALUSANDMED/json-all/AMT-Master_annotated.json` "
             f"— {tot} lemmas (`Amt-Master-ID`).  ")
lines.append("**APTSK:** *Eesti vanema piiblitõlke sõnastik 1600–1739* (EKI/EKSA 2025) "
             f"— {len(apt)} entries, {len(formset)} distinct headword forms "
             f"({sum(1 for e in apt if e.get('reference_only'))} reference-only redirects).\n")
lines.append("## How matching works\n")
lines.append("Each Master lemma is assigned to the single best tier below "
             "(APTSK headwords compared NFC-normalised & lowercased):\n")
lines.append("| Tier | Meaning |")
lines.append("|------|---------|")
lines.append("| **T1 EXACT** | Master lemma *is* an APTSK headword form |")
lines.append("| **T2 SPACING** | equal after removing spaces/hyphens (spelling-of-compound variant) |")
lines.append("| **T3 COMPOUND** | compound whose parts are **all** APTSK headwords (multi-word, or solid compound split against APTSK vocabulary) |")
lines.append("| **T4 HEAD** | only the head element / a ≥4-char suffix is attested; modifier is not |")
lines.append("| **T0 NONE** | no correspondence found |\n")
lines.append("## Headline numbers\n")
lines.append(f"- **Same word present in APTSK (T1+T2): {same} / {tot} = {pct(same)}**")
lines.append(f"  - T1 exact: {tc['T1']} ({pct(tc['T1'])})")
lines.append(f"  - T2 spacing/hyphen variant: {tc['T2']} ({pct(tc['T2'])})")
lines.append(f"- Broad correspondence incl. attested compounds (T1+T2+T3): **{broad} / {tot} = {pct(broad)}**")
lines.append(f"  - T3 compound, all parts attested: {tc['T3']} ({pct(tc['T3'])})")
lines.append(f"- Widest incl. head-only (T1–T4): {broad_w} / {tot} = {pct(broad_w)}")
lines.append(f"  - T4 head/suffix only: {tc['T4']} ({pct(tc['T4'])})")
lines.append(f"- **No correspondence (T0): {tc['T0']} ({pct(tc['T0'])})**\n")

lines.append(f"## T1 EXACT — full matches ({tc['T1']})\n")
lines.append("Master lemmas present verbatim as an APTSK headword:\n")
t1 = [r["master_id"] for r in records if r["tier"]=="T1"]
lines.append(", ".join(f"`{w}`" for w in t1))
lines.append("")

lines.append("## Examples per tier\n")
for tier, label in [("T2","T2 SPACING"),("T3","T3 COMPOUND"),("T4","T4 HEAD")]:
    ex = [f"`{r['master_id']}` → {r['note']}" for r in records if r["tier"]==tier][:20]
    lines.append(f"### {label} ({tc[tier]})\n")
    lines.extend(f"- {e}" for e in ex)
    lines.append("")

lines.append(f"### T0 NONE ({tc['T0']}) — Master lemmas absent from APTSK\n")
t0 = [r["master_id"] for r in records if r["tier"]=="T0"]
lines.append(", ".join(f"`{w}`" for w in t0))
lines.append("")

lines.append("## Near-miss candidates for manual review\n")
lines.append("T0 lemmas that have a *close* (≥0.86 ratio) APTSK headword form — likely "
             "spelling/inflection variants worth a human check. **Not counted above.**\n")
nm = []
for r in records:
    if r["tier"] != "T0":
        continue
    c = difflib.get_close_matches(norm(r["master_id"]), formset, n=2, cutoff=0.86)
    if c:
        nm.append((r["master_id"], c))
for mid, cands in nm:
    lines.append(f"- `{mid}` ~ " + ", ".join(f"`{c}`" for c in cands))
lines.append(f"\n({len(nm)} near-miss candidates.)\n")

lines.append("## Full detail\n")
lines.append("Per-lemma evidence (tier, matched APTSK headword, id, page, gloss) is in "
             "`APTSK-Master_diff_detail.csv`.")

(OUT_DIR / "APTSK-Master_diff_report.md").write_text("\n".join(lines) + "\n")

print(f"Master lemmas: {tot}")
print(f"T1 exact   : {tc['T1']}")
print(f"T2 spacing : {tc['T2']}")
print(f"T3 compound: {tc['T3']}")
print(f"T4 head    : {tc['T4']}")
print(f"T0 none    : {tc['T0']}")
print(f"Same word (T1+T2): {same} ({pct(same)})")
print(f"Broad (T1-T3)    : {broad} ({pct(broad)})")
print("Wrote report + CSV to", OUT_DIR)

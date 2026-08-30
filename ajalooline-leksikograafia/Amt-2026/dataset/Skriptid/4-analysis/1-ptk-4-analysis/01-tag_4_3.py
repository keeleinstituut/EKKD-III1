#!/usr/bin/env python3
# Created: 2026-07-12 20-34-20
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
# Co-Authored-By: Claude Fable 5
"""Auto first-pass tagger for sub-chapter 4.3 (societal themes).

Adds four analytical axes on top of the existing Amt-Cat / Sem-Cat fields and
writes an intermediary tagged copy plus a review CSV. This is a FIRST PASS only:
heuristics from the German (`*-de`) glosses, the Estonian (`*-et`) forms and the
DEF fields. Every label carries a confidence; manual review is expected
(mandatory for `Sugu` and every `NÕID` row — see 4-3_evidence-plan.md).

Fields added (after Sem-Cat):
    Teema           primary theme (highest-priority match)
    Teema-all       all matched themes, comma-separated
    Teema-conf      high / med / low
    Sugu            M / N / Ü   (gender)
    Sugu-conf       high / med / low
    Hierarhia       (left blank — manual)
    Hääbunud        (left blank — filled from 04-ghost_screen.py + Sõnaveeb)
    DE_hüperonüüm   German hypernym(s) for NÕID rows (else blank)

Usage:
    uv run python 01-tag_4_3.py

NB (retrospective note): this script ran against the pre-fold-in master
(2026-07-12), whose Sem-Cat values were uppercase (IN_ELUKUTSE, IN_MÜT) and
whose catch-all theme was still named MUU. The published dataset carries
lowercase ÜS tags (in_elukutse, in_müt) and the theme renamed to
OMADUS_SEOS_KUULUVUS, so the Sem-Cat fallback branches below would not fire
against the released file. Kept as run, for provenance.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "Ptk-3" / "AMT-Master_annotated.json"
OUT_JSON = HERE / "AMT-Master_4-3-tagged.json"
OUT_CSV = HERE / "4-3_tagging_review.csv"

DE_KEYS = ["Stahl-1637-de", "Gutslaff-1648-de", "Göseken-1660-de", "Vestring-17XX-de",
           "Helle-1732-de", "Hupel-1780-est-ger-de"]
ET_KEYS = ["Stahl-1637-et", "Gutslaff-1648-et", "Göseken-1660-et", "Vestring-17XX-et",
           "Helle-1732-et", "Hupel-1780-est-ger-et"]
PLACEHOLDERS = {"", "---", "NULL", "???"}


def blob(entry: dict, keys: list[str]) -> str:
    return " ".join((entry.get(k) or "") for k in keys
                    if (entry.get(k) or "").strip() not in PLACEHOLDERS)


# ---------------------------------------------------------------------------
# THEME (Teema) — ordered by specificity; primary = first match in this order.
# Each theme: (German-gloss substrings, Estonian-id/form substrings)
# ---------------------------------------------------------------------------
THEME_RULES = [
    ("NÕID",            (r"zauber|hexe|wahrsag|warsag|beschwör|beschwer der geist|"
                         r"teuffel|teufel|segen.?sprech|zeichendeut|zeichen.?deut",
                         r"nõid|noid|\btark\b|lausuja|arbuja|soolapuhuja|kuradikunst|"
                         r"\bvõlu|lummaja|ninatark")),
    ("SUGU_REPRO",      (r"hebamme|wehmutter|weh.?mutter|\bamme\b|säug|kindermutter|nabel",
                         r"\bamm\b|ämmaemand|nurganaine|imetaja|nabaema")),
    ("MORAAL_HÄLVE",    (r"\bhure\b|huren|\bdieb\b|betrieg|betrüg|mörder|räuber|"
                         r"schelm|ehebrecher|lügner|verräter|verrähter",
                         r"\bhoor|\bport\b|\bvaras\b|röövel|petis|valelik|abielurikkuja|"
                         r"kelm|mõrtsukas|tapja")),
    ("KIRIK_VAIMULIK",  (r"pfaff|pfarr|priester|mönch|münch|bischof|küster|probst|"
                         r"capellan|caplan|kaplan|diacon|abt\b|äbtissin|nonne",
                         r"\bpapp|piiskop|\bmunk\b|köster|preester|nunn|abt")),
    ("KIRIK_FUNKTSIOON", (r"prediger|seelsorger|catechet|glaubig|gläubig|jünger",
                         r"jutlustaja|kirikuõpetaja|\bõpetaja|\bõppija|kogudus|usklik|jünger")),
    ("MÜÜT",            (r"\briese\b|\bgeist\b|gespenst|kobold|drache",
                         r"\btont\b|vaim|hiid|kratt|luupainaja")),
    ("TEENISTUS",       (r"knecht|\bmagd\b|diener|bediente|sklave|sclave|gesinde|dienstbote|"
                         r"\bdienst\b|leibeigen",
                         r"sulane|teenija|\bori\b|ümmardaja|teener|tüdruk|orja")),
    ("HALDUS_VÕIM",     (r"amtmann|\bvogt\b|aufseher|ältest|verwalter|schultze|frohnvogt|"
                         r"statthalter|richter|edelmann|adelsmann|gutsherr|junker|graf|baron|"
                         r"könig|kaiser|fürst|herzog",
                         r"kubjas|kupjas|vöörmünder|\bvanem\b|\bülem\b|mõisnik|asehaldur|"
                         r"kohtu|raad|raeisand|pealik")),
    ("FEOD_MAA",      (r"\bbauer\b|\bbaur\b|erbbauer|erbkerl|häker|häkler|hakenbauer|"
                         r"bauerwirth|gesinde.?wirth|\bwirth\b|hausvater|hausmutter",
                         r"talupoeg|talumees|adramees|vabadik|pärismees|peremees|perenaine|"
                         r"\bsaks\b|kadakasaks")),
    ("KÄSITÖÖ",         (r"schmid|schuster|töpfer|töpffer|weber|gläser|glaser|maurer|mäurer|"
                         r"drechsler|dreschler|bäcker|becker|schneider|fleischer|gärtner|"
                         r"böttcher|sattler|gerber|macher",
                         r"\bsepp\b|kingsepp|\bnik\b|kangur|aednik|pottsepp|müürsepp|treial|"
                         r"rätsep|köösner|parkal|tisler")),
]
THEME_COMPILED = [(name, re.compile(de, re.I), re.compile(et, re.I))
                  for name, (de, et) in THEME_RULES]
# distinctive (high-confidence) themes when matched at all
THEME_DISTINCTIVE = {"NÕID", "SUGU_REPRO", "MORAAL_HÄLVE", "KIRIK_VAIMULIK", "MÜÜT"}


def tag_theme(de: str, et: str, sem: str) -> tuple[str, list[str], str]:
    matched = []
    for name, de_re, et_re in THEME_COMPILED:
        if de_re.search(de) or et_re.search(et):
            matched.append(name)
    if "IN_MÜT" in (sem or "") and "MÜÜT" not in matched:
        matched.append("MÜÜT")
    if not matched:
        # Sem-Cat fallback: a lexicalised profession with no marked social theme
        # is an ordinary trade/craft (Langemets IN_ELUKUTSE). Reserve MUU for the
        # genuinely non-occupational / non-thematic remainder.
        if "IN_ELUKUTSE" in (sem or ""):
            return "KÄSITÖÖ", ["KÄSITÖÖ"], "med"
        return "MUU", [], "low"
    primary = matched[0]
    conf = "high" if primary in THEME_DISTINCTIVE else ("med" if len(matched) == 1 else "low")
    return primary, matched, conf


# ---------------------------------------------------------------------------
# GENDER (Sugu)
# ---------------------------------------------------------------------------
FEM_DE = re.compile(r"\b(weib|frau|magd|mutter|dirne|hure|wittwe|witwe|gattin|tochter|"
                    r"jungfer|jungfrau|amme|hebamme|nonne|äbtissin|königin|kaiserin|"
                    r"\w*erin|\w*erinn)\b", re.I)
# capitalised -in/-inn agent nouns, excluding -ein/-lein and known non-female words
FEM_DE_SUFFIX = re.compile(r"\b[A-ZÄÖÜ][a-zäöü]{3,}inn?\b")
FEM_SUFFIX_STOP = {"capitain", "klein", "stein", "schwein", "herrlein", "mägdlein", "verein"}
FEM_ET = re.compile(r"(naine|naene|tüdruk|emand|\bämm\b|tar\b|tütar|\blesk|hoor|nurga|"
                    r"\bamm\b|imetaja|perenaine|kokanaine|kedranaine)", re.I)

MALE_DE = re.compile(r"\b(mann|knecht|vater|herr|sohn|bruder|ehemann|bursch|kerl|knabe|"
                     r"junge|vogt|graf|herzog|könig|kaiser|meister|gesell)\w*\b", re.I)
MALE_ET = re.compile(r"(mees\b|poiss|poeg|\bisa\b|vend|sulane|härra|isand|peremees)", re.I)


def _has_fem_suffix(de: str) -> bool:
    for w in FEM_DE_SUFFIX.findall(de):
        lw = w.lower()
        if lw.endswith("ein") or lw.endswith("lein") or lw in FEM_SUFFIX_STOP:
            continue
        return True
    return False


def tag_gender(de: str, et: str, defn: str) -> tuple[str, str]:
    fem = bool(FEM_DE.search(de) or _has_fem_suffix(de) or FEM_ET.search(et) or "naine" in defn.lower())
    male = bool(MALE_DE.search(de) or MALE_ET.search(et))
    if fem and not male:
        return "N", "med"
    if male and not fem:
        return "M", "med"
    if fem and male:
        return "Ü", "low"   # both genders attested (e.g. der X / die Y) — review
    return "Ü", "low"       # no marker — default neutral, review


# ---------------------------------------------------------------------------
# DE hypernym (witchcraft cluster only)
# ---------------------------------------------------------------------------
HYPERNYMS = ["Zauberer", "Zauberin", "Hexe", "Hexenmeister", "Wahrsager", "Beschwörer",
             "Zeichendeuter", "Teufelskünstler", "Segensprecher"]
HYP_RE = {h: re.compile(re.escape(h.replace("ö", "o")).replace("o", "[oö]"), re.I) for h in HYPERNYMS}


def tag_hypernym(de: str) -> str:
    de_norm = de
    found = [h for h in HYPERNYMS if re.search(re.escape(h[:6]), de_norm, re.I)
             or (h == "Wahrsager" and re.search(r"wa[hr]rsag", de_norm, re.I))]
    return ", ".join(dict.fromkeys(found))


def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    rows = data["AMT-Master"]

    review_rows = []
    theme_counter: dict[str, int] = {}
    gender_counter: dict[str, int] = {}

    for r in rows:
        de = blob(r, DE_KEYS)
        et = blob(r, ET_KEYS)
        defn = (r.get("DEF_en") or "") + " " + (r.get("DEF_et") or "")
        sem = r.get("Sem-Cat") or ""

        theme, theme_all, t_conf = tag_theme(de, et, sem)
        gender, g_conf = tag_gender(de, et, defn)
        hyper = tag_hypernym(de) if theme == "NÕID" else ""

        # insert new fields right after Sem-Cat, preserving original order
        new = {}
        for k, v in r.items():
            new[k] = v
            if k == "Sem-Cat":
                new["Teema"] = theme
                new["Teema-all"] = ", ".join(theme_all)
                new["Teema-conf"] = t_conf
                new["Sugu"] = gender
                new["Sugu-conf"] = g_conf
                new["Hierarhia"] = ""        # manual
                new["Hääbunud"] = ""         # from ghost_screen + Sõnaveeb
                new["DE_hüperonüüm"] = hyper
        r.clear()
        r.update(new)

        theme_counter[theme] = theme_counter.get(theme, 0) + 1
        gender_counter[gender] = gender_counter.get(gender, 0) + 1
        review_rows.append({
            "Amt-Master-ID": new["Amt-Master-ID"], "Amt-Cat": new.get("Amt-Cat"),
            "Sem-Cat": sem, "Teema": theme, "Teema-conf": t_conf, "Teema-all": ", ".join(theme_all),
            "Sugu": gender, "Sugu-conf": g_conf, "DE_hüperonüüm": hyper,
            "de_glosses": de, "et_forms": et,
        })

    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(review_rows[0].keys()))
        w.writeheader()
        w.writerows(review_rows)

    print(f"Tagged {len(rows)} entries.")
    print(f"  JSON -> {OUT_JSON.name}")
    print(f"  CSV  -> {OUT_CSV.name}")
    print("\nTeema distribution:")
    for k, v in sorted(theme_counter.items(), key=lambda x: -x[1]):
        print(f"  {k:18} {v}")
    print("\nSugu distribution:")
    for k, v in sorted(gender_counter.items(), key=lambda x: -x[1]):
        print(f"  {k:4} {v}")
    lowconf = sum(1 for x in review_rows if x["Teema-conf"] == "low")
    print(f"\nLow-confidence Teema rows (review priority): {lowconf}")
    print(f"NÕID rows (review all): {theme_counter.get('NÕID', 0)}")


if __name__ == "__main__":
    main()

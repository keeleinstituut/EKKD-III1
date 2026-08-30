# Created: 2026-07-08 16-18-21
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
"""
Step 3/3 — validate the parsed dictionary against the PDF extraction.

Checks:
 1. losslessness — the concatenation of all entries' raw_lines must equal,
    character for character, the concatenation of all body lines extracted
    from the PDF (build/spans.jsonl);
 2. frequency arithmetic — per entry, sum of source counts == printed total;
 3. vocabulary — source abbreviations and example labels are from the known
    set; verse references match the expected pattern;
 4. ordering — headwords are (loosely) alphabetical inside each letter;
 5. structure — reference entries have targets, normal entries have
    statistics; parser warnings are summarised.

Output: json/validation_report.json (+ console summary).

Run:  uv run python APTSK-scripts/aptsk_03_validate.py
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPANS = ROOT / "build" / "spans.jsonl"
JSON_DIR = ROOT / "json"

SOURCES = {"Ml", "Rs", "St", "GtVT", "GtUT", "Gt", "Bl", "WT", "Vr", "Mn", "PR"}
# verse refs: 'Ilm22:13', '1Ms2:7', 'Ilm8 algus', 'Mt23:34 pk algus',
# plus printed variants 'Ef lõpp', 'Lk 1:5', '1Ms8;5', 'Js 53.5'
REF_RE = re.compile(
    r"^[1-5]?[A-ZÕÄÖÜŠŽ][a-zõäöüšž]{0,3}\s?\d*(?:[:;.]\d+[ab]?)?"
    r"(?:\s(?:pk\s)?(?:algus|lõpp))?$")

EST_ORDER = "abcdefghijklmnopqrsšzžtuvwõäöüxy"
COLL = {c: i for i, c in enumerate(EST_ORDER)}


def collate(word):
    w = word.lower()
    return [COLL.get(c, 99) for c in w if c.isalpha()]


def main():
    entries = json.loads((JSON_DIR / "all_entries.json").read_text())["entries"]
    report = {"checks": {}, "issues": {}}

    # 1. losslessness --------------------------------------------------------
    body = []
    with SPANS.open() as f:
        for raw in f:
            rec = json.loads(raw)
            if "letter" in rec:
                continue
            body.append("".join(t for _, t in rec["s"]))
    extracted = "\n".join(body)
    reconstructed = "\n".join(l for e in entries for l in e["raw_lines"])
    if extracted == reconstructed:
        report["checks"]["lossless_coverage"] = "PASS — every extracted body "\
            "line is preserved verbatim in exactly one entry"
    else:
        # locate first divergence for debugging
        i = next((k for k, (a, b) in enumerate(zip(extracted, reconstructed))
                  if a != b), min(len(extracted), len(reconstructed)))
        report["checks"]["lossless_coverage"] = (
            f"FAIL at char {i}: extracted …{extracted[max(0,i-40):i+40]!r} "
            f"vs reconstructed …{reconstructed[max(0,i-40):i+40]!r}")

    # 2. frequency arithmetic ------------------------------------------------
    mismatches = []
    for e in entries:
        if e["reference_only"] or e["counts_omitted"] or e["total_count"] is None:
            continue
        s = sum(sc["count"] or 0 for sc in e["source_counts"])
        if s != e["total_count"]:
            mismatches.append({"id": e["id"], "total": e["total_count"],
                               "sum": s, "pages": e["pages"]})
    report["checks"]["frequency_sums"] = (
        f"{len(mismatches)} mismatches out of "
        f"{sum(1 for e in entries if not e['reference_only'] and not e['counts_omitted'])} counted entries")
    report["issues"]["frequency_mismatches"] = mismatches

    # 3. vocabulary ----------------------------------------------------------
    bad_src = [{"id": e["id"], "source": sc["source"]}
               for e in entries for sc in e["source_counts"]
               if sc["source"] not in SOURCES]
    bad_ex_src = [{"id": e["id"], "source": x["source"]}
                  for e in entries for x in e["examples"]
                  if x["source"] is not None and x["source"] not in SOURCES]
    bad_refs = [{"id": e["id"], "ref": r}
                for e in entries for x in e["examples"] for r in x["references"]
                if not REF_RE.match(r)]
    report["checks"]["source_abbreviations"] = \
        f"{len(bad_src)} bad in statistics, {len(bad_ex_src)} bad in examples"
    report["checks"]["reference_format"] = \
        f"{len(bad_refs)} references not matching the verse pattern"
    report["issues"]["bad_sources"] = bad_src + bad_ex_src
    report["issues"]["bad_references"] = bad_refs[:200]

    # 4. alphabetical order --------------------------------------------------
    disorder = []
    by_letter = {}
    for e in entries:
        by_letter.setdefault(e["letter"], []).append(e)
    for letter, es in by_letter.items():
        prev = None
        for e in es:
            key = collate(e["headword"] or "")
            if prev and key < prev[1] and abs(len(key) - len(prev[1])) < 30:
                disorder.append({"letter": letter, "after": prev[0],
                                 "id": e["id"]})
            prev = (e["id"], key)
    report["checks"]["alphabetical_order"] = \
        f"{len(disorder)} out-of-order neighbours (Estonian collation, loose)"
    report["issues"]["order_anomalies"] = disorder[:100]

    # 5. structure -----------------------------------------------------------
    no_stats = [e["id"] for e in entries
                if not e["reference_only"] and not e["source_counts"]]
    no_target = [e["id"] for e in entries
                 if e["reference_only"] and not e.get("see")]
    warn_counter = Counter(re.sub(r"\(.*?\)", "", w).strip()
                           for e in entries for w in e["warnings"])
    report["checks"]["entries_without_statistics"] = len(no_stats)
    report["checks"]["reference_entries_without_target"] = len(no_target)
    report["issues"]["entries_without_statistics"] = no_stats
    report["issues"]["reference_entries_without_target"] = no_target
    report["checks"]["warnings_summary"] = dict(warn_counter)
    report["issues"]["entries_with_warnings"] = [
        {"id": e["id"], "warnings": e["warnings"], "pages": e["pages"]}
        for e in entries if e["warnings"]]

    # summary ----------------------------------------------------------------
    report["summary"] = {
        "entries": len(entries),
        "reference_entries": sum(e["reference_only"] for e in entries),
        "entries_with_examples": sum(bool(e["examples"]) for e in entries),
        "total_examples": sum(len(e["examples"]) for e in entries),
        "total_source_count_items": sum(len(e["source_counts"]) for e in entries),
        "sum_of_totals": sum(e["total_count"] or 0 for e in entries),
        "letters": {l: len(es) for l, es in sorted(by_letter.items())},
    }

    out = JSON_DIR / "validation_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps(report["checks"], ensure_ascii=False, indent=2))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()

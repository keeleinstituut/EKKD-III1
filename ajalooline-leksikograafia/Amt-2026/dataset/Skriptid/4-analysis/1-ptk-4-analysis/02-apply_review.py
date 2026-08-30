#!/usr/bin/env python3
# Created: 2026-07-12 20-34-20
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
# Co-Authored-By: Claude Fable 5
"""Apply a batch of manual tag corrections and (re)write the corrected CSV.

The semantic judgment is made by the reviewer (me), not this script. This only
persists those decisions. Corrections are keyed by ROW INDEX (the source JSON
has duplicate Amt-Master-IDs, so the id is not a safe key). A master corrections
file accumulates across batches; after each merge the corrected CSV is rewritten
for all reviewed rows in index order — the file grows incrementally.

Batch file format (JSON): {"<idx>": {"Teema": "...", "Sugu": "M|N|Ü", "note": "..."}}
Only the keys present are overridden; omit Sugu/note to keep the auto value.

Usage:
    uv run python apply_review.py --batch /tmp/batch.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAGGED = HERE / "AMT-Master_4-3-tagged.json"
CORR = HERE / "4-3_corrections.json"
OUT_CSV = HERE / "4-3_tagging_review_corrected.csv"

VALID_TEEMA = {"NÕID", "SUGU_REPRO", "MORAAL_HÄLVE", "KIRIK_VAIMULIK", "KIRIK_FUNKTSIOON",
               "HARIDUS", "MÜÜT", "TEENISTUS", "HALDUS_VÕIM", "FEOD_MAA", "KÄSITÖÖ", "MUU"}
VALID_SUGU = {"M", "N", "Ü"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    args = ap.parse_args()

    rows = json.loads(TAGGED.read_text(encoding="utf-8"))["AMT-Master"]
    corrections = json.loads(CORR.read_text(encoding="utf-8")) if CORR.exists() else {}
    batch = json.loads(Path(args.batch).read_text(encoding="utf-8"))

    for idx, c in batch.items():
        i = int(idx)
        if not (0 <= i < len(rows)):
            raise ValueError(f"index {i} out of range")
        if "Teema" in c and c["Teema"] not in VALID_TEEMA:
            raise ValueError(f"idx {i}: bad Teema {c['Teema']!r}")
        if "Sugu" in c and c["Sugu"] not in VALID_SUGU:
            raise ValueError(f"idx {i}: bad Sugu {c['Sugu']!r}")
        corrections[str(i)] = {**corrections.get(str(i), {}), **c}

    CORR.write_text(json.dumps(corrections, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = ["idx", "Amt-Master-ID", "Amt-Cat", "Sem-Cat",
              "Teema_old", "Teema", "Sugu_old", "Sugu", "DE_hüperonüüm", "note", "DEF_en"]
    changed = 0
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i in sorted((int(k) for k in corrections), key=int):
            r = rows[i]
            c = corrections[str(i)]
            new_t, new_s = c.get("Teema", r.get("Teema")), c.get("Sugu", r.get("Sugu"))
            if new_t != r.get("Teema") or new_s != r.get("Sugu"):
                changed += 1
            w.writerow({
                "idx": i, "Amt-Master-ID": r["Amt-Master-ID"], "Amt-Cat": r.get("Amt-Cat"),
                "Sem-Cat": r.get("Sem-Cat"),
                "Teema_old": r.get("Teema"), "Teema": new_t,
                "Sugu_old": r.get("Sugu"), "Sugu": new_s,
                "DE_hüperonüüm": r.get("DE_hüperonüüm", ""), "note": c.get("note", ""),
                "DEF_en": r.get("DEF_en", ""),
            })
    print(f"Merged {len(batch)}. Reviewed {len(corrections)}/{len(rows)} "
          f"({changed} differ from auto) -> {OUT_CSV.name}")


if __name__ == "__main__":
    main()

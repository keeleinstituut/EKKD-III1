# Created: 2026-07-13 12-20-34
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
"""Recompute the Master's `Cross-source count` from actual attestation.

A source counts when `<Source>-et` is a real form, i.e. none of the
placeholders --- / ??? / NULL / empty. Rewrites AMT-Master_annotated.json in
place (key order untouched, ids untouched) and prints every changed concept.

Run dry by default; pass --write to modify the file.
"""
import json
import sys

MASTER_PATH = "Katus-ALUSANDMED/json-all/AMT-Master_annotated.json"
SOURCES = ["Stahl-1637", "Gutslaff-1648", "Göseken-1660",
           "Vestring-17XX", "Helle-1732", "Hupel-1780-est-ger"]
PLACE = {"---", "???", "NULL", "", " "}            # not a real attestation


def main():
    write = "--write" in sys.argv[1:]
    with open(MASTER_PATH, encoding="utf-8") as f:
        data = json.load(f)
    master = data["AMT-Master"]

    changed = 0
    for c in master:
        n = 0
        for s in SOURCES:
            et = c.get(f"{s}-et", "NULL")
            if isinstance(et, str) and et not in PLACE:
                n += 1
        old = str(c.get("Cross-source count", "NULL"))
        if old != str(n):
            print(f"  {c['Amt-Master-ID']:32} {old:>4} -> {n}")
            c["Cross-source count"] = n          # JSON integer per D8 (2026-07-13)
            changed += 1

    print(f"{changed} of {len(master)} concepts corrected"
          + ("" if write else " (dry run — nothing written)"))
    if write and changed:
        with open(MASTER_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"wrote {MASTER_PATH}")


if __name__ == "__main__":
    main()

# Created: 2026-06-18 21-41-49
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
"""Write reverse links: each edition entry gets a `master-id` list naming the
AMT-Master entries that link to it (inverse of the Master's `<Source>-id`).

Reads the forward links already present in the Master; re-dumps each edition in
canonical key order WITHOUT regenerating any uuid (existing ids are preserved).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import importlib
katus_lib = importlib.import_module("00-katus_lib")  # numbered filename requires importlib

JD = "Katus-ALUSANDMED/json-all"
SOURCES = ["Stahl-1637", "Gutslaff-1648", "Göseken-1660",
           "Vestring-17XX", "Helle-1732", "Hupel-1780-est-ger"]


def main():
    with open(f"{JD}/AMT-Master_annotated.json", encoding="utf-8") as f:
        master = json.load(f)["AMT-Master"]

    # edition entry id -> set of master ids
    rev = {}
    for m in master:
        mid = m["id"]
        for tag in SOURCES:
            for eid in m.get(f"{tag}-id", []):
                rev.setdefault(eid, set()).add(mid)

    grand = 0
    for tag in SOURCES:
        path = f"{JD}/{tag}.json"
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)[tag]
        linked = 0
        for e in entries:
            ids = sorted(rev.get(e["id"], ()))
            e["master-id"] = ids
            if ids:
                linked += 1
        katus_lib.dump(entries, path, tag)      # preserves existing ids
        grand += linked
        print(f"  {tag:22} {linked:6} / {len(entries):6} entries back-linked")
    print(f"  {'TOTAL':22} {grand} edition entries carry a master-id")


if __name__ == "__main__":
    main()

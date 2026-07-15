# Created: 2026-07-05 13-29-38
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
"""Replace placeholder ids ('to-be-added') with real am-<uuid7> ids in the
REVIEW copy of the Master. Same id scheme as inject_master_ids.py; only touches
entries whose id is the placeholder, leaving every real id untouched.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import katus_lib

PATH = "Katus-ALUSANDMED/Review-JSON-AMT/AMT-Master_annotated_REVIEW.json"
PLACEHOLDER = "to-be-added"


def main():
    with open(PATH, encoding="utf-8") as f:
        data = json.load(f)

    changed = []
    for e in data["AMT-Master"]:
        if str(e.get("id", "")).strip().lower() == PLACEHOLDER:
            e["id"] = katus_lib.new_id("AMT-Master")   # replace value in place
            changed.append((e.get("Amt-Master-ID"), e["id"]))

    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"{len(changed)} placeholder id(s) replaced:")
    for code, newid in changed:
        print(f"  {code!r}: {newid}")


if __name__ == "__main__":
    main()

# Created: 2026-06-18 16-58-06
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
"""Insert an 'id' (am-<uuid7>) right after 'Amt-Master-ID' in every entry."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import katus_lib

PATH = "Katus-ALUSANDMED/json-all/AMT-Master_annotated.json"


def main():
    with open(PATH, encoding="utf-8") as f:
        data = json.load(f)

    entries = data["AMT-Master"]
    out = []
    for e in entries:
        if "id" in e:                      # idempotent: keep existing id
            out.append(e)
            continue
        new = {}
        for k, v in e.items():
            new[k] = v
            if k == "Amt-Master-ID":
                new["id"] = katus_lib.new_id("AMT-Master")
        out.append(new)
    data["AMT-Master"] = out

    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"{len(out)} entries; first id = {out[0]['id']}")
    print("keys[0:3]:", list(out[0])[:3])


if __name__ == "__main__":
    main()

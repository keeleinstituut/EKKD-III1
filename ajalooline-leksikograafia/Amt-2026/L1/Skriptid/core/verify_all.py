# Created: 2026-07-05 19-13-38
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
"""Verify every produced JSON: schema, ids, key order, counts."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import katus_lib

JDIR = "Katus-ALUSANDMED/json-all"
EDITIONS = {
    "Stahl-1637": "st", "Gutslaff-1648": "gu", "Göseken-1660": "go",
    "Vestring-17XX": "ve", "Helle-1732": "he", "Hupel-1780-est-ger": "hu",
}
MWU_CORE = set(katus_lib.MWU_CORE)                       # always present
MWU_ALLOWED = MWU_CORE | set(katus_lib.MWU_EXTRA)        # + optional labels


def check(name, prefix):
    path = os.path.join(JDIR, f"{name}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert list(data) == [name], f"{name}: root key {list(data)}"
    entries = data[name]
    ids = set()
    for i, e in enumerate(entries):
        assert list(e) == katus_lib.CANONICAL_KEYS, \
            f"{name}[{i}] key order: {list(e)}"
        assert e["id"].startswith(prefix + "-"), f"{name}[{i}] id {e['id']}"
        assert e["source"] == name, f"{name}[{i}] source {e['source']}"
        ids.add(e["id"])
        assert isinstance(e["mwu"], list), f"{name}[{i}] mwu not list"
        for m in e["mwu"]:
            mk = set(m)
            assert MWU_CORE <= mk <= MWU_ALLOWED, f"{name}[{i}] mwu keys {list(m)}"
    assert len(ids) == len(entries), f"{name}: duplicate ids!"
    n_hw = sum(1 for e in entries if e["headword-et"] != "NULL")
    n_mwu = sum(len(e["mwu"]) for e in entries)
    print(f"  {name:14} entries={len(entries):6}  uniq_ids={len(ids):6}  "
          f"with_headword-et={n_hw:6}  mwu_items={n_mwu}")


def check_master():
    path = os.path.join(JDIR, "AMT-Master_annotated.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data["AMT-Master"]
    ids = set()
    for i, e in enumerate(entries):
        keys = list(e)
        assert keys[0] == "Amt-Master-ID" and keys[1] == "id", \
            f"master[{i}] keys {keys[:3]}"
        assert e["id"].startswith("am-"), f"master[{i}] id {e['id']}"
        ids.add(e["id"])
    assert len(ids) == len(entries), "master: duplicate ids!"
    print(f"  {'AMT-Master':14} entries={len(entries):6}  uniq_ids={len(ids):6}  "
          f"(id is 2nd key, am- prefix)")


def main():
    print("Verifying all JSON outputs:")
    check_master()
    for name, prefix in EDITIONS.items():
        check(name, prefix)
    # cross-file id uniqueness
    all_ids = []
    for name in list(EDITIONS) + ["AMT-Master_annotated"]:
        root = "AMT-Master" if name.startswith("AMT-Master") else name
        with open(os.path.join(JDIR, f"{name}.json"), encoding="utf-8") as f:
            all_ids += [e["id"] for e in json.load(f)[root]]
    print(f"\nTotal ids across all 7 files: {len(all_ids)}; "
          f"globally unique: {len(set(all_ids)) == len(all_ids)}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

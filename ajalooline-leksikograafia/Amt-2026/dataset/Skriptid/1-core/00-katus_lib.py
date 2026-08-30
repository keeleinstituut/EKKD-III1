# Created: 2026-07-07 12-44-14
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
# Co-Authored-By: Claude Fable 5
"""Shared helpers for Katus edition converters.

One canonical, harmonized entry schema is used by every edition JSON so the
files share an identical tagset (modelled on AMT-Master_annotated.json).
Conventions (NULL for empty cells, faithful string rendering) match
convert_amt_master.py.
"""
import json
import os

import uuid6

# Per-source uuid7 prefixes (non-digit, so ids never start with a digit).
PREFIX = {
    "AMT-Master": "am",
    "Stahl-1637": "st",
    "Gutslaff-1648": "gu",
    "Göseken-1660": "go",
    "Vestring-17XX": "ve",
    "Helle-1732": "he",
    "Hupel-1780-est-ger": "hu",
}

# Canonical harmonized key order. Every edition entry has EXACTLY these keys,
# in this order. Absent data -> "NULL" (scalars) or [] (mwu).
CANONICAL_KEYS = [
    "headword-et",    # Estonian lemma/form (for DE-keyed dicts: the ee vaste)
    "id",             # <prefix>-<uuid7>
    "equiv-de",       # German equivalent (for DE-keyed dicts: the märksõna)
    "source",         # edition tag, e.g. "Helle-1732"
    "master-id",      # list of AMT-Master ids that link to this entry (reverse link)
    "headword-modern",# normalised modern Estonian headword, if the source has one
    "explanation",    # explanatory note on the headword (Hupel :ex:)
    "pos",            # part of speech / sõnaliik
    "grammar",        # grammatical forms / notes
    "latin",          # Latin
    "meaning-et",     # source's own Estonian meaning / explanation
    "syn-et",         # Estonian synonym(s)
    "syn-de",         # German synonym(s)
    "example-et",     # Estonian example phrase
    "example-de",     # German translation of the example
    "mwu",            # list of {"mwu-et": str, "mwu-de": str}
    "variant",        # alternative spelling/form (Hupel ~)
    "dialect",        # dialect markers (Hupel :di:)
    "regional",       # regional name (Hupel :rn:)
    "usage",          # usage marker (Hupel :us:)
    "xref",           # cross-reference (Hupel :xr:)
    "page",           # page number
    "comment",        # editor/source comment(s)
]


# MWU-item schema. Core keys are always present; the extras appear ONLY on the
# items that actually carry that content (no NULL padding). `comment` holds only
# editorial notes (TOIM / OK / komm). Real content always goes under a label.
MWU_CORE = ("mwu-et", "mwu-de", "page", "comment")
MWU_EXTRA = ("pos", "grammar", "latin", "syn-et", "syn-de")
_EMPTY = {None, "NULL", "", " "}


def mwu(mwu_et, mwu_de, page="NULL", comment="NULL",
        pos="NULL", grammar="NULL", latin="NULL", syn_et="NULL", syn_de="NULL"):
    """A canonical MWU item. mwu-et/mwu-de/page/comment always present; the
    other labels are added only when non-empty, keeping items lean and readable.
    Key order: mwu-et, mwu-de, [extras…], page, comment."""
    item = {"mwu-et": mwu_et, "mwu-de": mwu_de}
    for key, val in (("pos", pos), ("grammar", grammar), ("latin", latin),
                     ("syn-et", syn_et), ("syn-de", syn_de)):
        if val not in _EMPTY:
            item[key] = val
    item["page"] = page
    item["comment"] = comment
    return item


def new_id(source):
    """Prefixed uuid7 id for a source, e.g. 'he-019edb04-...'."""
    return f"{PREFIX[source]}-{uuid6.uuid7()}"


def reuse_ids(entries, path, key="headword-et"):
    """Reuse `id` and `master-id` from an existing JSON at `path`, matched
    positionally, so re-running a converter does not break existing links.
    Applies only when the old file has the same entry count and the same
    sequence of `key` values; otherwise leaves the fresh ids untouched and
    reports why. Returns the number of entries whose id was reused."""
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        old = json.load(f)
    old_entries = old[next(iter(old))]          # single root key -> its list
    if len(old_entries) != len(entries):
        print(f"  id-reuse SKIPPED: {len(old_entries)} old vs {len(entries)} new entries")
        return 0
    mism = sum(1 for n, o in zip(entries, old_entries) if n.get(key) != o.get(key))
    if mism:
        print(f"  id-reuse SKIPPED: {mism} '{key}' mismatches — links would misalign")
        return 0
    for n, o in zip(entries, old_entries):
        n["id"] = o["id"]
        n["master-id"] = o.get("master-id", [])
    return len(entries)


def cell_to_str(v):
    """Faithful string rendering of a cell value; empty -> 'NULL'.

    Mirrors convert_amt_master.py: None / '' -> 'NULL'; ints/whole floats
    rendered without a decimal; whitespace preserved verbatim otherwise.
    """
    if v is None or v == "":
        return "NULL"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else str(v)
    return str(v).strip() if isinstance(v, str) else str(v)


def blank_entry(source):
    """A canonical entry pre-filled with NULL / [] and a fresh id."""
    e = {k: "NULL" for k in CANONICAL_KEYS}
    e["mwu"] = []
    e["master-id"] = []
    e["source"] = source
    e["id"] = new_id(source)
    return e


def order_entry(e):
    """Return a new dict with keys in CANONICAL_KEYS order (validates keys)."""
    extra = set(e) - set(CANONICAL_KEYS)
    if extra:
        raise ValueError(f"non-canonical keys: {sorted(extra)}")
    list_keys = {"mwu", "master-id"}
    return {k: (e[k] if k in e else ([] if k in list_keys else "NULL"))
            for k in CANONICAL_KEYS}


def dump(entries, out_path, root_key):
    """Write {root_key: [entries]} as UTF-8 JSON with 2-space indent."""
    data = {root_key: [order_entry(e) for e in entries]}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return len(entries)

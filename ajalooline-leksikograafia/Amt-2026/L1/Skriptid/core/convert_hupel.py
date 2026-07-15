# Created: 2026-07-07 12-44-21
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
# Co-Authored-By: Claude Fable 5
"""Convert Hupel-1780 .txt (custom markup) -> Hupel-1780.json.

Markup per Codebook-Hu-1780-et-de.txt. Entry blocks start at
`<entry xml:id="...">`, end at a blank line / next entry / page marker.
`:se:` introduces a nested sub-entry: it is emitted as its own harmonized
entry, linked to its parent via a comment note.
"""
import os
import re
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(__file__))
import katus_lib

SRC = "Katus-ALUSANDMED/new-editions/6-Hupel-1780_EKI_20260211_latest.txt"
OUT = "Katus-ALUSANDMED/json-all/Hupel-1780-est-ger.json"
SOURCE = "Hupel-1780-est-ger"

TAG_RE = re.compile(r"^:([A-Za-z0-9/_-]+):\s?(.*)$")
HEAD_RE = re.compile(r"^([*~])\s+(.*)$")
ENTRY_RE = re.compile(r'^<entry xml:id="(.*)">\s*$')
PAGE_RE = re.compile(r"^---\s*page\s+(\S+)\s*---\s*$")


def render_translations(senses):
    """senses: OrderedDict key->list[str]; '_' is the unnumbered group."""
    if not senses:
        return "NULL"
    keys = list(senses)
    if keys == ["_"]:
        return ", ".join(senses["_"])
    parts = []
    for k in keys:
        joined = ", ".join(senses[k])
        parts.append(joined if k == "_" else f"{k}. {joined}")
    return "; ".join(parts)


def build_entry(headword, lines, page, xml_id=None, parent=None):
    """lines: list of (kind, value). kind is the tag name, '~', '*', or '#'."""
    e = katus_lib.blank_entry(SOURCE)
    e["headword-et"] = headword if headword else "NULL"
    e["page"] = page if page else "NULL"

    variants, grammar, dialect, regional, usage, xref = [], [], [], [], [], []
    comments, latin, explanations = [], [], []
    senses = OrderedDict()
    mwus = []

    def add_tr(num, val):
        key = num if num else "_"
        senses.setdefault(key, []).append(val)

    for kind, val in lines:
        if kind == "~":
            variants.append(val)
        elif kind == "gr":
            grammar.append(val)
        elif kind == "tr" or kind == "t" or kind == "tr1":
            add_tr(None, val)
        elif kind.startswith("tr-") and kind[3:].isdigit():
            add_tr(kind[3:], val)
        elif kind == "tr-la":
            latin.append(val)
        elif kind == "mw":
            mwus.append({"mwu-et": val, "mwu-de": "NULL",
                         "page": "NULL", "comment": "NULL"})
        elif kind == "mw/tr":
            if mwus:
                mwus[-1]["mwu-de"] = val
            else:
                mwus.append({"mwu-et": "NULL", "mwu-de": val,
                             "page": "NULL", "comment": "NULL"})
        elif kind in ("di", "d"):
            dialect.append(val)
        elif kind == "rn":
            regional.append(val)
        elif kind == "us":
            usage.append(val)
        elif kind == "xr":
            xref.append(val)
        elif kind == "ex":
            explanations.append(val)
        elif kind == "#":
            comments.append(f"note: {val}")
        else:                                   # unknown tag -> keep, lossless
            comments.append(f"{kind}: {val}")

    e["equiv-de"] = render_translations(senses)
    e["explanation"] = "; ".join(explanations) if explanations else "NULL"
    e["variant"] = "; ".join(variants) if variants else "NULL"
    e["grammar"] = "; ".join(grammar) if grammar else "NULL"
    e["latin"] = "; ".join(latin) if latin else "NULL"
    e["dialect"] = " ".join(dialect) if dialect else "NULL"
    e["regional"] = "; ".join(regional) if regional else "NULL"
    e["usage"] = "; ".join(usage) if usage else "NULL"
    e["xref"] = "; ".join(xref) if xref else "NULL"
    e["mwu"] = mwus

    meta = []
    if xml_id:
        meta.append(f"xml:id: {xml_id}")
    if parent:
        meta.append(f"sub-entry of: {parent}")
    meta.extend(comments)
    e["comment"] = "; ".join(meta) if meta else "NULL"
    return e


def flush_block(xml_id, page, body, entries):
    headword = None
    main_lines = []
    subs = []                       # list of (sub_headword, lines)
    target = main_lines             # where new tag lines go
    for kind, val in body:
        if kind == "*":
            headword = val
        elif kind == "se":
            subs.append([val, []])
            target = subs[-1][1]
        else:
            target.append((kind, val))
    parent_ref = f"{headword} (xml:id {xml_id})" if headword else f"xml:id {xml_id}"
    entries.append(build_entry(headword, main_lines, page, xml_id=xml_id))
    for sub_head, sub_lines in subs:
        entries.append(build_entry(sub_head, sub_lines, page, parent=parent_ref))


def main():
    entries = []
    page = None
    xml_id = None
    body = []
    in_entry = False
    n_sub = 0

    def end_block():
        nonlocal body, in_entry, xml_id
        if in_entry and xml_id is not None:
            flush_block(xml_id, page, body, entries)
        body = []
        in_entry = False

    with open(SRC, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            mp = PAGE_RE.match(line)
            if mp:
                end_block()
                page = mp.group(1)
                continue
            me = ENTRY_RE.match(line)
            if me:
                end_block()
                xml_id = me.group(1)
                in_entry = True
                body = []
                continue
            if line.strip() == "":
                end_block()
                continue
            if not in_entry:
                continue
            mh = HEAD_RE.match(line)
            if mh:
                body.append((mh.group(1), mh.group(2).strip()))
                continue
            mt = TAG_RE.match(line)
            if mt:
                body.append((mt.group(1), mt.group(2).strip()))
                continue
            if line.startswith("#"):
                body.append(("#", line.lstrip("#").strip()))
                continue
            # stray continuation line -> attach to previous as comment
            body.append(("#", line.strip()))
    end_block()

    n_sub = sum(1 for e in entries if "sub-entry of:" in e["comment"])
    reused = katus_lib.reuse_ids(entries, OUT)   # keep ids + master-id stable
    n = katus_lib.dump(entries, OUT, SOURCE)
    print(f"ids reused: {reused}")
    print(f"entries: {n} (sub-entries: {n_sub}, main: {n - n_sub})")
    total_mwu = sum(len(e["mwu"]) for e in entries)
    print(f"total MWU items: {total_mwu}")


if __name__ == "__main__":
    main()

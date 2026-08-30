# Created: 2026-07-05 19-13-02
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
"""Convert Helle-1732 xlsx -> Helle-1732.json (harmonized schema).

Special rule: a headword containing a space is a multi-word unit (MWU). Such
rows are NOT independent entries — they are appended as additional MWUs to the
most recent single-word (base) entry that precedes them. Base entries may also
carry an MWU in the dedicated estonian_mwu/translated_mwu columns.
"""
import os
import sys

import openpyxl

sys.path.insert(0, os.path.dirname(__file__))
import importlib
katus_lib = importlib.import_module("00-katus_lib")  # numbered filename requires importlib

SRC = "Katus-ALUSANDMED/new-editions/5-Helle-1732_EKI_20250306_latest.xlsx"
SHEET = "ATH_Master-1_NT_20250306_output"
OUT = "Katus-ALUSANDMED/json-all/Helle-1732.json"
SOURCE = "Helle-1732"

# column index -> header (for reference)
# 0 estonian_headword 1 german_equivalent 2 part_of_speech 3 estonian_declension
# 4 estonian_synonyms 5 german_s-non-ms 6 latin_explanation 7 estonian_mwu
# 8 translated_mwu 9 page_number 10 TOIM 11 OK? +/- 12 TOIM komm 13 TOIM komm 2


def ne(v):
    return v is not None and str(v).strip() != ""


def join_comment(pairs):
    """pairs: list of (label, raw_value) -> 'label: val; ...' or 'NULL'."""
    parts = [f"{lab}: {katus_lib.cell_to_str(v)}" for lab, v in pairs if ne(v)]
    return "; ".join(parts) if parts else "NULL"


def cell(v):
    return katus_lib.cell_to_str(v)


def editorial_comment(r):
    """comment holds ONLY editorial columns (TOIM / OK / komm / komm2)."""
    return join_comment([
        ("TOIM", r[10]), ("OK", r[11]), ("komm", r[12]), ("komm2", r[13]),
    ])


def mwu_items(r):
    """MWU item(s) from an MWU row: the phrase itself (real content under proper
    labels, added only when present), plus any nested MWU the row carries in the
    estonian_mwu / translated_mwu columns (emitted as its own MWU item)."""
    items = [katus_lib.mwu(
        cell(r[0]), cell(r[1]), page=cell(r[9]), comment=editorial_comment(r),
        pos=cell(r[2]), grammar=cell(r[3]), latin=cell(r[6]),
        syn_et=cell(r[4]), syn_de=cell(r[5]))]
    if ne(r[7]) or ne(r[8]):
        items.append(katus_lib.mwu(cell(r[7]), cell(r[8]), page=cell(r[9])))
    return items


def base_entry(r):
    e = katus_lib.blank_entry(SOURCE)
    e["headword-et"] = katus_lib.cell_to_str(r[0])
    e["equiv-de"] = katus_lib.cell_to_str(r[1])
    e["pos"] = katus_lib.cell_to_str(r[2])
    e["grammar"] = katus_lib.cell_to_str(r[3])      # estonian_declension
    e["syn-et"] = katus_lib.cell_to_str(r[4])
    e["syn-de"] = katus_lib.cell_to_str(r[5])       # german_s-non-ms
    e["latin"] = katus_lib.cell_to_str(r[6])
    e["page"] = katus_lib.cell_to_str(r[9])
    e["comment"] = join_comment([
        ("TOIM", r[10]), ("OK", r[11]), ("komm", r[12]), ("komm2", r[13]),
    ])
    # dedicated MWU columns on a base row -> an MWU pair
    if ne(r[7]) or ne(r[8]):
        e["mwu"].append(katus_lib.mwu(cell(r[7]), cell(r[8]), page=cell(r[9])))
    return e


def main():
    wb = openpyxl.load_workbook(SRC, data_only=True)
    ws = wb[SHEET]
    rows = list(ws.iter_rows(min_row=2, values_only=True))

    entries = []
    last_base = None
    n_blank = n_mwu_attached = n_mwu_orphan = 0

    for r in rows:
        hw = r[0]
        if not ne(hw):
            if not any(ne(c) for c in r):
                n_blank += 1
            continue
        is_mwu = " " in str(hw).strip()
        if is_mwu and last_base is not None:
            last_base["mwu"].extend(mwu_items(r))
            n_mwu_attached += 1
        elif is_mwu:                      # MWU with no preceding base -> standalone
            e = base_entry(r)
            entries.append(e)
            last_base = e
            n_mwu_orphan += 1
        else:
            e = base_entry(r)
            entries.append(e)
            last_base = e

    reused = katus_lib.reuse_ids(entries, OUT)   # keep ids + master-id stable
    n = katus_lib.dump(entries, OUT, SOURCE)
    print(f"entries written: {n}")
    print(f"ids + master-id reused from existing file: {reused}")
    print(f"blank rows skipped: {n_blank}")
    print(f"MWU rows attached to a base entry: {n_mwu_attached}")
    print(f"MWU rows with no preceding base (kept standalone): {n_mwu_orphan}")
    total_mwu = sum(len(e["mwu"]) for e in entries)
    print(f"total MWU items across all entries: {total_mwu}")


if __name__ == "__main__":
    main()

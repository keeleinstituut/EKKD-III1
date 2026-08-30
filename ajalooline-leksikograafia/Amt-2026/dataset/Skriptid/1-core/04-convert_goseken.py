# Created: 2026-06-18 17-02-44
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
"""Convert Göseken-1660 .xls -> Göseken-1660.json (harmonized schema).

Old binary .xls (openpyxl cannot read it) -> use xlrd. Sheet 'sõnastik',
German-keyed. Only columns 0..9 carry data.
"""
import os
import sys

import xlrd

sys.path.insert(0, os.path.dirname(__file__))
import importlib
katus_lib = importlib.import_module("00-katus_lib")  # numbered filename requires importlib

SRC = "Katus-ALUSANDMED/new-editions/3-Goseken-1660_Kingisepp-et-al_latest.xls"
SHEET = "sõnastik"
OUT = "Katus-ALUSANDMED/json-all/Göseken-1660.json"
SOURCE = "Göseken-1660"
# 0 sks 1 ld 2 ee põhivorm 3 ee gr 4 märkused 5 MÄRKSÕNA 6 sõnaliik 7 Lk
# 8 sup.par 9 tähendus ja tähendusseletus


def ne(s):
    return s != "NULL" and s.strip() != ""


def main():
    wb = xlrd.open_workbook(SRC)
    ws = wb.sheet_by_name(SHEET)
    entries = []
    for ri in range(1, ws.nrows):                 # row 0 is the header
        raw = [ws.cell_value(ri, ci) for ci in range(min(10, ws.ncols))]
        vals = [katus_lib.cell_to_str(v) for v in raw]
        if not any(ne(v) for v in vals):
            continue
        e = katus_lib.blank_entry(SOURCE)
        e["equiv-de"] = vals[0]
        e["latin"] = vals[1]
        e["headword-et"] = vals[2]
        # grammar = ee gr, with sup.par (col 8) appended when present
        gr, sup = vals[3], vals[8]
        if ne(gr) and ne(sup):
            e["grammar"] = f"{gr}; sup.par: {sup}"
        elif ne(sup):
            e["grammar"] = f"sup.par: {sup}"
        else:
            e["grammar"] = gr
        e["comment"] = f"Gö märkused: {vals[4]}" if ne(vals[4]) else "NULL"
        e["headword-modern"] = vals[5]
        e["pos"] = vals[6]
        e["page"] = vals[7]
        e["meaning-et"] = vals[9]
        entries.append(e)
    n = katus_lib.dump(entries, OUT, SOURCE)
    print(f"entries: {n}; source rows (excl header): {ws.nrows - 1}")


if __name__ == "__main__":
    main()

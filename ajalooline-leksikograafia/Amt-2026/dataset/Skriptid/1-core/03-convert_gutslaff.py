# Created: 2026-06-18 17-02-14
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
"""Convert Gutslaff-1648 xlsx -> Gutslaff-1648.json (harmonized schema).

German-keyed dictionary: Estonian form -> headword-et, German -> equiv-de.
"""
import os
import sys

import openpyxl

sys.path.insert(0, os.path.dirname(__file__))
import importlib
katus_lib = importlib.import_module("00-katus_lib")  # numbered filename requires importlib

SRC = "Katus-ALUSANDMED/new-editions/2-Gutslaff-1648_EKI_20250103_latest.xlsx"
SHEET = "Sheet1"
OUT = "Katus-ALUSANDMED/json-all/Gutslaff-1648.json"
SOURCE = "Gutslaff-1648"
# A ee vaste B sks märksõna C ladina D ee süno E sks süno F lk G gramm H komm


def ne(v):
    return v is not None and str(v).strip() != ""


def main():
    wb = openpyxl.load_workbook(SRC, data_only=True)
    ws = wb[SHEET]
    entries = []
    for r in ws.iter_rows(min_row=2, max_col=8, values_only=True):
        if not any(ne(c) for c in r):
            continue
        e = katus_lib.blank_entry(SOURCE)
        e["headword-et"] = katus_lib.cell_to_str(r[0])
        e["equiv-de"] = katus_lib.cell_to_str(r[1])
        e["latin"] = katus_lib.cell_to_str(r[2])
        e["syn-et"] = katus_lib.cell_to_str(r[3])
        e["syn-de"] = katus_lib.cell_to_str(r[4])
        e["page"] = katus_lib.cell_to_str(r[5])
        e["grammar"] = katus_lib.cell_to_str(r[6])
        e["comment"] = katus_lib.cell_to_str(r[7])
        entries.append(e)
    n = katus_lib.dump(entries, OUT, SOURCE)
    print(f"entries: {n}")


if __name__ == "__main__":
    main()

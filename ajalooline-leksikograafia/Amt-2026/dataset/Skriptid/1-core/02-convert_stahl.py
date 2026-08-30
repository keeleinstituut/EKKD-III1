# Created: 2026-06-18 17-02-03
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
"""Convert Stahl-1637 xlsx -> Stahl-1637.json (harmonized schema)."""
import os
import sys

import openpyxl

sys.path.insert(0, os.path.dirname(__file__))
import importlib
katus_lib = importlib.import_module("00-katus_lib")  # numbered filename requires importlib

SRC = "Katus-ALUSANDMED/new-editions/1-Stahl-1637_Veskimae-Kikas_20241205_latest.xlsx"
SHEET = "Sheet1"
OUT = "Katus-ALUSANDMED/json-all/Stahl-1637.json"
SOURCE = "Stahl-1637"
# A modern B tähendus C pos D märksõna E sks vaste F gram G ladina H sks komm I lk


def ne(v):
    return v is not None and str(v).strip() != ""


def main():
    wb = openpyxl.load_workbook(SRC, data_only=True)
    ws = wb[SHEET]
    entries = []
    n_stub = 0
    for r in ws.iter_rows(min_row=2, max_col=9, values_only=True):
        if not any(ne(c) for c in r):
            continue
        e = katus_lib.blank_entry(SOURCE)
        # cross-reference stub: only column A filled and it contains an arrow
        only_a = ne(r[0]) and not any(ne(c) for c in r[1:])
        if only_a and "→" in str(r[0]):
            full = str(r[0]).strip()
            e["headword-modern"] = full
            e["xref"] = full.split("→", 1)[1].strip()
            entries.append(e)
            n_stub += 1
            continue
        e["headword-modern"] = katus_lib.cell_to_str(r[0])
        e["meaning-et"] = katus_lib.cell_to_str(r[1])
        e["pos"] = katus_lib.cell_to_str(r[2])
        e["headword-et"] = katus_lib.cell_to_str(r[3])
        e["equiv-de"] = katus_lib.cell_to_str(r[4])
        e["grammar"] = katus_lib.cell_to_str(r[5])
        e["latin"] = katus_lib.cell_to_str(r[6])
        e["comment"] = (f"sks komm: {katus_lib.cell_to_str(r[7])}"
                        if ne(r[7]) else "NULL")
        e["page"] = katus_lib.cell_to_str(r[8])
        entries.append(e)
    n = katus_lib.dump(entries, OUT, SOURCE)
    print(f"entries: {n}; cross-ref stubs: {n_stub}")


if __name__ == "__main__":
    main()

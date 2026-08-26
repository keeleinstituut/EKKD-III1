# Created: 2026-07-08 16-18-21
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
"""
Step 1/3 — extract styled text lines from APTSK_ALL.pdf.

Reads the dictionary body (PDF pages 21-926) of "Eesti vanema piiblitõlke
sõnastik 1600-1739" with PyMuPDF, drops running heads / page numbers, assigns
every line to its column, merges spans that sit on the same visual line and
maps each span's font to a semantic role. Output: build/spans.jsonl, one JSON
record per logical line in reading order, plus {"letter": ...} section markers.

Run:  uv run python APTSK-scripts/aptsk_01_extract_spans.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "APTSK_ALL.pdf"
OUT_PATH = ROOT / "build" / "spans.jsonl"

PAGE_FIRST, PAGE_LAST = 21, 926   # 1-based, inclusive — dictionary body
HEADER_Y_CUTOFF = 44.0            # lines above this are running heads / page numbers
COLUMN_SPLIT_X = 243.5            # page width 481.89; two columns
LINE_MERGE_TOL = 5.0              # max Δy for spans belonging to one visual line


def role_of(span):
    """Map a PyMuPDF span to a semantic role (see PLAN.md)."""
    font, size, flags = span["font"], span["size"], span["flags"]
    if size > 20:
        return "LETTER"
    if (flags & 1) and size < 7:
        return "SUP"                       # superscript homonym digit
    if font == "WarnockPro-Bold":
        return "HW"                        # headword / bold reference line
    if font == "WarnockPro-BoldIt":
        return "HL"                        # highlighted form inside example
    if font == "WarnockPro-It":
        return "IT" if size > 8.5 else "GI"  # example text / italic gloss part
    if font == "WarnockPro-Regular":
        return "REG" if size > 8.5 else "G"  # apparatus / gloss text
    if font.startswith("TimesNewRoman"):
        return "COMB"                      # combining diacritics (m̃ …)
    return "OTHER"


def extract():
    doc = fitz.open(PDF_PATH)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    other_fonts = Counter()
    letters = []

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for pno in range(PAGE_FIRST - 1, PAGE_LAST):
            page_no = pno + 1
            raw_lines = []  # (col, y0, x0, [(role, text), ...], is_letter)
            for block in doc[pno].get_text("dict")["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    y0, x0 = line["bbox"][1], line["bbox"][0]
                    if y0 < HEADER_Y_CUTOFF:
                        stats["header_lines_skipped"] += 1
                        continue
                    spans = []
                    is_letter = False
                    for s in line["spans"]:
                        if not s["text"]:
                            continue
                        r = role_of(s)
                        if r == "LETTER":
                            is_letter = True
                        if r == "OTHER":
                            other_fonts[(s["font"], round(s["size"], 1))] += 1
                        spans.append((r, s["text"], s["bbox"][0]))
                    if not spans:
                        continue
                    col = 0 if x0 < COLUMN_SPLIT_X else 1
                    raw_lines.append([col, y0, x0, spans, is_letter])

            # reading order: left column top-down, then right column
            raw_lines.sort(key=lambda l: (l[0], l[1], l[2]))

            # merge fragments on the same visual line (|Δy| <= tolerance)
            merged = []
            for col, y0, x0, spans, is_letter in raw_lines:
                if (merged and merged[-1][0] == col
                        and abs(merged[-1][1] - y0) <= LINE_MERGE_TOL
                        and not is_letter and not merged[-1][4]):
                    merged[-1][3].extend(spans)
                    merged[-1][1] = min(merged[-1][1], y0)
                else:
                    merged.append([col, y0, x0, spans, is_letter])

            for col, y0, x0, spans, is_letter in merged:
                spans.sort(key=lambda s: s[2])  # order fragments left-to-right
                if is_letter:
                    letter = "".join(t for _, t, _ in spans).strip()
                    letters.append((letter, page_no))
                    out.write(json.dumps({"letter": letter, "p": page_no},
                                         ensure_ascii=False) + "\n")
                    stats["letter_markers"] += 1
                    continue
                rec = {"p": page_no, "c": col, "y": round(y0, 1),
                       "s": [[r, t] for r, t, _ in spans]}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["lines"] += 1

    print(f"pages processed : {PAGE_LAST - PAGE_FIRST + 1}")
    print(f"body lines      : {stats['lines']}")
    print(f"header lines cut: {stats['header_lines_skipped']}")
    print(f"letter sections : {stats['letter_markers']} -> "
          + " ".join(f"{l}(p{p})" for l, p in letters))
    if other_fonts:
        print("WARNING unmapped fonts:", dict(other_fonts), file=sys.stderr)
    print(f"written: {OUT_PATH}")


if __name__ == "__main__":
    extract()

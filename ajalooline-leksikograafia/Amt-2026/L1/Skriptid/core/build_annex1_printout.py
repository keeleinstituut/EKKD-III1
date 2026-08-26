#!/usr/bin/env python3
# Created: 2026-07-12 15-21-06
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
"""Build Annex 1 dataset printout (PDF) from AMT-Master_annotated.json.

Layout: A4, two columns, Times New Roman 11 pt, 1-inch margins (matching
Katus-tervik-WIP docx), page numbers bottom center in 12 pt starting at 55.

Per entry:
    **Amt-Master-ID** <cross-source count>
    DEF_et in italics, no label
    et: forms grouped case-insensitively, displayed lowercased, with
        chronological author labels (St, Gu, Gö, Ve, He, Hu)
    de: forms grouped case-insensitively, displayed with the earliest
        source's spelling (German capitalization preserved)

Placeholders (---, ???, NULL) are not attestations and are skipped.
Compiles with tectonic.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "Katus-ALUSANDMED/json-all/AMT-Master_annotated.json"
OUT_DIR = ROOT / "Katus-DRAFTS/Katus-tervik-WIP"
TEX = OUT_DIR / "Annex-1_AMT-Master-printout.tex"

SOURCES = [
    ("Stahl-1637", "St"),
    ("Gutslaff-1648", "Gu"),
    ("Göseken-1660", "Gö"),
    ("Vestring-17XX", "Ve"),
    ("Helle-1732", "He"),
    ("Hupel-1780-est-ger", "Hu"),
]
PLACEHOLDERS = {"---", "???", "NULL", ""}

TEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def esc(s: str) -> str:
    return "".join(TEX_SPECIALS.get(c, c) for c in s)


def grouped_forms(entry: dict, lang: str) -> list[tuple[str, list[str]]]:
    """Group one entry's forms across sources, case-insensitively.

    Returns (display_form, [author labels]) in order of first attestation.
    et forms display lowercased; de forms keep the earliest source's spelling.
    """
    groups: dict[str, dict] = {}
    for src, label in SOURCES:
        raw = entry[f"{src}-{lang}"].strip()
        if raw in PLACEHOLDERS:
            continue
        key = raw.lower()
        g = groups.setdefault(key, {"display": key if lang == "et" else raw, "labels": []})
        g["labels"].append(label)
    return [(g["display"], g["labels"]) for g in groups.values()]


PREAMBLE = r"""\documentclass[11pt,a4paper,twocolumn]{article}
\usepackage[a4paper,margin=1in,footskip=0.5in]{geometry}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\usepackage{ragged2e}
\usepackage{needspace}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\fontsize{12}{14}\selectfont\thepage}
\renewcommand{\headrulewidth}{0pt}
\setlength{\columnsep}{28pt}
\setlength{\parindent}{0pt}
\raggedbottom
\setcounter{page}{55}

% entry head: lemma + cross-source count in angle brackets
\newcommand{\entryhead}[2]{%
  \needspace{3\baselineskip}%
  \hangindent1em\textbf{#1}\ $\langle$#2$\rangle$\par}

\begin{document}
\RaggedRight
\twocolumn[{\centering\bfseries\large Annex 1. AMT-Master dataset\par\vspace{10pt}}]
"""


def main() -> None:
    data = json.load(MASTER.open())["AMT-Master"]

    lines = [PREAMBLE]
    for entry in data:
        head = esc(entry["Amt-Master-ID"])
        count = esc(str(entry["Cross-source count"]).strip())
        lines.append(f"\\entryhead{{{head}}}{{{count}}}")
        definition = entry.get("DEF_et", "").strip()
        if definition not in PLACEHOLDERS:
            lines.append(f"\\textit{{{esc(definition)}}}\\par")
        for lang in ("et", "de"):
            parts = [
                f"{esc(form)} ({', '.join(labels)})"
                for form, labels in grouped_forms(entry, lang)
            ]
            if parts:
                lines.append(
                    f"\\hangindent1.6em {lang}:\\ {'; '.join(parts)}\\par"
                )
        lines.append("\\medskip")
        lines.append("")
    lines.append("\\end{document}")

    TEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {TEX}")

    r = subprocess.run(
        ["tectonic", str(TEX)], cwd=OUT_DIR, capture_output=True, text=True
    )
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        sys.exit(r.returncode)
    print(f"wrote {TEX.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

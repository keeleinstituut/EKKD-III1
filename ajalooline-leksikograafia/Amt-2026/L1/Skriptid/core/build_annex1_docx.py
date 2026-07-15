#!/usr/bin/env python3
# Created: 2026-07-12 15-49-56
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
"""Build Annex 1 dataset printout (DOCX) from AMT-Master_annotated.json.

Mirrors build_annex1_printout.py (the PDF/tectonic build): A4, two columns,
Times New Roman 11 pt, 1-inch margins, page numbers bottom center in 12 pt
starting at 55. Entries share the same grouping logic via import.
"""

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt

from build_annex1_printout import MASTER, OUT_DIR, grouped_forms
import json

DOCX = OUT_DIR / "Annex-1_AMT-Master-printout.docx"

FONT = "Times New Roman"
BODY_PT = Pt(11)
EM = Pt(11)  # 1 em at 11 pt


def set_two_columns(section, space_pt: int = 28) -> None:
    cols = section._sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        section._sectPr.append(cols)
    cols.set(qn("w:num"), "2")
    cols.set(qn("w:space"), str(space_pt * 20))  # twips


def set_page_start(section, start: int) -> None:
    pgnum = OxmlElement("w:pgNumType")
    pgnum.set(qn("w:start"), str(start))
    section._sectPr.append(pgnum)


def add_page_field(paragraph) -> None:
    run = paragraph.add_run()
    run.font.name = FONT
    run.font.size = Pt(12)
    for tag, attrs, text in (
        ("w:fldChar", {"w:fldCharType": "begin"}, None),
        ("w:instrText", {"xml:space": "preserve"}, " PAGE "),
        ("w:fldChar", {"w:fldCharType": "end"}, None),
    ):
        el = OxmlElement(tag)
        for k, v in attrs.items():
            el.set(qn(k), v)
        if text is not None:
            el.text = text
        run._r.append(el)


def styled(par, *, hang: Pt | None = None, after: Pt = Pt(0), keep_next=False):
    pf = par.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = after
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.widow_control = True
    if keep_next:
        pf.keep_with_next = True
    if hang is not None:
        pf.left_indent = hang
        pf.first_line_indent = -hang
    return par


def run(par, text: str, *, bold=False, size=BODY_PT):
    r = par.add_run(text)
    r.font.name = FONT
    r.font.size = size
    r.bold = bold
    return r


def main() -> None:
    data = json.load(MASTER.open())["AMT-Master"]

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = BODY_PT
    normal.paragraph_format.space_after = Pt(0)

    # section 1: full-width title
    s1 = doc.sections[0]
    s1.page_width, s1.page_height = Inches(8.27), Inches(11.69)  # A4
    for side in ("top", "bottom", "left", "right"):
        setattr(s1, f"{side}_margin", Inches(1))
    s1.footer_distance = Inches(0.5)
    set_page_start(s1, 55)

    footer_par = s1.footer.paragraphs[0]
    footer_par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_field(footer_par)

    title = styled(doc.add_paragraph(), after=Pt(10))
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run(title, "Annex 1. AMT-Master dataset", bold=True, size=Pt(14))

    # section 2: two-column body, continuous break
    s2 = doc.add_section(WD_SECTION.CONTINUOUS)
    set_two_columns(s2)

    for entry in data:
        head = styled(doc.add_paragraph(), hang=EM, keep_next=True)
        run(head, entry["Amt-Master-ID"], bold=True)
        run(head, f" ⟨{str(entry['Cross-source count']).strip()}⟩")

        lang_pars = []
        for lang in ("et", "de"):
            parts = [
                f"{form} ({', '.join(labels)})"
                for form, labels in grouped_forms(entry, lang)
            ]
            if parts:
                par = styled(doc.add_paragraph(), hang=Pt(17.6))  # 1.6 em
                run(par, f"{lang}: {'; '.join(parts)}")
                lang_pars.append(par)
        if lang_pars:
            lang_pars[0].paragraph_format.keep_with_next = len(lang_pars) > 1
            lang_pars[-1].paragraph_format.space_after = Pt(6)  # medskip
        else:
            head.paragraph_format.space_after = Pt(6)

    doc.save(DOCX)
    print(f"wrote {DOCX}")


if __name__ == "__main__":
    main()

# APTSK → JSON conversion pipeline (VPTS, Käsi jt 2025)

*Created: 2026-08-11 18-54-41 · Author: Madis Jürviste · Co-Authored-By: Claude Fable 5*

Scripts that convert *Eesti vanema piiblitõlke sõnastik 1600–1739* (EKI/EKSA
2025; viidatud tekstis kui VPTS, Käsi jt 2025) from the source PDF into
structured JSON, used as input for the AMT-Master ↔ APTSK overlap analysis.

Copied from a separate working repository (scripts only — the source PDF, the
intermediate `build/spans.jsonl` and the output `json/` dataset are
deliberately **not** included here; they live in that repo).

## Pipeline

| Step | Script | In → Out |
|------|--------|----------|
| 1 | `aptsk_01_extract_spans.py` | `APTSK_ALL.pdf` (pages 21–926) → `build/spans.jsonl` (styled lines, font→role mapping) |
| 2 | `aptsk_02_parse_entries.py` | `build/spans.jsonl` → `json/entries_<LETTER>.json`, `json/all_entries.json`, `json/manifest.json` |
| 3 | `aptsk_03_validate.py` | losslessness / frequency-arithmetic / vocabulary / ordering checks on the parsed JSON |
| 4 | `aptsk_04_build_html.py` | `json/` → single-file HTML view of the dictionary (`APTSK_sonastik_<date>.html`) |

`PLAN.md` documents the entry schema and parsing decisions.

## Diff against AMT-Master

The overlap analysis itself is `16-aptsk_master_diff.py` (published under
`1-core/`). It reads the `json/entries_*.json` produced above plus
`Katus-ALUSANDMED/json-all/AMT-Master_annotated.json` and writes:

- `Katus-ALUSANDMED/APTSK-Master_diff_report.md` — tiered results (the
  figures themselves are reported in the dissertation's framing chapter)
- `Katus-ALUSANDMED/APTSK-Master_diff_detail.csv` — one row per Master lemma

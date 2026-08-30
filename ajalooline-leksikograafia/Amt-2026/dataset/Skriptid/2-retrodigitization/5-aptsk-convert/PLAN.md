*Created: 2026-07-08 16-18-21 · Author: Madis Jürviste · Co-Authored-By: Claude Fable 5*

# Extraction plan — Eesti vanema piiblitõlke sõnastik 1600–1739 (APTSK_ALL.pdf)

Source: `APTSK_ALL.pdf`, 926 pages, InDesign-produced, fully embedded text.
Dictionary body: PDF pages **21–926** (front matter pages 1–20 is skipped per task).
28 letter sections (A B C D E F G H I J K L M N O P R S Š Z T U V W Õ Ä Ö Ü),
each starting at the top of a fresh page with a 60 pt display letter.

## 1. Entry structure plan

Typography (font → role), verified against the PDF:

| Font / size / flags               | Role                                        |
|-----------------------------------|---------------------------------------------|
| WarnockPro-Bold 9 pt               | `HW` — headword; also full "X vt Y" reference lines |
| WarnockPro-Regular 5.2 pt, superscript | `SUP` — homonym number (e.g. Aabel¹)     |
| WarnockPro-Regular 9 pt            | `REG` — counts, source labels, `(refs)`, "vt ka" |
| WarnockPro-Regular 8 pt            | `G` — gloss text inside `<…>`               |
| WarnockPro-It 8 pt                 | `GI` — italic segments inside gloss         |
| WarnockPro-It 9 pt                 | `IT` — quoted example text; italic subforms in statistics line |
| WarnockPro-BoldIt 9 pt             | `HL` — highlighted headword forms inside examples |
| WarnockPro-SemiboldDisp 60 pt      | `LETTER` — section heading                  |
| TimesNewRomanESUT 9 pt             | `COMB` — combining diacritics (m̃ etc.), inherit neighbouring role |

Entry anatomy in print:

```
headword[¹] [<gloss>] TOTAL            ← bold + optional sup digit + 8pt gloss + int
SRC [form[*|?]] N, SRC [form] N, …     ← per-source statistics ("allmärksõnad")
SRC example text … (Ref[, Ref]); SRC … ← examples, italic, bold-italic highlights
vt ka x, y-; z                         ← optional cross-references
```
plus pure reference entries: `aajast, aajasta vt aasta` (entirely bold).

Markers (per the dictionary's own introduction):
- `?` before a subform — lemma uncertain (only inflected forms attested);
- `*` after a subform — original orthography retained (spelling unreliable);
- references like `(Ilm22:13)`, `(1Ms2:7, 1Ms4:10)`, `(Ilm8 algus)`, `(Mt23:34 pk algus)`;
- `vt ka` list: items after `;` are compounds where the headword appears in a non-base form.

### Target JSON entry schema

```jsonc
{
  "id": "aabel-1",                  // unique slug (headword + homonym [+ counter])
  "letter": "A",                    // section letter
  "headword": "Aabel",              // primary headword (modern-form lemma)
  "headwords": ["Aabel"],           // all comma-separated headwords of the article
  "homonym": 1,                     // superscript homonym number, null if none
  "gloss": "…",                     // <…> content, plain text, null if absent
  "gloss_rich": [{"text": "…", "italic": false}], // gloss with italics preserved
  "reference_only": false,          // true for "X vt Y" entries
  "see": [{"form": "aasta", "homonym": null}],    // vt-targets (reference entries)
  "total_count": 40,                // total frequency printed after headword
  "source_counts": [                // per-source statistics line
    {"source": "Rs", "form": null, "form_raw": null, "count": 1,
     "uncertain": false, "original_spelling": false}
  ],
  "examples": [
    {"source": "GtUT",
     "text": "M. olle se A. n. O. …",       // dehyphenated quotation
     "highlighted": ["A"],                  // bold-italic form(s) in the quote
     "references": ["Ilm22:13"]}            // verse reference(s)
  ],
  "see_also": {"raw": "vt ka aia-, luhthein",
               "same_form": ["aia-", "luhthein"], "other_form": []},
  "pages": [21],                    // PDF pages the entry spans
  "raw_lines": ["…", "…"],          // exact printed lines, verbatim — lossless backup
  "warnings": []                    // parser notes, empty when clean
}
```

Further fields added while hardening against the book's typography:
- `counts_omitted` (bool) — numerals/pronouns are printed as a plain source
  list without frequencies; then `total_count` is null by design;
- `headword_details` — per headword: `form`, `homonym`, `uncertain` (`?`),
  `original_spelling` (`*`), `foreign` (component set in bold italics, i.e.
  foreign-language material such as `jähvke`, `schilt`);
- `source_counts[].forms` — `~`-separated variant subforms as a list;
- `see[]`/`see_also` items carry the same marker flags as headwords;
- `unparsed` (rare) — text the structural parser could not place, preserved
  verbatim alongside a warning.

All 16 entries with `warnings` document misprints in the book itself
(missing `<`/`>`/`(`/`)`, roman type where italics belong, a missing source
label, totals absent in print, `Gt`/`GUT`/`RS` label variants); nothing is
dropped in any of these cases.

Losslessness: every printed body line is stored verbatim in exactly one entry's
`raw_lines`; the validator re-concatenates them and character-compares against the
full extracted page text. Line-break dehyphenation in `text` fields is heuristic
(drop `-` when the next line starts lower-case, always drop soft hyphens U+00AD);
`raw_lines` keeps the original segmentation.

## 2. Script plan

`APTSK-scripts/` (run in order, all via `uv run python`):

1. **aptsk_01_extract_spans.py** — PDF → `build/spans.jsonl`.
   Reads pages 21–926 with PyMuPDF span data; drops running heads and page
   numbers (line y0 < 44); assigns each line to a column (x < 243.5 → left);
   merges visually-identical lines (|Δy| ≤ 5); maps fonts to roles; emits one
   JSON record per logical line in reading order plus LETTER markers.
2. **aptsk_02_parse_entries.py** — `build/spans.jsonl` → `json/`.
   Segments lines into entries (new entry = line starting with HW role, unless
   the previous header line is unfinished); builds per-entry character/role
   arrays with dehyphenation; parses header, statistics, examples, `vt ka`;
   writes `json/entries_<LETTER>.json`, `json/all_entries.json`, `json/manifest.json`.
   Any unparseable remainder is preserved in the entry under `"unparsed"`.
3. **aptsk_03_validate.py** — checks → `json/validation_report.json`:
   frequency sums vs printed totals, full-text coverage (lossless proof),
   source-abbreviation and reference-format checks, alphabetical order,
   headword-vs-section letter, parser warnings.

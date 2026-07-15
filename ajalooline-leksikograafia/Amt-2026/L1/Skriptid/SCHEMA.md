# Katus edition JSON — frozen harmonized schema

*Created: 2026-07-13 14-32-05 · Author: Madis Jürviste · Co-Authored-By: Claude Opus 4.8; Claude Fable 5*

Every edition JSON is `{ "<Source>": [ entries ] }` where each entry has
**exactly** the canonical keys, in the order defined in `katus_lib.CANONICAL_KEYS`:

```
headword-et, id, equiv-de, source, master-id, headword-modern, explanation,
pos, grammar, latin, meaning-et, syn-et, syn-de, example-et, example-de, mwu,
variant, dialect, regional, usage, xref, page, comment
```

`explanation` holds an explanatory note on the headword (Hupel `:ex:`; the
source places it immediately after the headword, hence its slot right after
`headword-modern`). Join multiple values with `; `.

Rules:
- `headword-et` = Estonian form; `equiv-de` = German form. This holds even for
  German-keyed dictionaries (Gutslaff, Göseken): Estonian goes in `-et`, German in `-de`.
- `id` = `katus_lib.new_id("<Source>")` (second key, right under the headword).
- `source` = the edition tag string, e.g. `"Stahl-1637"`.
- Empty / missing → `"NULL"` (use `katus_lib.cell_to_str`). `mwu` empty → `[]`.
- `mwu` = list of `{"mwu-et": str, "mwu-de": str}`.
- **Lossless**: every native column must land somewhere. If it doesn't fit a
  canonical field, append it to `comment` as `label: value` (join multiple with `; `).
- Build each entry with `katus_lib.blank_entry(source)`, fill fields, pass to
  `katus_lib.dump(entries, out_path, source)` (it validates key order).
- Skip fully-empty rows. Use `data_only=True` for xlsx.

Output: `Katus-ALUSANDMED/json-all/<Source>.json` (e.g. `Stahl-1637.json`).
Converter script: `converters/convert_<source>.py`, run from repo root via
`uv run python converters/convert_<source>.py`. Import helpers with:
`import sys, os; sys.path.insert(0, os.path.dirname(__file__)); import katus_lib`

---

## Per-source column maps (header row = row 1; data from row 2)

### Stahl-1637  (`1-Stahl-1637_Veskimae-Kikas_20241205_latest.xlsx`, sheet `Sheet1`, ET-keyed)
| col | header | → field |
|-----|--------|---------|
| A `tänapäevane vaste` | modern Estonian headword | `headword-modern` |
| B `tähendus` | meaning | `meaning-et` |
| C `sõnaliik/fraasiliik` | POS | `pos` |
| D `St märksõna` | Stahl Estonian form | `headword-et` |
| E `saksakeelne vaste` | German equiv | `equiv-de` |
| F `St gram-vormid` | grammar | `grammar` |
| G `St ladina` | Latin | `latin` |
| H `sks komm` | German comment | `comment` (prefix `sks komm: `) |
| I `leheküljenumber` | page | `page` |

Note: rows where only column A is filled and it contains `→` are cross-reference
stubs (e.g. `aasta → iga aasta`). Keep them as entries: `headword-modern` = the
text, put the arrow target into `xref`. (`headword-et` stays NULL.)

### Gutslaff-1648  (`2-Gutslaff-1648_EKI_20250103_latest.xlsx`, sheet `Sheet1`, DE-keyed)
| col | header | → field |
|-----|--------|---------|
| A `Eesti vaste Gu` | Estonian form | `headword-et` |
| B `Saksa märksõna` | German headword | `equiv-de` |
| C `Ladina` | Latin | `latin` |
| D `Eesti süno` | Estonian synonym | `syn-et` |
| E `Saksa süno` | German synonym | `syn-de` |
| F `Lk-nr Gu` | page | `page` |
| G `Gramm` | grammar | `grammar` |
| H `Komm MJ-TP-SS` | comment | `comment` |

(Columns I–M are empty; ignore.)

### Göseken-1660  (`3-Goseken-1660_Kingisepp-et-al_latest.xls`, **.xls** → use pandas+xlrd or xlrd, sheet `sõnastik`, DE-keyed)
| col | header | → field |
|-----|--------|---------|
| 0 `Gö sks` | German headword | `equiv-de` |
| 1 `Gö ld` | Latin | `latin` |
| 2 `Gös ee põhivorm` | Göseken Estonian form | `headword-et` |
| 3 `Gö ee gr` | Estonian grammar | `grammar` |
| 4 `Gö märkused` | notes | `comment` (prefix `Gö märkused: `) |
| 5 `MÄRKSÕNA` | modern Estonian headword | `headword-modern` |
| 6 `sõnaliik` | POS | `pos` |
| 7 `Lk` | page | `page` |
| 8 `sup. par` | (grammatical) | `grammar` (append `; sup.par: …` if both present) |
| 9 `tähendus ja tähendusseletus` | meaning/explanation | `meaning-et` |

Read with `xlrd` directly (`xlrd.open_workbook(...).sheet_by_name("sõnastik")`)
or `pandas.read_excel(..., sheet_name="sõnastik", header=0, dtype=str)`. Stop at
the last row with any content; ignore trailing all-empty columns (10–18).

### Vestring-17XX  (`4-Vestring-1720_Veskimae_20250116_latest.xlsx`, sheet `Sheet1`, ET-keyed)
(source tag / filename use `Vestring-17XX` to match the Master's column label.)
| col | header | → field |
|-----|--------|---------|
| A `eestikeelne vaste` | Estonian form | `headword-et` |
| B `saksakeelne vaste` | German equiv | `equiv-de` |
| C `grammatika jm kommentaar` | grammar/comment | `grammar` |
| D `sünonüüm` | synonym | `syn-et` |
| E `ee näide` | Estonian example | `example-et` |
| F `saksa tõlge` | German translation of example | `example-de` |
| G `toimetaja märkus` | editor note | `comment` |
| H `leheküljenumber` | page | `page` |

Worksheet is unsized — load without `read_only` and iterate to `ws.max_row`.

---

## AMT-Master_annotated.json — value markers and placeholders (legend)

- `---` — not attested in that source (lemma has no entry there).
- `NULL` — no value. Special case `Vestring-17XX-de`: attested in Vestring but the
  source gives no German gloss (decision 2026-07-13). Comment-1/2/3 are always
  `NULL` — the comment archive is `Katus-DRAFTS/Comments-in-JSON_2026-07-13.md`.
- `???` — unresolved/illegible in source (reserved; currently unused).
- `[xN]` after a word form — the form occurs N times in that source's entry
  (only `[xN]`; `[N]`/`[Nx]` are violations).
- `[Gram]`, `[er]` etc. and `{…}` braces — grammatical/editorial markup imported
  verbatim from Vestring/Göseken transcriptions.
- `Sem-Cat` — comma-separated, fixed salience order:
  elukutse > roll > tegija > omadus > sugulane > müt > rahvas > tiitel > inimene.
- `Sugu` — M/N only with a clear gender marker (ET word, e.g. "-mees" ⇒ M; DE gloss;
  or unambiguous sense); Ü wherever both genders are present or the sense allows either.
- `Cross-source count` — JSON integer: number of sources with a real
  (non-placeholder) `-et` attestation.
- Record order — Estonian collation, space/hyphen before letters.
- Validator: `uv run python scripts/validate_master.py` (exit 0 = clean).

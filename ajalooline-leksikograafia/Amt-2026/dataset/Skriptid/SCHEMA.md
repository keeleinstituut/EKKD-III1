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
Converter scripts are published under `1-core/` (`02-convert_stahl.py` …
`07-convert_hupel.py`); in the working repo they run from the repo root via
`uv run python <converter>.py`. Import helpers with:
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

### Helle-1732  (`5-Helle-1732_EKI_20250306_latest.xlsx`, sheet `ATH_Master-1_NT_20250306_output`, ET-keyed)
| col | header | → field |
|-----|--------|---------|
| 0 `estonian_headword` | Estonian form | `headword-et` |
| 1 `german_equivalent` | German equiv | `equiv-de` |
| 2 `part_of_speech` | POS | `pos` |
| 3 `estonian_declension` | declension | `grammar` |
| 4 `estonian_synonyms` | Estonian synonyms | `syn-et` |
| 5 `german_s-non-ms` | German synonyms | `syn-de` |
| 6 `latin_explanation` | Latin | `latin` |
| 7 `estonian_mwu` | MWU, Estonian | `mwu[].mwu-et` |
| 8 `translated_mwu` | MWU, German | `mwu[].mwu-de` |
| 9 `page_number` | page | `page` |
| 10–13 `TOIM` / `OK? +/-` / `TOIM komm` / `TOIM komm 2` | editorial columns | `comment` (labels `TOIM: / OK: / komm: / komm2:`; **only** these four ever land in `comment`) |

Special rules: a headword containing a space is a **multi-word unit row**, not an
independent entry — it is appended to the most recent single-word base entry's
`mwu[]` (carrying its pos/grammar/latin/syn values as MWU-item fields; an MWU
row with no preceding base is kept standalone). Base rows may additionally
carry their own MWU pair from cols 7–8. The converter calls
`katus_lib.reuse_ids()`, so re-running it keeps existing ids and links.

### Hupel-1780  (`6-Hupel-1780_EKI_20260211_latest.txt`, line-based markup, ET-keyed; source tag `Hupel-1780-est-ger`)
Not a spreadsheet: a text file in the custom markup produced by the
retrodigitization chain (tagset defined in the codebook
`Codebook-Hu-1780-et-de.txt`, a working document not included in this
release). An entry block runs from `<entry xml:id="...">` to a blank line, a
`--- page N ---` marker, or the next `<entry>`.

| line / tag | meaning | → field |
|------------|---------|---------|
| `* headword` | headword | `headword-et` |
| `~ variant` | variant form | `variant` |
| `:gr:` | grammar | `grammar` |
| `:tr:` `:t:` `:tr1:` | translation (unnumbered sense) | `equiv-de` |
| `:tr-N:` | numbered sense | `equiv-de` (rendered `1. a, b; 2. c`) |
| `:tr-la:` | Latin | `latin` |
| `:mw:` + `:mw/tr:` | MWU pair | `mwu[]` |
| `:di:` `:d:` | dialect marker | `dialect` |
| `:rn:` | regional marker | `regional` |
| `:us:` | usage marker | `usage` |
| `:xr:` | cross-reference | `xref` |
| `:ex:` | explanatory note | `explanation` |
| `:se:` | nested sub-entry | emitted as its **own entry**, `comment` = `sub-entry of: <headword> (xml:id …)` |
| `# note` / stray continuation line | note | `comment` (`note: …`) |
| unknown `:tag:` | anything else | `comment` (`tag: value` — lossless, nothing dropped) |

The `xml:id` is preserved in `comment` (`xml:id: …`). The converter calls
`katus_lib.reuse_ids()`, so re-running it keeps existing ids and links.

---

## AMT-Master_annotated.json — value markers and placeholders (legend)

- `---` — not attested in that source (lemma has no entry there).
- `NULL` — no value. Special case `Vestring-17XX-de`: attested in Vestring but the
  source gives no German gloss (decision 2026-07-13). Comment-1/2/3 are always
  `NULL` — the comments are archived in a working document
  (`Comments-in-JSON_2026-07-13.md`, not included in this release).
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
- Validator: `uv run python 1-core/13-validate_master.py <path/to/master.json>` (exit 0 = clean).

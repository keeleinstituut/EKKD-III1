# DQ fixes — consolidated changelog, run of 2026-07-13

*Created: 2026-07-13 14-54-15 · Author: Madis Jürviste · Co-Authored-By: Claude Fable 5*

- **Target:** `Katus-ALUSANDMED/json-all/AMT-Master_annotated.json` (882 records, 30-key schema).
- **Diff baseline / pre-run backup:** `Katus-ALUSANDMED/json-all/AMT-Master_annotated.BACKUP-2026-07-13.json` (read-only, untouched).
- **Run scope:** tasks T1–T20 of `DQ-tasklist_AMT-Master_2026-07-12.md`, with decisions D1 (salience tag order), D2 (NULL marker), D3 (case-only = duplicates), D4 (teeäärne/äärne Ü; global "-mees" ⇒ M sweep), D5 (b: plural DEFs), D6 (a: delete moot comments), D7 (approve `(sks …)` tails), D8 (CSC → integer), plus D9 (lemma normalizations), D10 (variant freeze note into review F-section), D12 (vaht family ⇒ TEENISTUS). D11 is intentionally NOT applied: the T11 feedback file awaits the user's manual review before any Sem-Cat fold-in.
- Executed by three agents (Section 1; Section 2 + D-sweeps; final T20 + VERIFY). Exact old → new values below are verbatim from the executing agents' logs — the T17/T19 DEF pairs are the user's D7/D5 spot-check material.
- Serialization preserved throughout: `json.dump(..., ensure_ascii=False, indent=2)` + trailing newline; record order = Estonian collation (space/hyphen before letters); edition JSONs never modified.

---

# SECTION 1 (T1–T12)

## T1 — trailing space in primary key (review B1)

- `maavalitseja ` . `Amt-Master-ID`: `maavalitseja ` → `maavalitseja`
- Collation re-check (G0.3): record stays at position 335, between `maasundija` and `maavanemad` — no move needed (space sorts before letters, so dropping a *trailing* space cannot change relative order).
- **Downstream artifacts embedding the spaced key `"maavalitseja "` need regeneration (NOT regenerated here, per task):**
  - APTSK T1 list (aptsk_master_diff output)
  - Annex-1 printout / docx (build_annex1_printout.py / build_annex1_docx.py output)
  - LexLex viewer (`LexLex.html`, build_viewer.py output)

## T2 — lemma rename tolwan → tolvan (review B2)

- `tolwan` . `Amt-Master-ID`: `tolwan` → `tolvan`
- Attested form untouched: `Hupel-1780-est-ger-et` still `tolwan` (verified).
- Collation re-check (G0.3): record stays at position 696, between `tollipealine` and `tont` — no move needed (l < v < n at the deciding position, order already correct).
- Acceptance: exactly one `tolvan` record, zero `tolwan` records. PASS.

## T3 — exact duplicate word-forms within one field (review B3, exact dupes only)

- `abinaine` . `Göseken-1660-et`: `Abbi Naine; abbi Naine; Abbi nain; Abbi Naine` → `Abbi Naine; abbi Naine; Abbi nain` (second exact `Abbi Naine` dropped; case variant `abbi Naine` KEPT — that is D3/T15)
- `hoorasundija` . `Göseken-1660-et`: `Hohra sundia; hohra Sundia; Hohra sundia` → `Hohra sundia; hohra Sundia`
- Acceptance: no `-et`/`-de` field contains the same exact trimmed token twice (global scan). PASS.

## T4 — non-breaking spaces U+00A0 (review B4)

Rule applied: U+00A0 → U+0020, collapse resulting double spaces, strip.

- `kõigist varastest vargam` . `Vestring-17XX-et`: `keikede wargade wargam\xa0` → `keikede wargade wargam`
- `kõik ilmamaa hulgus` . `Vestring-17XX-de`: `Der in d gantzen Welt herum\xa0{läufft}\xa0schwärmt` → `Der in d gantzen Welt herum {läufft} schwärmt`
- `käsiline` . `Vestring-17XX-de`: `Ein Gehülffe,\xa0d zur Hand gehet.\xa0it.\xa0Ein Handgriff` → `Ein Gehülffe, d zur Hand gehet. it. Ein Handgriff`
- `ninamees` . `Vestring-17XX-de`: `Ein Redelsführer\xa0it.\xa0{Weis-naser}\xa0Nasen weiser` → `Ein Redelsführer it. {Weis-naser} Nasen weiser`
- `vahva kõnemees` . `Vestring-17XX-et`: `Wahwa Könne Mees\xa0` → `Wahwa Könne Mees`
- `vahva tööinimene` . `Vestring-17XX-et`: `Wahwa Tö Innimenne\xa0` → `Wahwa Tö Innimenne`
- `valuvõtja` . `Vestring-17XX-et`: `Wallo wotja\xa0` → `Wallo wotja`
- `õige varas` . `Vestring-17XX-de`: `ein\xa0{Haupt}\xa0Ertz-Dieb` → `ein {Haupt} Ertz-Dieb`
- Acceptance: zero U+00A0 anywhere in the file (global scan). PASS.

## T5 — ordinary leading/trailing whitespace (review B5)

- `liigmees` . `Vestring-17XX-de`: `Eine Zeuge, Advocat. ` → `Eine Zeuge, Advocat.`
- `majamees` . `Helle-1732-de`: `ein guter æconomus ` → `ein guter æconomus`
- `tont` . `Helle-1732-de`: `das Gespenst ` → `das Gespenst`
- `võõras` . `Helle-1732-de`: `fremde, ein Gast ` → `fremde, ein Gast`
- `üleastuja` . `Helle-1732-de`: `der Uebertreter ` → `der Uebertreter`
- `ülem` . `Helle-1732-de`: `der Obere, Vornehmste ` → `der Obere, Vornehmste`
- `ümberkaudne rahvas` . `Vestring-17XX-et`: `Ümberkaudne rahwas ` → `Ümberkaudne rahwas`
- `pealik` . `Comment-1`: `Hu "peälik" on Rev: siin veel eP vok-harmoonia? ` → `Hu "peälik" on Rev: siin veel eP vok-harmoonia?`
- Acceptance: no string value anywhere satisfies `v != v.strip()` (global scan). PASS.

## T6 — NFC normalization (review B6)

All three cells contained a decomposed `ü` (`u` + U+0308 COMBINING DIAERESIS); NFC composes it to U+00FC. Visible text unchanged, byte representation only.

- `soldat` . `Comment-1`: decomposed `ü` in `üle` (…`soldateid üle vaatama`…) → composed. Full value otherwise identical: `Gö annab näitelauses "neid soldateid munstrima; neid soldateid üle vaatama". Ve annab näitelauses "Nemmad tahtwad neid Soldatit, ühhele polelle Jögge, et polle tarbis ülle Jöggede minna.".`
- `viks tegija` . `Comment-1`: decomposed `ü` in `Kül` → composed (see also T7 below on the same cell).
- `ülbe ja üleannetu inimene` . `Vestring-17XX-et`: decomposed `ü` in `ülle` → composed: `Ülbe ja ülleanneto Innime`
- Acceptance: every string value in the file satisfies `NFC(v) == v` (global scan). PASS.

## T7 — double spaces (review B7)

- `meheeksja` . `Vestring-17XX-et`: `Mehhe  eksia [x2]` → `Mehhe eksia [x2]`
- `vaim` . `Comment-1`: `St ja Gu ja Gö ja Ve "vaim" on 'vaim', sks  "Geist", vrd abivaim. ENT Hu annab AMT!` → `St ja Gu ja Gö ja Ve "vaim" on 'vaim', sks "Geist", vrd abivaim. ENT Hu annab AMT!`
- `viks tegija` . `Comment-1`: (after T6's NFC) `Ve annab s.v. "Wiks" näitelause  "Kül se wiks teggia olli."` → `Ve annab s.v. "Wiks" näitelause "Kül se wiks teggia olli."` — T6 did NOT remove the double space, so both T6 and T7 applied to this cell (net old→new from backup: decomposed-ü + double space → composed-ü + single space).
- Acceptance: no string value contains two consecutive spaces (global scan). PASS.

## T8 — multiplicity markers `[xN]` (review B8)

- `mõisnik` . `Göseken-1660-et`: `Moisnick [2x]` → `Moisnick [x2]`
- `lobiseja` . `Göseken-1660-et`: `lobbiseja [3]; löbbiseia` → `lobbiseja [x3]; löbbiseia`
- Acceptance: regex `\[\d+x?\]` matches nothing in the file; only `\[x\d+\]` markers remain (39 cells). PASS.

## T9 — leskmees gender (review B9, first bullet only)

- `leskmees` . `Sugu`: `Ü` → `M`
- `leskmees` . `Comment-1`: `NULL` → `Sugu Ü→M 2026-07: DEF 'widower', paarisvaste lesknaine=N.` — **placed in Comment-1, not Comment-2 as the task literally says; see Discrepancies below.**
- Acceptance: leskmees Sugu == `M`; lesk Sugu == `Ü`; lesknaine Sugu == `N`. PASS.
- teeäärne/äärne and kellamees/kellalööja untouched (D4/D5); the global "-mees" sweep untouched (later agent).
- NOTE for final VERIFY step 3: this flips Tabel-11-adjacent Sugu totals M 267→268, Ü 554→553; WIP text around Tabel 11 must say 268/553.

## T10 — jalamees comment slot (review B11)

- `jalamees` . `Comment-1`: `NULL` → `otsi Ve: kas sks vaste alusel ei leia? Mul algselt oli Ve olemas.`
- `jalamees` . `Comment-2`: `otsi Ve: kas sks vaste alusel ei leia? Mul algselt oli Ve olemas.` → `NULL`
- Acceptance: no record has Comment-2 ≠ NULL with Comment-1 == NULL, nor Comment-3 ≠ NULL with Comment-2 == NULL (global scan). PASS.

## T11 — ÜS feedback file (report only, no master edits) (review C1–C5)

- Created `Katus-DRAFTS/YS-feedback_sem-cat_2026-07-12.md` covering C1 (perepoeg vs peretütar), C2 (bare-`inimene` records incl. the seven "review only" ones), C3 (six morphological-pair mismatches), C4 (isand-family `in_sugulane` + `vana tungus`), C5 (`tüdruk` six-tag question); each item lists lemma, current Sem-Cat, proposed Sem-Cat, one-line DEF + German-gloss evidence.
- Master untouched by this task (per D11 the proposals are NOT folded in until the user reviews the file).

## T12 — standing validator script (review K2, A-block)

- Created `scripts/validate_master.py` (read-only): schema/key-order uniformity, ID uniqueness + prefix, CSC arithmetic (accepts str or int per pending/applied D8), placeholder coherence, whitespace/NBSP/NFC/double-space hygiene, `[xN]` marker format, comment-slot order, Amt-Cat value set, Estonian collation of record order, duplicate tokens within a field.
- Exits 0 on the fixed master: `uv run python scripts/validate_master.py` → `Total: 0 error(s), 0 warning(s)`.

---

# SECTION 2 (T13–T19, D9, D12, D10, D4 sweep)

## T13 — Sem-Cat canonical order (D1 = fixed salience order)

Salience order applied: elukutse > roll > tegija > omadus > sugulane > müt > rahvas > tiitel > inimene (tag mapping: in_elukutse, in_roll, in_tegija, in_omadus, in_sugulane, in_müt, in_rahvas, esitus_tiitel, inimene — the full 9-tag inventory of the file is covered; no unlisted tags were encountered). 141 records reordered. Acceptance verified: identical tag sets now have byte-identical Sem-Cat strings.

- `apostel` Sem-Cat: `in_roll, in_elukutse, in_omadus` -> `in_elukutse, in_roll, in_omadus`
- `arst` Sem-Cat: `esitus_tiitel, in_elukutse` -> `in_elukutse, esitus_tiitel`
- `arutaja` Sem-Cat: `in_omadus, in_roll, in_müt` -> `in_roll, in_omadus, in_müt`
- `aruteleja` Sem-Cat: `in_omadus, in_roll, in_müt` -> `in_roll, in_omadus, in_müt`
- `eestseisja` Sem-Cat: `in_roll, in_elukutse` -> `in_elukutse, in_roll`
- `eksinud naisterahvas` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `ettejooksja` Sem-Cat: `in_tegija, in_roll` -> `in_roll, in_tegija`
- `etteseisja` Sem-Cat: `in_roll, in_elukutse` -> `in_elukutse, in_roll`
- `hoor` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `hädaajaja` Sem-Cat: `inimene, in_tegija` -> `in_tegija, inimene`
- `hädaline` Sem-Cat: `inimene, in_tegija` -> `in_tegija, inimene`
- `ilma usklik` Sem-Cat: `inimene, in_omadus` -> `in_omadus, inimene`
- `jalamees` Sem-Cat: `in_tegija, in_elukutse` -> `in_elukutse, in_tegija`
- `jekk` Sem-Cat: `inimene, in_elukutse, in_omadus, in_roll` -> `in_elukutse, in_roll, in_omadus, inimene`
- `kaeja nõid` Sem-Cat: `in_omadus, in_roll, in_müt` -> `in_roll, in_omadus, in_müt`
- `kaitsja` Sem-Cat: `in_tegija, in_roll` -> `in_roll, in_tegija`
- `kammerer` Sem-Cat: `esitus_tiitel, in_roll` -> `in_roll, esitus_tiitel`
- `kandlelööja` Sem-Cat: `in_tegija, in_elukutse` -> `in_elukutse, in_tegija`
- `kapten` Sem-Cat: `in_elukutse, esitus_tiitel, in_omadus, in_roll` -> `in_elukutse, in_roll, in_omadus, esitus_tiitel`
- `karjane` Sem-Cat: `in_tegija, in_elukutse` -> `in_elukutse, in_tegija`
- `kasakas` Sem-Cat: `in_rahvas, in_roll` -> `in_roll, in_rahvas`
- `keelekoer` Sem-Cat: `in_omadus, in_roll, in_tegija` -> `in_roll, in_tegija, in_omadus`
- `keelekurn` Sem-Cat: `in_omadus, in_roll, in_tegija` -> `in_roll, in_tegija, in_omadus`
- `keelepeksja` Sem-Cat: `in_omadus, in_roll, in_tegija` -> `in_roll, in_tegija, in_omadus`
- `kihnumees` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `kiimane hoor` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `kirikuvanem` Sem-Cat: `in_roll, in_elukutse` -> `in_elukutse, in_roll`
- `kirjatundja` Sem-Cat: `inimene, in_elukutse` -> `in_elukutse, inimene`
- `kodakondne` Sem-Cat: `inimene, in_roll` -> `in_roll, inimene`
- `koduhoidja` Sem-Cat: `in_roll, in_elukutse` -> `in_elukutse, in_roll`
- `kohtuhärra` Sem-Cat: `esitus_tiitel, in_roll` -> `in_roll, esitus_tiitel`
- `krahv` Sem-Cat: `esitus_tiitel, in_omadus` -> `in_omadus, esitus_tiitel`
- `krahviisand` Sem-Cat: `esitus_tiitel, in_omadus` -> `in_omadus, esitus_tiitel`
- `kuningas` Sem-Cat: `esitus_tiitel, in_omadus` -> `in_omadus, esitus_tiitel`
- `kunsikas` Sem-Cat: `in_omadus, in_roll, in_müt` -> `in_roll, in_omadus, in_müt`
- `kuradikunstlik` Sem-Cat: `in_omadus, in_roll, in_müt` -> `in_roll, in_omadus, in_müt`
- `kuulaja` Sem-Cat: `in_tegija, in_roll` -> `in_roll, in_tegija`
- `kõigist varastest vargam` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `kõnemees` Sem-Cat: `inimene, in_roll` -> `in_roll, inimene`
- `kõrgpreester` Sem-Cat: `esitus_tiitel, in_elukutse` -> `in_elukutse, esitus_tiitel`
- `külalähk` Sem-Cat: `in_omadus, in_roll, in_müt` -> `in_roll, in_omadus, in_müt`
- `laevavanem` Sem-Cat: `in_elukutse, esitus_tiitel, in_omadus, in_roll` -> `in_elukutse, in_roll, in_omadus, esitus_tiitel`
- `liigmees` Sem-Cat: `in_roll, in_elukutse` -> `in_elukutse, in_roll`
- `lipukaitsja` Sem-Cat: `in_tegija, in_roll` -> `in_roll, in_tegija`
- `lits` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `lummaja` Sem-Cat: `in_omadus, in_roll, in_müt` -> `in_roll, in_omadus, in_müt`
- `lähk` Sem-Cat: `in_omadus, in_roll, in_müt` -> `in_roll, in_omadus, in_müt`
- `lämmija` Sem-Cat: `in_omadus, in_roll, in_müt` -> `in_roll, in_omadus, in_müt`
- `maahulkuja` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `maarahvas` Sem-Cat: `inimene, in_rahvas` -> `in_rahvas, inimene`
- `madal mees` Sem-Cat: `inimene, in_roll, in_omadus` -> `in_roll, in_omadus, inimene`
- `marssal` Sem-Cat: `esitus_tiitel, in_omadus` -> `in_omadus, esitus_tiitel`
- `mehine mees` Sem-Cat: `inimene, in_roll, in_omadus` -> `in_roll, in_omadus, inimene`
- `meister` Sem-Cat: `in_omadus, in_elukutse, in_roll` -> `in_elukutse, in_roll, in_omadus`
- `murdja` Sem-Cat: `in_omadus, in_elukutse` -> `in_elukutse, in_omadus`
- `mädanud hoor` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `müüja` Sem-Cat: `in_tegija, in_elukutse` -> `in_elukutse, in_tegija`
- `naiseline mees` Sem-Cat: `inimene, in_roll, in_omadus` -> `in_roll, in_omadus, inimene`
- `naisemees` Sem-Cat: `inimene, in_omadus` -> `in_omadus, inimene`
- `naljamees` Sem-Cat: `in_omadus, in_elukutse` -> `in_elukutse, in_omadus`
- `narr` Sem-Cat: `inimene, in_elukutse, in_omadus, in_roll` -> `in_elukutse, in_roll, in_omadus, inimene`
- `nõid` Sem-Cat: `in_omadus, in_roll, in_müt` -> `in_roll, in_omadus, in_müt`
- `ooberst` Sem-Cat: `esitus_tiitel, in_omadus` -> `in_omadus, esitus_tiitel`
- `paader` Sem-Cat: `in_roll, in_elukutse` -> `in_elukutse, in_roll`
- `paimendaja` Sem-Cat: `in_tegija, in_roll` -> `in_roll, in_tegija`
- `peremees` Sem-Cat: `in_roll, in_omadus, in_tegija` -> `in_roll, in_tegija, in_omadus`
- `pordik` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `port` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `priihärra` Sem-Cat: `esitus_tiitel, in_omadus` -> `in_omadus, esitus_tiitel`
- `päästja` Sem-Cat: `in_elukutse, in_tegija, in_roll` -> `in_elukutse, in_roll, in_tegija`
- `püssimees` Sem-Cat: `in_omadus, in_elukutse` -> `in_elukutse, in_omadus`
- `raad` Sem-Cat: `inimene, in_roll` -> `in_roll, inimene`
- `rahamees` Sem-Cat: `inimene, in_omadus` -> `in_omadus, inimene`
- `riisuja` Sem-Cat: `inimene, in_omadus` -> `in_omadus, inimene`
- `rittmeister` Sem-Cat: `esitus_tiitel, in_elukutse` -> `in_elukutse, esitus_tiitel`
- `rootsik` Sem-Cat: `in_rahvas, in_omadus` -> `in_omadus, in_rahvas`
- `rätsepasulane` Sem-Cat: `inimene, in_omadus, in_elukutse` -> `in_elukutse, in_omadus, inimene`
- `röövel` Sem-Cat: `inimene, in_omadus` -> `in_omadus, inimene`
- `rüütel` Sem-Cat: `in_elukutse, in_roll, esitus_tiitel, in_omadus, inimene` -> `in_elukutse, in_roll, in_omadus, esitus_tiitel, inimene`
- `saadik` Sem-Cat: `in_roll, in_elukutse` -> `in_elukutse, in_roll`
- `saks` Sem-Cat: `in_omadus, in_roll, in_rahvas` -> `in_roll, in_omadus, in_rahvas`
- `sakste lämmija` Sem-Cat: `in_omadus, in_roll, in_müt` -> `in_roll, in_omadus, in_müt`
- `sant` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `saunamees` Sem-Cat: `in_roll, in_elukutse` -> `in_elukutse, in_roll`
- `seltsimees` Sem-Cat: `in_omadus, esitus_tiitel, in_roll` -> `in_roll, in_omadus, esitus_tiitel`
- `soolapuhuja` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `suurpreester` Sem-Cat: `esitus_tiitel, in_elukutse` -> `in_elukutse, esitus_tiitel`
- `sõidetud hoor` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `sängikaeja` Sem-Cat: `esitus_tiitel, in_roll` -> `in_roll, esitus_tiitel`
- `söödik` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `söönud mees` Sem-Cat: `inimene, in_roll, in_omadus` -> `in_roll, in_omadus, inimene`
- `taldrikunolp` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `taldrikunolpija` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `talgurahvas` Sem-Cat: `inimene, in_roll` -> `in_roll, inimene`
- `tallatav` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `talutaja` Sem-Cat: `in_tegija, in_roll` -> `in_roll, in_tegija`
- `tapleja` Sem-Cat: `in_omadus, in_tegija` -> `in_tegija, in_omadus`
- `teejuhataja` Sem-Cat: `in_tegija, in_roll` -> `in_roll, in_tegija`
- `tohter` Sem-Cat: `esitus_tiitel, in_elukutse` -> `in_elukutse, esitus_tiitel`
- `tuleroog` Sem-Cat: `in_omadus, in_roll, in_müt` -> `in_roll, in_omadus, in_müt`
- `tunnistaja` Sem-Cat: `in_omadus, in_tegija` -> `in_tegija, in_omadus`
- `tunnistusemees` Sem-Cat: `in_omadus, in_tegija` -> `in_tegija, in_omadus`
- `tunnistusmees` Sem-Cat: `in_omadus, in_tegija` -> `in_tegija, in_omadus`
- `turuhoor` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `tuvimüüja` Sem-Cat: `in_tegija, in_elukutse` -> `in_elukutse, in_tegija`
- `tõmbaja` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `tõmbaja inimene` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `töötegija` Sem-Cat: `inimene, in_roll` -> `in_roll, inimene`
- `tüdruk` Sem-Cat: `inimene, in_omadus, in_sugulane, esitus_tiitel, in_roll, in_elukutse` -> `in_elukutse, in_roll, in_omadus, in_sugulane, esitus_tiitel, inimene`
- `usklik` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `vaekojaisand` Sem-Cat: `in_tegija, in_elukutse` -> `in_elukutse, in_tegija`
- `vaekojamees` Sem-Cat: `in_tegija, in_elukutse` -> `in_elukutse, in_tegija`
- `vaenlane` Sem-Cat: `inimene, in_omadus` -> `in_omadus, inimene`
- `vahemees` Sem-Cat: `in_roll, in_elukutse` -> `in_elukutse, in_roll`
- `vahva kõnemees` Sem-Cat: `inimene, in_roll` -> `in_roll, inimene`
- `vahva mees` Sem-Cat: `inimene, in_omadus, esitus_tiitel` -> `in_omadus, esitus_tiitel, inimene`
- `vahva mees rääkima` Sem-Cat: `inimene, in_roll` -> `in_roll, inimene`
- `vahva sõnamees` Sem-Cat: `inimene, in_roll` -> `in_roll, inimene`
- `vaim` Sem-Cat: `in_omadus, inimene, in_müt, in_roll` -> `in_roll, in_omadus, in_müt, inimene`
- `vana juas` Sem-Cat: `inimene, in_elukutse, in_omadus, in_roll` -> `in_elukutse, in_roll, in_omadus, inimene`
- `vana nõid` Sem-Cat: `in_müt, in_omadus` -> `in_omadus, in_müt`
- `vana tudi` Sem-Cat: `inimene, in_omadus` -> `in_omadus, inimene`
- `varas` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `vastane` Sem-Cat: `in_omadus, inimene, in_roll` -> `in_roll, in_omadus, inimene`
- `veisekarjane` Sem-Cat: `in_tegija, in_elukutse` -> `in_elukutse, in_tegija`
- `veomeister` Sem-Cat: `in_tegija, in_elukutse` -> `in_elukutse, in_tegija`
- `viks tegija` Sem-Cat: `in_omadus, in_elukutse, in_roll` -> `in_elukutse, in_roll, in_omadus`
- `vilajas inimene` Sem-Cat: `inimene, in_omadus` -> `in_omadus, inimene`
- `vilepuhuja` Sem-Cat: `in_tegija, in_roll` -> `in_roll, in_tegija`
- `vindunud inimene` Sem-Cat: `inimene, in_omadus` -> `in_omadus, inimene`
- `voorster` Sem-Cat: `in_roll, in_elukutse` -> `in_elukutse, in_roll`
- `võla` Sem-Cat: `in_omadus, in_roll, in_müt` -> `in_roll, in_omadus, in_müt`
- `võlu` Sem-Cat: `in_müt, in_omadus` -> `in_omadus, in_müt`
- `võõras` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `vägimees` Sem-Cat: `in_müt, in_omadus` -> `in_omadus, in_müt`
- `vürst` Sem-Cat: `esitus_tiitel, in_omadus, in_roll` -> `in_roll, in_omadus, esitus_tiitel`
- `õige varas` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`
- `õitsiline` Sem-Cat: `in_tegija, in_elukutse` -> `in_elukutse, in_tegija`
- `ülem proua` Sem-Cat: `in_omadus, in_roll, inimene` -> `in_roll, in_omadus, inimene`
- `ülempreester` Sem-Cat: `esitus_tiitel, in_elukutse` -> `in_elukutse, esitus_tiitel`
- `ümberhulkuja` Sem-Cat: `in_omadus, in_roll` -> `in_roll, in_omadus`

## T14 — Vestring attested-but-no-German-gloss marker (D2 = NULL)

8 records: `Vestring-17XX-de` "---" → "NULL" (each verified to have a real -et and non-empty -id). Global acceptance verified: no record anywhere has real Vestring -et + non-empty -id + -de "---".

- `külakubjas` Vestring-17XX-de: `---` -> `NULL` (et='Külla Kubjas', 1 id)
- `külalähk` Vestring-17XX-de: `---` -> `NULL` (et='Külla Lähk', 1 id)
- `preester` Vestring-17XX-de: `---` -> `NULL` (et='Preester', 1 id)
- `pruar` Vestring-17XX-de: `---` -> `NULL` (et='Pruar', 1 id)
- `pruul` Vestring-17XX-de: `---` -> `NULL` (et='Pruul, Pruar', 1 id)
- `söönud mees` Vestring-17XX-de: `---` -> `NULL` (et='sönut mees', 1 id)
- `teekäija` Vestring-17XX-de: `---` -> `NULL` (et='Tekäia', 1 id)
- `teomees` Vestring-17XX-de: `---` -> `NULL` (et='Teomees', 1 id)

## T15 — case-only duplicate word-forms removed (D3 = duplicates, keep first-listed)

21 edition `-et` fields deduplicated under casefold(), first occurrence kept, later ones dropped. German `-de` fields NOT touched (exact-match dedup there was already clean). validate_master.py dup-tokens check: 0 remaining.

- `abinaine` Göseken-1660-et: `Abbi Naine; abbi Naine; Abbi nain` -> `Abbi Naine; Abbi nain` (dropped: 'abbi Naine')
- `alevisulane` Göseken-1660-et: `allewe Sullane; Allewe Sullane` -> `allewe Sullane` (dropped: 'Allewe Sullane')
- `eelkäija` Göseken-1660-et: `Eelkeija; eelkeija; eel keija` -> `Eelkeija; eel keija` (dropped: 'eelkeija')
- `ettejooksja` Göseken-1660-et: `Ette johxia; ette johxia` -> `Ette johxia` (dropped: 'ette johxia')
- `hoorasundija` Göseken-1660-et: `Hohra sundia; hohra Sundia` -> `Hohra sundia` (dropped: 'hohra Sundia')
- `kirikuisand` Göseken-1660-et: `Kircko Jssand; kircko Jssand` -> `Kircko Jssand` (dropped: 'kircko Jssand')
- `kõrtsmik` Göseken-1660-et: `Kortzmick; körtzmick; Körtzmick` -> `Kortzmick; körtzmick` (dropped: 'Körtzmick')
- `laevasulane` Göseken-1660-et: `Laiwa Sullane; laiwa Sullane; laiwa sullane` -> `Laiwa Sullane` (dropped: 'laiwa Sullane', 'laiwa sullane')
- `meier` Göseken-1660-et: `Meijer; meijer` -> `Meijer` (dropped: 'meijer')
- `sant` Göseken-1660-et: `Sant; sant` -> `Sant` (dropped: 'sant')
- `sulane` Göseken-1660-et: `Sullane; sullane` -> `Sullane` (dropped: 'sullane')
- `sundija` Göseken-1660-et: `sundia; Sundia` -> `sundia` (dropped: 'Sundia')
- `taldrikunolpija` Göseken-1660-et: `talricko nolpia [x2]; Talricko nolpia [x2]` -> `talricko nolpia [x2]` (dropped: 'Talricko nolpia [x2]')
- `tapja` Stahl-1637-et: `tappija; Tappija` -> `tappija` (dropped: 'Tappija')
- `tark` Göseken-1660-et: `Tarck [x2]; tarck [x2]` -> `Tarck [x2]` (dropped: 'tarck [x2]')
- `teener` Göseken-1660-et: `Teener; teener` -> `Teener` (dropped: 'teener')
- `teotaja` Göseken-1660-et: `Teotaja; teotaja` -> `Teotaja` (dropped: 'teotaja')
- `vabadik` Stahl-1637-et: `wabbadick; Wabbadick` -> `wabbadick` (dropped: 'Wabbadick')
- `vahva mees` Göseken-1660-et: `wahho Mees; wahho mees` -> `wahho Mees` (dropped: 'wahho mees')
- `vahva sõnamees` Göseken-1660-et: `wahho sönnamees; wahho sönna mees; wahho Sönna mees` -> `wahho sönnamees; wahho sönna mees` (dropped: 'wahho Sönna mees')
- `voorkööper` Göseken-1660-et: `Woorköper; woorköper` -> `Woorköper` (dropped: 'woorköper')

## T16 + D4 sweep — gender

teeäärne N→Ü per D4 (äärne verified already Ü, unchanged; kellalööja left as-is, Ü). Independent D4 sweep: every Amt-Master-ID ending in "mees" now has Sugu = M; the sweep changed exactly pandimees and teadvamees (Ü→M) — leskmees was already M (T9), all other -mees lemmas verified already M.

- `teeäärne` Sugu: `N` -> `Ü`
- `teeäärne` Comment-1: `NULL` -> `Sugu N→Ü 2026-07: Vestring/Helle materjal ühine äärne-ga; Hupeli 'öffentliche Hure' feminiinne tähendus märgitud siin.`
- `äärne` Sugu: already `Ü`, no change (verified)
- `pandimees` Sugu: `Ü` -> `M` (D4 sweep)
- `pandimees` Comment-1: `NULL` -> `Sugu Ü→M 2026-07: "-mees" lemmad süstemaatiliselt M (D4).`
- `teadvamees` Sugu: `Ü` -> `M` (D4 sweep)
- `teadvamees` Comment-1: `NULL` -> `Sugu Ü→M 2026-07: "-mees" lemmad süstemaatiliselt M (D4).`

## T17 — sötiku / sötiku istja plural DEFs (D5 = b)

Both records now have consistently PLURAL referents. sötiku's DEFs were already plural and were left verbatim; sötiku istja's DEFs were pluralized. Old → new pairs below for spot-check. Plural-lemma rationale comment added to both records' first free slot.

- `sötiku` DEF_et: already plural, unchanged: `talvekorteris olevad soldatid, talvituvad sõdurid (sks Soldaten in Winterquartier)`
- `sötiku` DEF_en: already plural, unchanged: `soldiers in winter quarters, billeted soldiers`
- `sötiku` Comment-2: `NULL` -> `Mitmuslik lemma (R-klubi 19.06: ainult mitmuses esinev sõna võib jääda mitmusliku lemmaga); DEF mitmuslik 2026-07 (D5b).`
- `sötiku istja` DEF_et:
  - OLD: `talvekorteris istuv soldat, talvituv sõdur (sks Soldaten in Winterquartier)`
  - NEW: `talvekorteris istuvad soldatid, talvituvad sõdurid (sks Soldaten in Winterquartier)`
- `sötiku istja` DEF_en:
  - OLD: `soldier in winter quarters, billeted soldier`
  - NEW: `soldiers in winter quarters, billeted soldiers`
- `sötiku istja` Comment-3: `NULL` -> `Mitmuslik lemma (R-klubi 19.06: ainult mitmuses esinev sõna võib jääda mitmusliku lemmaga); DEF mitmuslik 2026-07 (D5b).`

## T18 — moot cross-source comments deleted (D6 = a)

10 records: the comment `KONTROLLI üle cross-source, kas kõik sees.` (exactly as quoted in review J2) set to NULL, then comment slots re-packed so no higher slot is filled above a NULL one. All other comments left untouched (7 records had a second comment that moved up to Comment-1).

- `eelsõitja` Comment-1: deleted comment `KONTROLLI üle cross-source, kas kõik sees.`
- `eelsõitja` Comment-1: re-packed -> `eelsõitja`
- `eelsõitja` Comment-2: re-packed -> `NULL`
- `eestseisja` Comment-1: deleted comment `KONTROLLI üle cross-source, kas kõik sees.`
- `eestseisja` Comment-1: re-packed -> `eestseisja, eesseisja (ühenda need)`
- `eestseisja` Comment-2: re-packed -> `NULL`
- `isetalumees` Comment-1: deleted comment `KONTROLLI üle cross-source, kas kõik sees.`
- `kaassundija` Comment-1: deleted comment `KONTROLLI üle cross-source, kas kõik sees.`
- `kaassundija` Comment-1: re-packed -> `ORTO: Hu kaaskodanik ja kaasnaine, ENT kasundja. Hu Mitknecht? Really?`
- `kaassundija` Comment-2: re-packed -> `NULL`
- `kindral` Comment-1: deleted comment `KONTROLLI üle cross-source, kas kõik sees.`
- `kindral` Comment-1: re-packed -> `kindral, keneral, generaal, vali sobiv.`
- `kindral` Comment-2: re-packed -> `NULL`
- `meheeksja` Comment-1: deleted comment `KONTROLLI üle cross-source, kas kõik sees.`
- `pordusundija` Comment-1: deleted comment `KONTROLLI üle cross-source, kas kõik sees.`
- `pordusundija` Comment-1: re-packed -> `Hu kus? Varem leidsin.`
- `pordusundija` Comment-2: re-packed -> `NULL`
- `reisija mees` Comment-1: deleted comment `KONTROLLI üle cross-source, kas kõik sees.`
- `silmamoondaja` Comment-1: deleted comment `KONTROLLI üle cross-source, kas kõik sees.`
- `silmamoondaja` Comment-1: re-packed -> `silmamoondaja, silmamodaja, silmamuundaja, silmamuundja`
- `silmamoondaja` Comment-2: re-packed -> `NULL`
- `võltsija` Comment-1: deleted comment `KONTROLLI üle cross-source, kas kõik sees.`
- `võltsija` Comment-1: re-packed -> `Kas võltsija? Valetaja?`
- `võltsija` Comment-2: re-packed -> `NULL`

## T19 — `(sks …)` attested-gloss tails for the 33 H1 DEF_et (D7 approved)

All 33 old → new pairs below for the user's MANDATORY spot-check. Tails built ONLY from each record's own attested `-de` values, leading articles der/die/das/ein/eine stripped, at most two glosses, attested spelling kept verbatim (e.g. Lackey not Lackei, Chor singer, Pabstumb, umbstand, mörder, Verläumder, Landräthe, Heidenthum). METHOD NOTE (deviation from a literal "append"): 28 of the 33 DEFs already carried a *normalized-German* note embedded in a mixed parenthetical (e.g. `(mitmus; sks Lackei)`); purely appending would have duplicated the German with conflicting spellings, so the embedded `; sks …` note was folded into the new attested tail — the label part of the parenthetical (mitmus / deminutiiv / halvustav / source attribution) is preserved unchanged. raad additionally uses the house `; ld` marker for Göseken's attested Latin `senatus`. Acceptance verified: every DEF_et in the file now contains `(sks `.

- `abikaasa` DEF_et:
  - OLD: `abikaasa, abielupartner (Stahlil 'abielunaine')`
  - NEW: `abikaasa, abielupartner (Stahlil 'abielunaine') (sks Ehegatte, Gattin)`
- `abimees` DEF_et:
  - OLD: `abiline, abistaja; ka 'abikaasa, abielumees' (Stahlil Ehemann, Vestringil Ehegatte); Gösekenil ka 'seaduseväänaja, halb advokaat'`
  - NEW: `abiline, abistaja; ka 'abikaasa, abielumees' (Stahlil Ehemann, Vestringil Ehegatte); Gösekenil ka 'seaduseväänaja, halb advokaat' (sks Helfer, Gehülfe)`
- `aganakott` DEF_et:
  - OLD: `kerjus (halvustav sõimusõna; sks Bettler)`
  - NEW: `kerjus (halvustav sõimusõna) (sks Bettler)`
- `haug` DEF_et:
  - OLD: `narr, tobuke (Hupelil; sks Narr, Thor) — enamasti siiski kalanimetus 'haug'`
  - NEW: `narr, tobuke (Hupelil) — enamasti siiski kalanimetus 'haug' (sks Narr, Thor)`
- `isandakene` DEF_et:
  - OLD: `isandike, väike härra (deminutiiv; sks Herrchen)`
  - NEW: `isandike, väike härra (deminutiiv) (sks Herrchen, Herrlein)`
- `isandike` DEF_et:
  - OLD: `isandike, väike härra (deminutiiv; sks Herrchen)`
  - NEW: `isandike, väike härra (deminutiiv) (sks Herrchen, Herrlein)`
- `jalapoisid` DEF_et:
  - OLD: `lakeid, jalapoisid (mitmus; sks Lackei)`
  - NEW: `lakeid, jalapoisid (mitmus) (sks Lackey)`
- `kandijanne` DEF_et:
  - OLD: `kandja, kandev inimene (poeetiline; sks ein Tragender)`
  - NEW: `kandja, kandev inimene (poeetiline) (sks Tragender)`
- `kannikapoiss` DEF_et:
  - OLD: `kerjus, sant (halvustav; sks Bettler, Scheltwort)`
  - NEW: `kerjus, sant (halvustav) (sks Bettler)`
- `keelekandja` DEF_et:
  - OLD: `keelekandja, laimaja, pealekaebaja (Gösekenil 'lipitseja'; sks Verleumder, Ohrenbläser, Fuchsschwänzer)`
  - NEW: `keelekandja, laimaja, pealekaebaja (Gösekenil 'lipitseja') (sks Ohrenbläser, Verläumder)`
- `kekk` DEF_et:
  - OLD: `narr, tobu (Gösekenil ka 'fanaatik'; sks Narr, Stocknarr, Tor, Schwärmer)`
  - NEW: `narr, tobu (Gösekenil ka 'fanaatik') (sks Narr, Stocknarr)`
- `kihnumees` DEF_et:
  - OLD: `varas (mitte 'kihnlane'; sks Dieb)`
  - NEW: `varas (mitte 'kihnlane') (sks Dieb)`
- `kirikumees` DEF_et:
  - OLD: `kirikuteenija: kellamees, koerapeksja (mitte 'vaimulik'; sks Kirchenkerl, Glockenläuter, Hundeschläger)`
  - NEW: `kirikuteenija: kellamees, koerapeksja (mitte 'vaimulik') (sks Kirchenkerl, Glockenläuter)`
- `kivinek` DEF_et:
  - OLD: `kiviraidur (halvustav nimetus; sks Steinhauer, Scheltwort)`
  - NEW: `kiviraidur (halvustav nimetus) (sks Steinhauer)`
- `kohtuvanem` DEF_et:
  - OLD: `kohtuvanemad, ülemus, võimukandjad (mitmus; sks Obrigkeit)`
  - NEW: `kohtuvanemad, ülemus, võimukandjad (mitmus) (sks Obrigkeit)`
- `koorilaulja` DEF_et:
  - OLD: `koorilauljad (mitmus; sks Chorsänger)`
  - NEW: `koorilauljad (mitmus) (sks Chor singer)`
- `leivavanemad` DEF_et:
  - OLD: `leivavanemad, isandad, tööandjad (keda teenitakse; mitmus; sks Herrschaft, Brodeltern)`
  - NEW: `leivavanemad, isandad, tööandjad (keda teenitakse; mitmus) (sks Brodeltern, Herrschaft)`
- `liigrahvas` DEF_et:
  - OLD: `kohalviibijad, juuresolijad, tunnistajad (mitmus; sks Umstand des Volkes)`
  - NEW: `kohalviibijad, juuresolijad, tunnistajad (mitmus) (sks umbstand des Volkes)`
- `lirva` DEF_et:
  - OLD: `lobiseja, latatara (ajalooline täh; sks Plaudertasche)`
  - NEW: `lobiseja, latatara (ajalooline täh) (sks Plaudertasche)`
- `maamuusikas` DEF_et:
  - OLD: `mõrtsukas, tapja (mitte 'muusik'; sks Mörder)`
  - NEW: `mõrtsukas, tapja (mitte 'muusik') (sks mörder)`
- `maarahvas` DEF_et:
  - OLD: `maarahvas, talurahvas, eesti maainimesed (mitmus; sks die Bauern)`
  - NEW: `maarahvas, talurahvas, eesti maainimesed (mitmus) (sks Bauern)`
- `maavanemad` DEF_et:
  - OLD: `maavanemad, maanõunikud, ülemus (mitmus; sks Landräte, Obrigkeit)`
  - NEW: `maavanemad, maanõunikud, ülemus (mitmus) (sks Landräthe, Obrigkeit)`
- `marterer` DEF_et:
  - OLD: `märter, usukannataja`
  - NEW: `märter, usukannataja (sks Martyrer)`
- `mungarahvas` DEF_et:
  - OLD: `mungad, mungarahvas, paavstiusulised (mitmus; sks Papsttum)`
  - NEW: `mungad, mungarahvas, paavstiusulised (mitmus) (sks Pabstumb)`
- `muusikas` DEF_et:
  - OLD: `mõrtsukas, tapja; maanteeröövel (mitte 'muusik'; sks Mörder, Buschklepper)`
  - NEW: `mõrtsukas, tapja; maanteeröövel (mitte 'muusik') (sks mörder, Buschklepper)`
- `mõisavanemad` DEF_et:
  - OLD: `mõisarahvas, mõisaisandad, mõisavalitsejad (mitmus; sks Hofsherrschaft)`
  - NEW: `mõisarahvas, mõisaisandad, mõisavalitsejad (mitmus) (sks Hofsherrschaft)`
- `neitsike` DEF_et:
  - OLD: `neitsike, neiuke (deminutiiv; sks Jungferchen)`
  - NEW: `neitsike, neiuke (deminutiiv) (sks Jungferchen)`
- `nobedad poisid` DEF_et:
  - OLD: `jooksupoisid, lakeid (mitmus; sks Lackei)`
  - NEW: `jooksupoisid, lakeid (mitmus) (sks Lackey)`
- `nõumehed` DEF_et:
  - OLD: `vandenõulased, salaliitlased, õhutajad (mitmus; sks die in einem Complot stehen, Urheber)`
  - NEW: `vandenõulased, salaliitlased, õhutajad (mitmus) (sks die in einem Complot stehen, Urheber)`
- `paganarahvas` DEF_et:
  - OLD: `paganarahvas, paganad (mitmus; sks die Heiden, Heidentum)`
  - NEW: `paganarahvas, paganad (mitmus) (sks Heiden, Heidenthum)`
- `pandimees` DEF_et:
  - OLD: `üürileandja`
  - NEW: `üürileandja (sks Pachtman, miehtman)`
- `pandisaks` DEF_et:
  - OLD: `üürileandja`
  - NEW: `üürileandja (sks Miehtman, der ligende Gründe vermietet)`
- `raad` DEF_et:
  - OLD: `raad, linnavolikogu; ka raehärra (asutus v. ametiisik; sks Rat, Stadtrat, senatus)`
  - NEW: `raad, linnavolikogu; ka raehärra (asutus v. ametiisik) (sks Rath, Stadtrath; ld senatus)`

## D9 — old-orthography lemma renames

6 Amt-Master-ID renames (no target collisions; attested forms in edition fields untouched); file re-sorted with the canonical Estonian collation key imported from scripts/validate_master.py. The keep-as-is D9 lemmas (tolvan, sötiku, sötiku istja, kandijanne, pruar, vana juas, kõigist varastest vargam) verified present under exactly those shapes. NOTE: downstream artifacts embedding the old lemmas (LexLex viewer, Annex-1 printout/docx, APTSK/linkage lists, Tabelid sõnakuju-id lists) need regeneration — not regenerated here.

- Amt-Master-ID: `umbleja` -> `õmbleja`
- Amt-Master-ID: `opja poiss` -> `õppija poiss`
- Amt-Master-ID: `heikaja` -> `hõikaja`
- Amt-Master-ID: `meeste piddäja` -> `meestepidaja`
- Amt-Master-ID: `saia kitsäja` -> `saiaküpsetaja`
- Amt-Master-ID: `üte maa mees` -> `ühe maa mees`
- `tolvan`: kept as-is (verified present)
- `sötiku`: kept as-is (verified present)
- `sötiku istja`: kept as-is (verified present)
- `kandijanne`: kept as-is (verified present)
- `pruar`: kept as-is (verified present)
- `vana juas`: kept as-is (verified present)
- `kõigist varastest vargam`: kept as-is (verified present)

## D12 — Teema TEENISTUS for the whole vaht family

All 7 vaht-family records now TEENISTUS (user chose all 7). mõisavaht verified already TEENISTUS. NOTE: Tabelid 8/9/12/13 Teema counts shift — TEENISTUS +6, MUU −4 (uksevaht, vaht, vahtja, öövaht), HALDUS_VÕIM −2 (vahimees, vahisoldat); regeneration required before reuse.

- `mõisavaht` Teema: already `TEENISTUS` (verified)
- `uksevaht` Teema: `MUU` -> `TEENISTUS`
- `vahimees` Teema: `HALDUS_VÕIM` -> `TEENISTUS`
- `vahisoldat` Teema: `HALDUS_VÕIM` -> `TEENISTUS`
- `vaht` Teema: `MUU` -> `TEENISTUS`
- `vahtja` Teema: `MUU` -> `TEENISTUS`
- `öövaht` Teema: `MUU` -> `TEENISTUS`

## D10 — variant-freeze decision written into the review file

Appended a dated decision note to the F-section of `Katus-DRAFTS/DQ-review_AMT-Master_2026-07-12.md` (after F5, before the G-section divider); no existing review text altered: variants frozen as-is, orthographic variants count as different lemmas even with shared linked attestations (variants can occur inside the linked source's entry body), no merges before the kappa numbers are frozen.

---

# T20 — `Cross-source count` string → JSON integer (D8 = yes; run LAST)

## Field conversion

- All **882** records: `"Cross-source count"` converted from JSON string to JSON integer (`"3"` → `3`; every value was a plain digit string, asserted before conversion). Value distribution unchanged: 1×446, 2×138, 3×159, 4×71, 5×39, 6×29. No CSC value equals 0 (relevant for JS truthiness in the viewer, see below).

## Consumer scripts patched (2)

- `scripts/recompute_crosssource.py` — the only writer of the field; it wrote the recomputed count back as a string. Patched line 34:
  - OLD: `c["Cross-source count"] = str(n)`
  - NEW: `c["Cross-source count"] = n          # JSON integer per D8 (2026-07-13)`
  - (The change-detection compare `old = str(c.get("Cross-source count", "NULL"))` / `old != str(n)` already handles both types and was left as-is.)
- `scripts/report_master_review.py` — read the field through `cellstr()`, which returns the literal string `"NULL"` for any non-str value, so an integer CSC would have flagged all 882 records as mismatches in report section 4.3. Patched lines 189–191:
  - OLD: `raw = cellstr(c, "Cross-source count")` … `stored = int(raw)`
  - NEW: `raw = c.get("Cross-source count", "NULL")   # str or int (D8 2026-07-13)` … `stored = int(str(raw))`

## Consumer scripts inspected and confirmed int-safe (no change needed)

- `scripts/validate_master.py` — accepts str or int by design (`int(str(raw))`); left untouched as instructed.
- `scripts/verify_all.py` — does not read the field.
- `scripts/build_annex1_printout.py` (l.106) and `scripts/build_annex1_docx.py` (l.112) — both coerce via `str(entry["Cross-source count"]).strip()`.
- `scripts/build_viewer.py` — CSC is display-only in the embedded JS (`esc()` does `String(s)`; `fld()`/`v2cell()` truthiness `!val` would only misfire on CSC `0`, which does not occur in the data; search haystacks never include CSC).
- `scripts/ptk-4-analysis/fold_and_analyze.py` — `int(r.get("Cross-source count") or 0)` and f-string printing, both type-agnostic.
- `scripts/ptk-4-analysis/semantic_change_analysis.py` — passes the value through to a dict/f-string only.
- `scripts/tabelid-3-13/tabel_04.py` and `tabel_09.py` — `int(e["Cross-source count"])`, idempotent on int.
- `scripts/convert_amt_master.py` — legacy xlsx→JSON converter that writes a *different* file (`analyzed-tables/AMT-Master.json`) from the 2026-06 workbook; not a consumer of the annotated master; deliberately left untouched.

## Frozen Tabelid outputs — evidence of non-interference

- All 11 tabelid scripts (`tabel_03.py` … `tabel_13.py`) print to **stdout only** — verified by grep, none opens a file for writing — so running them cannot overwrite anything. All 11 were executed against the converted master: **all exit 0 with int CSC** (the "before" run for the impact table below used a copy of the BACKUP in the session scratchpad, never the repo).
- The frozen deliverables `Katus-DRAFTS/Katus-tervik-WIP/Tabelid-3-13_uuendatud-20260712.md` / `.docx` (and every other file in `Katus-tervik-WIP/`) still carry their 2026-07-12 mtimes; `git status --porcelain` is byte-identical before vs after the run; the six edition JSONs still carry their 2026-07-07 mtimes; the BACKUP file is unmodified.
- `uv run python scripts/validate_master.py` → exit 0 (`Total: 0 error(s), 0 warning(s)`); `uv run python scripts/verify_all.py` → `ALL CHECKS PASSED` (882 master records, all 7 files schema-clean, 46 253 globally unique ids).

---

# Judgement calls / deviations

- **T9 comment placed in Comment-1, not Comment-2 (task-internal conflict).** T9 says add the note to `Comment-2` "(currently NULL)", but leskmees `Comment-1` was also NULL — following T9 literally would make leskmees the sole violator of T10's global acceptance ("no Comment-2 ≠ NULL while Comment-1 == NULL"), which the final VERIFY re-checks. The exact T9 text went into `Comment-1`; swap the two slots if Comment-2 placement was intentional.
- **T19: embedded `; sks …` notes folded into the new attested tails.** 28 of the 33 DEFs already carried a normalized-German note inside a mixed parenthetical (e.g. `(mitmus; sks Lackei)`); a literal append would have produced two conflicting German glosses per DEF. The embedded note was replaced by the tail built from attested `-de` values (attested spelling kept verbatim: Lackey, Chor singer, Pabstumb, Landräthe, …); the label part (mitmus / deminutiiv / halvustav / attribution) is preserved unchanged. `raad` additionally uses the house `; ld` marker for Göseken's Latin `senatus`.
- **T17 (D5b): `sötiku` DEFs were already plural** and were left verbatim; only `sötiku istja` DEF_et/DEF_en were pluralized. Both records got the plural-lemma rationale comment.
- **Already-correct values encountered during D-sweeps** (no-ops, recorded per G0.7): `äärne` Sugu already Ü; `mõisavaht` Teema already TEENISTUS; `leskmees` already M when the D4 "-mees" sweep ran (set by T9); the D4 sweep therefore changed exactly `pandimees` and `teadvamees`.
- **T18 comment deletions required slot re-packing** in 7 of the 10 records (a second comment moved up to Comment-1) to keep the comment-slot-order invariant.
- **Part-1's provisional Tabel-11 note is superseded.** T9's log predicted "M 267→268, Ü 554→553"; after the full run (T9 + D4 sweep + teeäärne N→Ü) the actual gender totals are M 267→270, Ü 554→552, N 61→60 — and Tabel 11 itself counts *female* lemmas per source, so the numbers that actually change there are the ones in the impact section below.
- **T20:** `convert_amt_master.py` left unpatched (legacy converter of a different file, see above); `validate_master.py` left as-is per instruction.

---

# VERIFY results (final battery, post-T20)

1. `uv run python scripts/validate_master.py` → **exit 0**; all 14 checks `[OK]`, `Total: 0 error(s), 0 warning(s)` (schema, master-id-unique, id-unique-prefix, csc, placeholder-id, hygiene-strip, hygiene-nbsp, hygiene-nfc, hygiene-dspace, multiplicity, comment-order, amt-cat, collation, dup-tokens).
2. `uv run python scripts/verify_all.py` → **ALL CHECKS PASSED** (master 882/882 unique; editions 3 830 / 1 859 / 14 097 / 6 953 / 4 900 / 13 732 entries, canonical schema; 46 253 ids globally unique).
3. Independent battery (scratchpad `verify_battery.py`):
   - **882 records**, uniform **30-key schema**, key order byte-identical to the backup's.
   - **Edition id resolution:** 2 104 `<Source>-id` references checked against the six edition JSONs (read-only) — **0 unresolved**.
   - **CSC:** all 882 values are JSON integers and match the recomputed per-record attestation count.
   - **Estonian collation** of record order intact (space/hyphen before letters).
   - **Hygiene:** 0 unstripped values, 0 U+00A0, 0 non-NFC values, 0 double spaces.
   - **Multiplicity markers:** 0 `[N]`/`[Nx]`-style markers (only `[xN]` remains).
   - **Comment-slot order:** 0 violations.

## Categorical-field impact (BACKUP-2026-07-13 → current master)

| Field | Records changed | Detail | Totals old → new |
|---|---|---|---|
| Amt-Cat | **0** | — | unchanged: K1 369, K2 159, K3 336, K1+K2 4, K1+K3 10, K2+K3 4 |
| Sem-Cat | 141 | **0 set changes** (expected ZERO — confirmed), 141 pure D1 reorders | tag multisets identical |
| Teema | 6 | uksevaht, vaht, vahtja, öövaht MUU→TEENISTUS; vahimees, vahisoldat HALDUS_VÕIM→TEENISTUS | TEENISTUS 56→62, MUU 230→226, HALDUS_VÕIM 142→140; other 9 themes unchanged |
| Sugu | 4 | leskmees (T9), pandimees, teadvamees (D4 sweep) Ü→M; teeäärne N→Ü (D4) | M 267→270, Ü 554→552, N 61→60 |
| Amt-Master-ID | 8 | maavalitseja␣→maavalitseja, tolwan→tolvan, heikaja→hõikaja, meeste piddäja→meestepidaja, opja poiss→õppija poiss, saia kitsäja→saiaküpsetaja, umbleja→õmbleja, üte maa mees→ühe maa mees | — |
| Word-form tokens (-et/-de) | 23 fields | **24 duplicate tokens removed** = 2 exact (T3) + 22 case-only (T15/D3); a further 8 tokens rewritten in place (NBSP/double-space/NFC/`[xN]` fixes), not removed | raw `;`-token count 4 014 → 3 990 |
| Cross-source count | 882 | type only, `"N"` → N | value distribution identical |

## WIP-text / Tabelid numbers that MUST change (all six changed tables verified by before/after script runs; Tabelid 3–7 outputs byte-identical)

- **Tabel 8** (Teema shares): MUU 230 → **226** (26,1 → **25,6 %**); HALDUS_VÕIM 142 → **140** (16,1 → **15,9 %**); TEENISTUS 56 → **62** (6,3 → **7,0 %**).
- **Tabel 9** (KAS per Teema): MUU 230 / 2,33 → **226 / 2,34**; HALDUS_VÕIM 142 / 2,07 → **140 / 2,07**; TEENISTUS 56 / 1,95 → **62 / 1,94** (row also moves one position down in the KAS-sorted order).
- **Tabel 10** (NÕID forms): Göseken row loses the case duplicate — `… lausia / lausija, … Tarck / tarck, …` → `… Tarck …` (form count in that cell 10 stays, listed shapes change).
- **Tabel 11** (female lemmas per source; teeäärne N→Ü): Vestring 26 → **25** (7,2 → **6,9 %**); Helle 22 → **21** (7,7 → **7,3 %**); Hupel 37 → **36** (6,1 → **6,0 %**).
- **Tabel 12** (female lemmas per Teema): MORAAL_HÄLVE 19 → **18**; Kokku 61 → **60**.
- **Tabel 13** (Amt-Cat × Teema): HALDUS_VÕIM K1 68 → **66**, row total 142 → **140**; TEENISTUS K1 28 → **34**, row total 56 → **62**; MUU K1 18 → **14**, row total 230 → **226**.
- **WIP text near Tabel 11 / gender discussion:** any quoted gender totals must become **M 270, Ü 552, N 60** (supersedes the interim "268/553" note from the Section-1 log); any "61 naissoole viitavat lemmat" → **60**.
- **Any "massiivis sõnakujusid" figure counted from `;`-tokens** must be recomputed under the new D3 rule (case-normalized): raw token total is now 3 990 (was 4 014). Tabel 3's linked-article counts are NOT affected (verified identical).

---

# Downstream artifacts requiring regeneration (none regenerated in this run)

- **Annex-1 printout / docx** (`build_annex1_printout.py` / `build_annex1_docx.py`) — embeds the old spaced key `maavalitseja␣`, old lemmas (tolwan, umbleja, …), pre-fix cell values, duplicated forms, and string CSC.
- **LexLex viewer** (`build_viewer.py` → `LexLex.html`) — same embedded values; rebuild also picks up int CSC.
- **APTSK T1 / linkage lists** (`aptsk_master_diff.py` output) — embed the spaced key and old lemma shapes.
- **Tabelid 3–13** — now REQUIRED (D-decisions changed categorical fields): Tabelid **8, 9, 12, 13** shift by the Teema/Sugu deltas quantified above; Tabel **10** by the case-dup form list; Tabel **11** by teeäärne N→Ü. Tabelid 3–7 verified unchanged. The frozen 15.07 deliverable `Katus-tervik-WIP/Tabelid-3-13_uuendatud-20260712.md/.docx` was NOT touched — regenerate as a new dated file when ready.
- **WIP text numbers** around Tabel 11 and the Teema/gender paragraphs, per the list above.

# Non-master deliverables created this run

- `Katus-DRAFTS/YS-feedback_sem-cat_2026-07-12.md` (T11 — ÜS Sem-Cat feedback, awaiting the user's manual review per D11).
- `scripts/validate_master.py` (T12 — standing read-only validator; exit 0 on the current master).

---

## Addendum 2026-07-13 (later same day) — ÜS feedback fold-in (C1–C5 per D11) + manual deletions

### Manual record deletions (by MJ, confirmed intentional)

12 records deleted from the master between the fix run and this addendum; **canonical record count is now 870** (was 882). Validator passes clean on 870.

`madal mees`, `mehine mees`, `mooramees`, `selge vana muld`, `söönud mees`, `vana kämaras`, `vana muld`, `vana mäda`, `vana tudi`, `vana tungus`, `vilajas inimene`, `vindunud inimene`

(Note: `söönud mees` had received the T14 Vestring-de NULL marker and `vana tungus` was a C4 item — both now moot.)

### Sem-Cat changes per MJ decisions in `YS-feedback_sem-cat_2026-07-12.md` (23 records)

| record | old Sem-Cat | new Sem-Cat | decision |
|---|---|---|---|
| perepoeg | inimene | in_sugulane | C1/C2a |
| mõisnik | inimene | in_roll | C2a |
| igavene sulane | inimene | in_roll | C2a |
| pärismees | inimene | in_roll | C2a |
| poistepidaja | inimene | in_omadus | C2a |
| poisteteotaja | inimene | in_omadus | C2a |
| mõisavanemad | inimene | in_roll | C2a + Discrepancy 2 |
| kohtuvanem | in_elukutse, in_roll | in_roll | Discrepancy 2 |
| maavanemad | in_elukutse | in_roll | Discrepancy 2 |
| abimees | in_roll | in_roll, in_sugulane | C3 |
| abinaine | in_tegija, in_sugulane | in_roll, in_sugulane | C3 |
| orjapoiss | in_elukutse, in_roll, in_omadus | in_elukutse, in_roll | C3 |
| orjatüdruk | in_roll | in_elukutse, in_roll | C3 |
| peremees | in_roll, in_tegija, in_omadus | in_roll | C3 (confirmed: exactly in_roll) |
| perenaine | in_elukutse, in_roll | in_roll | C3 (confirmed: exactly in_roll) |
| vabatmees | in_omadus | in_roll | C3 (vabatnaine already in_roll — unchanged) |
| vahva mees | in_omadus, esitus_tiitel, inimene | in_omadus | C3 |
| vahva naine | in_roll | in_omadus | C3 |
| tüdruk | in_elukutse, in_roll, in_omadus, in_sugulane, esitus_tiitel, inimene | in_elukutse, in_roll | C5 |
| isand | in_roll, in_omadus, in_sugulane | in_roll, in_omadus | C4 |
| isandakene | in_roll, in_omadus, in_sugulane | in_roll, in_omadus | C4 |
| isandike | in_roll, in_omadus, in_sugulane | in_roll, in_omadus | C4 |
| käraisand | in_roll, in_omadus, in_sugulane | in_roll, in_omadus | C4 |

(23 records changed in total.) No change (per decisions): kuningaemand/kuningaproua (both stay `in_roll, inimene`), all C2b records, vanakõu (C4 control case).

All new values follow the D1b salience order. The master now deliberately diverges from ÜS until they respond (per D11).

### Downstream impact

Record-count-sensitive artifacts (Tabelid 3–13, Annex-1, LexLex, APTSK/linkage lists, WIP totals incl. Sugu/Teema counts from the earlier run) must be regenerated against 870 records; Tabel 6 (Sem-Cat) additionally shifts by the 23 tag changes.

---

## Addendum 2 (2026-07-13, revision round) — D1/D2/J1 decisions applied

### D1 — junkur (mixed-sense records need Ü)

| record | field | old | new |
|---|---|---|---|
| junkur | Sugu | M | Ü |

### D2 — Sugu policy sweep (policy: M/N only with a clear gender marker in ET word, DE gloss, or unambiguous sense; Ü where both genders present or sense allows either)

All 323 M/N records reviewed; full report with evidence: `Katus-DRAFTS/Sugu-policy-review_2026-07-13.md`.

| record | field | old | new | evidence |
|---|---|---|---|---|
| lapse kaela murdja | Sugu | N | Ü | Göseken masc *kinder Mörder* vs Vestring/Helle/Hupel fem *Kinder-Mörderin(n)* |
| lapselootaja | Sugu | N | Ü | Vestring masc *Der die Frucht abtreibt* vs Helle/Hupel fem *die die Frucht abtreibt* |
| leivavanemad | Sugu | M | Ü | Hupel *Brodeltern* names both genders; otherwise neutral *Herrschaft* |

`meier` verified and kept N (only feminine glosses: *KebsWeib; beyschläfferin*). 7 candidates flagged for MJ decision (not changed): ori, tunnistaja, valetunnistaja, mõisavanemad, kohtuvanem, sillavanemad, õppija.

**Sugu distribution after this round: Ü 550 / M 262 / N 58** (870 records).

### J1 — comment extraction (MJ: the JSON must be clean)

All 503 non-NULL Comment-1/2/3 values (461 records) extracted verbatim into `Katus-DRAFTS/Comments-in-JSON_2026-07-13.md`; every Comment slot in the master set to `"NULL"`. This includes the comments added earlier in this run (T9 leskmees, T16 teeäärne, T17 sötiku pair, D4 sweep notes) — they are preserved in the extraction file.

### Closed without master edits

- E3: MJ noted the metsmuusikas/maamuusikas gloss fix in the kappa.
- F4: justification recorded — ET lemmas can be MWUs with spaces; no harmonizing.
- F5: kappa sentence decided — one dictionary entry can contain variants, hence several master lemmas can share one entry.
- K3: SCHEMA.md legend drafted, awaiting MJ approval.

Validator after all edits: exit 0 (14/14 checks, 870 records).

---

## Addendum 3 (2026-07-13) — Sugu candidate decisions + SCHEMA.md legend

MJ decided the 7 flagged Sugu candidates (see `Sugu-policy-review_2026-07-13.md`):

| record | field | old | new |
|---|---|---|---|
| ori | Sugu | M | Ü |
| tunnistaja | Sugu | M | Ü |
| valetunnistaja | Sugu | M | Ü |
| mõisavanemad | Sugu | M | Ü |
| õppija | Sugu | M | Ü |

`kohtuvanem` and `sillavanemad` confirmed M (historically F not plausible). **Final Sugu distribution: Ü 555 / M 257 / N 58.** Validator exit 0.

K3 closed: the master value-marker legend was appended to `scripts/SCHEMA.md` as approved.

---

## Addendum 4 (2026-07-13) — downstream artifacts regenerated (870-record master)

- **Tabelid 3–13**: `Katus-tervik-WIP/Tabelid-3-13_uuendatud-20260713.md` + `.docx` (rev 1; python-docx, same method as rev 4). No script patches needed. Deltas vs 2026-07-12 rev 4: sõnakujud 2104→2077; Tabel 4 CSC 3/2/1 = 155/132/444; Tabel 5 K3 336→324 (all 12 deletions were K3); Tabel 6 tags 1237→1201; Tabel 8 MUU 214, HALDUS_VÕIM 140, TEENISTUS 62; Tabel 9 KÄSITÖÖ/TEENISTUS row swap; Tabel 10 unchanged counts; Tabel 11 N-totals per source down; Tabel 12 kokku 61→58, MORAAL_HÄLVE 16; Tabel 13 rebalanced, grand total 870.
- **Annex-1**: TEX + PDF (tectonic, 37 pp) + DOCX regenerated in `Katus-tervik-WIP/`, exactly 870 entries, collation verified, no scripts patched.
- **LexLex v9**: `build_viewer.py` VERSION → v9 (only change); `global-view/LexLex.html`, `LexLex_v9.html`, `LexLex-doc.html` rebuilt; 870 records embedded; no literal NULL glosses; int CSC fine.
- **APTSK**: `aptsk_master_diff.py` repathed from stale `AMT-Master_annotated_20260707.json` to canonical master (2-line patch); report + CSV regenerated. T1 exact 344→346 (õmbleja, meestepidaja now match); T0 174; tolvan still T0.
- **WIP numbers**: `Katus-DRAFTS/WIP-numbers_2026-07-13.md` — canonical numbers + 14 grouped replacement locations for the kappa WIP; ÜK screen re-run from archived lemmafreq: 120 absent / 321 rare / 344 current (85 MWU + 785 single-word). 5 items flagged not recomputable (VPTS 39%, P2-era 678/542/378, original 806/129/333/102 derivation, "21 andmekirjes" denominator, literature article counts).
- Data-side note: only 7 Vestring `NULL`-gloss cells remain (T14 had set 8; `söönud mees` was deleted by MJ).

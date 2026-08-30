# Execution report — folding ÜS semantic types into AMT-Master

*Created: 2026-07-12 13-30-42 · Author: Madis Jürviste · Co-Authored-By: Claude Fable 5*

- **Date:** 2026-07-12
- **Actors:** Madis Jürviste (decisions, manual review), Claude Fable 5 (implementation, sub-agent orchestration)
- **Target file:** `Sem-Cat-YS-to-AMT/AMT-Master_annotated_with_YS-semcat.json`
- **Script:** `scripts/fold_ys_semcat_into_master.py` (working-repo path; published in this repository as `core/fold_ys_semcat_into_master.py`)
- **Pre-run backup:** `Sem-Cat-YS-to-AMT/AMT-Master_annotated_with_YS-semcat_BCKP-20260712-132807.json`
- **Result:** all **882** entries now carry a person-denoting ÜS semantic type in `Sem-Cat`; **no entry lost, no other field touched** (independently verified, §8).

---

## 1. Objective

Replace every `Sem-Cat` value in AMT-Master with semantic categories from the
EKI ühendsõnastik (ÜS / Ekilex, Sõnaveeb), folding in all decisions embodied in
`semcat-diff-report.html`, under these constraints set by MJ:

1. Include **only person-denoting** ÜS semantic types — exclude tags such as
   `toit`, `tegevus_kõnetegu`, `esitus_keel_suhtlus`, `taim`, `loom_kala`, …
2. Get **as many ÜS semcats as possible** into the JSON.
3. **All** current `Sem-Cat` labels are **replaced** by ÜS-derived ones.
4. **No entry may be lost**; **every** entry must end up with an ÜS semcat;
   **no other field** in the master JSON may change.
5. Special case: **jooks** takes the category ÜS gives to **kuller**.

## 2. Input data

| File | Role | Notes |
|---|---|---|
| `Sem-Cat-YS-to-AMT/AMT-Master_annotated_with_YS-semcat.json` | target (read + overwritten in place) | 882 entries under key `"AMT-Master"`; at start of this run it contained **no** `YS-lemma`/`YS-Sem-Cat` fields (stripped 2026-07-07) |
| `Lemmas/AMT-YS-match-report_20260706_read.csv` | reviewed AMT-lemma → ÜS-word mapping | 886 rows (`amt_lemma`, `tier`, `ys_candidates`, `; `-separated candidates) |
| `JSON-YS/_SELECT_..._202607061931.json` | raw ÜS export (DBeaver, one row per word/lexeme/meaning) | 1 067 rows, 603 distinct words, `semantic_types` already `, `-aggregated per meaning |
| `Sem-Cat-YS-to-AMT/semcat-diff-report.html` | collation report driving the decisions | generated 2026-07-06 from the 886-entry annotated file (now `_OLD_BCKP`) |
| `Sem-Cat-YS-to-AMT/semcat-72-review.tsv` | **new** — manually verified decisions for the 72 problem entries | created this run; reviewed and corrected by MJ 2026-07-12 |

### 2.1 Master-file drift since the diff report

The diff report was generated from the 886-entry version (2026-07-06,
now `AMT-Master_annotated_with_YS-semcat_OLD_BCKP.json`). The current target
has 882 entries. Reconciliation:

- **Removed since the report** (4): `at`, `att`, `essitaja`, `ohver`
  (known pending-promotion items — deliberately absent from the current master;
  no action taken).
- **Renamed** (1): `alw rahvas` → `alv rahvas`. The mapping CSV still lists
  the old spelling; the fold-in script resolves the new spelling via an
  explicit rename table (`alv rahvas` → mapping row `alw rahvas` → ÜS `pööbel`
  → `in_omadus`). No entry was lost to the rename.
- **Nothing else** differs in entry inventory: 882 = 886 − 5 removed/renamed + 1 renamed.

### 2.2 Diff-report group counts (basis of "fold in all decisions")

| Siglum | Group | n (of 886) | Decision folded in |
|---|---|---|---|
| = | Full match | 282 | ÜS tags adopted (agree with old Sem-Cat anyway) |
| + | ÜS adds categories | 256 | ÜS tags adopted, including the additions |
| ≠ | No overlap | 278 | ÜS person tags **replace** the old annotation; where ÜS had only non-person tags → decisions TSV |
| ~ | Partial overlap | 24 | ÜS person tags adopted |
| ∅ | No ÜS data | 46 | decisions TSV (manual, analogy-based) |

## 3. Person-tag whitelist (decision MJ, 2026-07-12)

Included (10 tags):

```
in_elukutse, in_roll, in_omadus, in_tegija, inimene,
in_sugulane, in_rahvas, in_rahvas_keel, in_müt, esitus_tiitel
```

`esitus_tiitel` was explicitly ruled **in** (borderline case; it is the only
ÜS tag of `keisriproua`, and a title of address does describe a person in this
lexicon). `in_müt` (mythological being) ruled in.

Excluded — the full inventory of non-person ÜS tags observed on mapped
candidates, with occurrence counts across entries (they were silently dropped
wherever they co-occurred with person tags):

```
 35 omadus_kval        26 omadus             25 ese_instru
 20 esitus_keel_suhtlus 19 omadus_psühh      18 koht_asutus
 13 loom               12 abstr/konkr        11 loom_omadus
  8 loom_putukas        7 nähtus              6 materjal/aine
  6 omadus_füüs         6 objekt              5 konkr
  5 taim                5 ese_semio           5 nähtus_füüs
  4 abstr               4 ese                 4 organism
  3 toit                3 omadus_aeg          3 tegevus
  3 tegevus_kõnetegu    3 esitus_kujutis      3 omadus_koht
  2 objekt_loodus       2 esitus              2 tegevus_tegu
  2 seisund             2 loom_liik           2 koht_geogr
  2 aeg                 2 seisund_haigus      2 koht_ala
  2 nähtus_psühh        1 loom_kala           1 esitus_arv
  1 vald                1 sündmus             1 ADV_aste
  1 koht_loodus         1 koht_hoone          1 ADV_seisund
  1 ese_raha            1 toit_jook           1 taim_omadus
  1 koht                1 ADV_modaalsus       1 toit_maitseaine
  1 esitus_keel         1 konkr_omadus        1 omadus_vanus
```

## 4. Method

### 4.1 Pipeline

1. **Mapping** — `amt_lemma → ÜS candidate word(s)` from the reviewed CSV
   (precision-first tiered matching, reviewed earlier in the workflow).
2. **ÜS aggregation** — per ÜS word, distinct `semantic_types` in first-seen
   order across all its lexemes/meanings (same logic as `join_ys_semcat.py`).
3. **Person filter** — tags restricted to the whitelist (§3), order preserved.
4. **Problem set** — entries where step 3 yields nothing: **72** in total
   (45 with no ÜS types at all, 27 with only non-person types, `jooks`
   counted among the 27). Handled by the decisions TSV (§5).
5. **Write** — `Sem-Cat` value replaced in place; ÜS lowercase orthography
   kept verbatim (decision MJ: traceability to ÜS); multiple tags
   `", "`-joined; no fields added or removed (decision MJ: no `YS-lemma`
   restoration).
6. **Validate** — fail-fast in-script checks plus an independent post-hoc
   comparison (§8).

### 4.2 Sub-agent proposal stage (for the 72 problem entries)

Three parallel sub-agents (Claude general-purpose) each processed 24 entries.
Inputs per entry: `DEF_en`, `DEF_et`, current `Sem-Cat`, ÜS candidates and
their raw tags, plus the full ÜS word→tags lexicon (570 tagged words) for
analogy search. Method priority:

1. **Analogy** — find a semantically analogous ÜS word that *does* carry
   person tags (the model MJ set with jooks → kuller), e.g.
   `kalamüüja` → `müüja` (`in_elukutse`), `sigur` → `seakarjus`,
   `piirits` → `timukas`, `sõna` → `käskjalg`.
2. **Derivation** — otherwise map the old Sem-Cat into ÜS vocabulary
   (`IN_ELUKUTSE→in_elukutse`, `AGENT_TEGEVUS→in_tegija`, `IN_ROLL:*→in_roll`,
   `GRP_INIMENE/IN→inimene`, `IN_OMADUS→in_omadus`, `IN_RAHVAS→in_rahvas`,
   `IN_MÜT→in_müt`), overridden when the DEF fields clearly suggested better.

One chunk-boundary duplicate (`künnipoiss`, identical proposals) and one skip
(`kündja`, filled by the same `karjane` pattern as `külvaja`) were reconciled.
All 72 proposals were then **manually reviewed by MJ (2026-07-12)** — see §5.

## 5. The 72 manual decisions (all verified by MJ, 2026-07-12)

Source of truth: `Sem-Cat-YS-to-AMT/semcat-72-review.tsv`
(column `proposal` = adopted value; `status` = `verified 2026-07-12 (MJ)`).

MJ corrections to the sub-agent proposals during review (4 rows; final values
below are the **exact ÜS tags of the named analogue word**):

| lemma | agent proposal | **final (MJ)** | rule |
|---|---|---|---|
| sigur | in_tegija (via seakarjus) | **in_elukutse** | as ÜS **lambur** |
| tasuja | in_tegija | **in_tegija** | confirmed, as ÜS **kättemaksja** |
| teekäija | in_tegija | **in_tegija, in_omadus** | as ÜS **reisija** |
| väljamees | in_tegija | **in_tegija, in_omadus** | as ÜS **reisija** |
| tuleroog | in_müt, in_omadus | **in_omadus, in_roll, in_müt** | as ÜS **nõid** |

All 68 other rows adopted as proposed. Full table:

| lemma | old Sem-Cat | ÜS raw tags (non-person) | **adopted** | via ÜS analogue |
|---|---|---|---|---|
| ebausklik | IN_OMADUS | omadus | in_omadus | usklik |
| eeljooksja | AGENT_TEGEVUS | — | in_tegija | — |
| eelsõitja | AGENT_TEGEVUS | — | in_tegija | — |
| eesjooksja | AGENT_TEGEVUS | — | in_tegija | — |
| ettejooksja | AGENT_TEGEVUS | — | in_tegija, in_roll | eelkäija |
| haug | IN_OMADUS | loom_kala, toit | in_omadus | narr |
| hukataja | IN_OMADUS | — | in_omadus | söödik |
| hukkaja | IN_OMADUS | — | in_omadus | söödik |
| hukutaja | IN_OMADUS | — | in_omadus | petis |
| **jooks** | IN_ELUKUTSE | tegevus, vald, sündmus, ese_instru | **in_elukutse** | **kuller** (MJ decision; kuller absent from export, in_elukutse confirmed by MJ) |
| jutujätk | IN_OMADUS | esitus_keel_suhtlus, ese_semio | in_omadus | lobiseja |
| jõle | IN_OMADUS | omadus_kval, ADV_aste | in_omadus | narr |
| kalamüüja | IN_ELUKUTSE | — | in_elukutse | müüja |
| kaupleja | AGENT_TEGEVUS | — | in_tegija | ostja |
| kekk | IN_OMADUS | omadus_kval | in_omadus | narr |
| kokaemand | IN_ELUKUTSE | — | in_elukutse | kokk |
| kokanaine | IN_ELUKUTSE | — | in_elukutse | kokk |
| kost | IN_ROLL:staatus | toit, abstr/konkr | in_roll | külaline |
| kuppar | IN_ELUKUTSE | — | in_elukutse | velsker |
| kupparimoor | IN_ELUKUTSE | — | in_elukutse | velsker |
| kupulaskja | IN_ELUKUTSE | — | in_elukutse | habemeajaja |
| käsk | IN_ELUKUTSE | tegevus_kõnetegu | in_elukutse | käskjalg |
| külvaja | IN_ELUKUTSE | — | in_elukutse, in_tegija | karjane |
| kündja | AGENT_TEGEVUS | — | in_elukutse, in_tegija | karjane |
| künnipoiss | IN_ELUKUTSE | — | in_elukutse | sulane |
| lahter | IN_ELUKUTSE | esitus_kujutis, ese_instru, koht_hoone | in_elukutse, in_tegija | lihunik |
| lapse kaela murdja | IN_OMADUS | — | in_roll, in_omadus | tapja |
| laulutegija | IN_ELUKUTSE | — | in_elukutse | laulja |
| linarabaja | AGENT_TEGEVUS | — | in_tegija | peksja |
| linaropsija | AGENT_TEGEVUS | — | in_tegija | peksja |
| loba | IN_OMADUS | tegevus_kõnetegu | in_omadus | lobiseja |
| longus | IN_OMADUS | ADV_seisund | in_omadus | — |
| miilipõletaja | IN_ELUKUTSE | — | in_elukutse | — |
| mõisasulane | IN_ELUKUTSE | — | in_elukutse | sulane |
| naljakas | IN_OMADUS | omadus_kval | in_omadus | naljamees |
| ninakas | IN_OMADUS | omadus_psühh | in_omadus | ninatark |
| pagarisulane | IN_ELUKUTSE | — | in_elukutse | sell |
| petja | IN_OMADUS | — | in_omadus | petis |
| piirits | IN_ELUKUTSE | ese_instru | in_elukutse | timukas |
| poepoiss | IN_ELUKUTSE | — | in_elukutse | poodnik |
| pordik | IN_OMADUS | omadus_kval | in_omadus, in_roll | hoor |
| pügaja | IN_ELUKUTSE | — | in_elukutse | habemeajaja |
| raad | GRP_INIMENE | koht_asutus, esitus | inimene, in_roll | — |
| raamatumüüja | IN_ELUKUTSE | — | in_elukutse | müüja |
| rootsik | IN_RAHVAS | taim, taim_omadus, omadus_füüs | in_rahvas, in_omadus | — |
| salakoi | IN_OMADUS | seisund_haigus | in_omadus | — |
| **sigur** | IN_ELUKUTSE | taim, toit_maitseaine | **in_elukutse** | **lambur** (MJ) |
| sõna | IN_ELUKUTSE | esitus_keel, tegevus_kõnetegu, ese_semio, esitus_kujutis | in_elukutse | käskjalg |
| sõnapõlgaja | IN_OMADUS | omadus_kval | in_omadus | — |
| sõnavõtja | IN_OMADUS | — | in_omadus | — |
| sööbija | IN_OMADUS | seisund_haigus | in_omadus | söödik |
| tagaajaja | AGENT_TEGEVUS | — | in_tegija | ajaja |
| talgurahvas | GRP_INIMENE | — | inimene, in_roll | talguline |
| **tasuja** | IN_OMADUS | — | **in_tegija** | **kättemaksja** (MJ) |
| **teekäija** | IN | — | **in_tegija, in_omadus** | **reisija** (MJ) |
| teeäärne | IN_OMADUS | omadus_koht, koht_ala | in_omadus | kerjus |
| **tuleroog** | IN_ELUKUTSE | ese | **in_omadus, in_roll, in_müt** | **nõid** (MJ) |
| uue usu ning õpetuse tooja | IN_ROLL:ideol | — | in_roll | ketser |
| vahisoldat | IN_ELUKUTSE | — | in_elukutse | soldat |
| vahva tööinimene | IN_OMADUS | — | in_omadus | — |
| valelik | IN_OMADUS | omadus | in_omadus | — |
| vana agar tööinimene | IN_OMADUS | — | in_omadus | — |
| vananartsukaupmees | IN_ELUKUTSE | — | in_elukutse | kaupmees |
| võlu | IN_ELUKUTSE | omadus_kval, omadus_psühh, abstr/konkr, nähtus_psühh, tegevus, tegevus_tegu | in_müt, in_omadus | nõid |
| võõra jumala ori | IN_ROLL:ideol | — | in_roll | usklik |
| võõra jumala teener | IN_ROLL:ideol | — | in_roll | usklik |
| võõra vastuvõtja | IN_OMADUS | — | in_omadus | — |
| väärjumala paluja | IN_ROLL:ideol | — | in_roll | usklik |
| **väljamees** | IN | — | **in_tegija, in_omadus** | **reisija** (MJ) |
| väärjumala pidaja | IN_ROLL:ideol | — | in_roll | usklik |
| õitsiline | IN_ELUKUTSE | — | in_tegija, in_elukutse | karjane |
| äärne | IN_OMADUS | omadus, omadus_koht, koht_ala | in_omadus | kerjus |

## 6. Execution

```
$ uv run python scripts/fold_ys_semcat_into_master.py
882 entries -> AMT-Master_annotated_with_YS-semcat.json   (backup: AMT-Master_annotated_with_YS-semcat_BCKP-20260712-132807.json)
  Sem-Cat from ÜS via mapping : 810
  Sem-Cat from decisions TSV  : 72
  Sem-Cat values changed      : 882  (unchanged: 0)
  validation: entry count, field sets/order, non-Sem-Cat values, whitelist — all OK
```

Note: "changed: 882" means every `Sem-Cat` **value string** differs from
before, which is expected — even semantically matching categories changed
orthography (e.g. `IN_OMADUS` → `in_omadus`).

## 7. Resulting statistics

### 7.1 Tag distribution, before → after

| old (AMT vocabulary) | n | | new (ÜS vocabulary) | n |
|---|---|---|---|---|
| IN_ELUKUTSE | 414 | | in_elukutse | 385 |
| IN_OMADUS | 209 | | in_roll | 310 |
| IN_ROLL:staatus | 163 | | in_omadus | 261 |
| AGENT_TEGEVUS | 45 | | in_tegija | 153 |
| IN_ROLL:sugulus | 24 | | inimene | 58 |
| GRP_INIMENE | 22 | | esitus_tiitel | 27 |
| IN_ROLL:ideol | 22 | | in_müt | 21 |
| IN_RAHVAS | 11 | | in_sugulane | 13 |
| IN | 9 | | in_rahvas | 7 |
| IN_ROLL | 3 | | in_rahvas_keel | 1 |
| IN_MÜT | 3 | | | |

Total tag tokens: 925 before → 1 236 after — the annotation became noticeably
richer (goal: "as many ÜS semcats as possible").

### 7.2 Tags per entry (after)

| tags | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| entries | 592 | 237 | 45 | 6 | 1 | 1 |

### 7.3 Most frequent new Sem-Cat values

```
 237 in_elukutse                     27 in_elukutse, in_tegija
 134 in_roll                         24 in_omadus, in_roll
 119 in_omadus                       18 in_elukutse, in_omadus
  80 in_tegija                       14 inimene
  47 in_elukutse, in_roll            13 in_omadus, in_roll, in_müt
```

### 7.4 Most frequent old → new transitions (158 distinct)

```
 205 IN_ELUKUTSE      -> in_elukutse
  83 IN_OMADUS        -> in_omadus
  62 IN_ROLL:staatus  -> in_roll
  29 IN_ELUKUTSE      -> in_tegija
  26 IN_ELUKUTSE      -> in_elukutse, in_roll
  24 IN_ELUKUTSE      -> in_roll
  24 IN_ELUKUTSE      -> in_elukutse, in_tegija
  23 IN_OMADUS        -> in_tegija
  21 IN_OMADUS        -> in_omadus, in_roll
  20 AGENT_TEGEVUS    -> in_tegija
```

## 8. Validation (all passed)

In-script (fail-fast, aborts before writing on any violation):

- every prospective `Sem-Cat` non-empty and whitelist-only — **0 violations**;
- every decisions-TSV row matched exactly one master entry — **0 unmatched**;
- post-write re-read: entry count 882, field sets and field order per entry
  unchanged, any value difference confined to `Sem-Cat`.

Independent post-hoc check (separate process, comparing the written file
against the pre-run backup `_BCKP-20260712-132807.json`):

- entry count: **882 = 882**; `id` sequence identical → **no entry lost,
  no reordering**;
- fields with any differing value: **only `Sem-Cat`** (882 entries) →
  **no other label touched**;
- entries with empty `Sem-Cat`: **0** → **every entry has an ÜS semcat**;
- spot checks: `jooks → in_elukutse`, `sigur → in_elukutse`,
  `tuleroog → in_omadus, in_roll, in_müt`, `teekäija → in_tegija, in_omadus`,
  `keisriproua → esitus_tiitel`, `alv rahvas → in_omadus`,
  `abielurikkuja → in_omadus`, `abikaasa → in_sugulane` — all as decided.

## 9. Known caveats / open items

1. **kuller** is absent from the ÜS export (it only covers the 603 reviewed
   candidate words); `jooks = in_elukutse` rests on MJ's decision, not on an
   export row. If exactness matters, re-run the ÜS SQL including `kuller` and
   compare.
2. **teekäija / väljamees** carry `in_omadus` only because ÜS `reisija`
   carries it (probably from a figurative meaning of reisija); MJ adopted
   reisija's tags verbatim.
3. Where a person tag co-occurred with non-person tags on the same ÜS word,
   the non-person tags were **dropped, not recorded** in the JSON. The full
   pre-filter tag sets remain recoverable from the ÜS export + mapping CSV
   (deterministic), and the dropped-tag inventory is in §3.
4. The ÜS **provenance word** (`YS-lemma`) is intentionally not in the JSON
   (MJ decision: no fields added). It remains derivable from the mapping CSV;
   for the 72 manual cases the analogue word is in `semcat-72-review.tsv`.
5. `semcat-diff-report.html` still shows the pre-fold state (886 entries,
   2026-07-06). If a post-fold collation is wanted, `semcat_diff_report.py`
   would need a YS-Sem-Cat field to compare against — not applicable to the
   current single-field state.
6. The removed lemmas `at`, `att`, `essitaja`, `ohver` (pending promotion to
   master per the DEF-revision workflow) are **not** covered by this run; if
   they are later promoted, their Sem-Cat must be assigned by the same
   procedure (both have mapping rows in the CSV).

## 10. Reproducibility

```
cd <working-repo-root>
uv run python scripts/fold_ys_semcat_into_master.py
```

The script is idempotent given the same inputs (mapping CSV, ÜS export,
decisions TSV); it always writes a fresh timestamped backup before
overwriting. To revert this run:

```
cp Katus-ALUSANDMED/YS-Master-semcat-diff/Sem-Cat-YS-to-AMT/AMT-Master_annotated_with_YS-semcat_BCKP-20260712-132807.json \
   Katus-ALUSANDMED/YS-Master-semcat-diff/Sem-Cat-YS-to-AMT/AMT-Master_annotated_with_YS-semcat.json
```

## 11. File inventory (this run)

| File | Status |
|---|---|
| `Sem-Cat-YS-to-AMT/AMT-Master_annotated_with_YS-semcat.json` | **modified** — 882 × `Sem-Cat` replaced |
| `Sem-Cat-YS-to-AMT/AMT-Master_annotated_with_YS-semcat_BCKP-20260712-132807.json` | **new** — pre-run backup |
| `Sem-Cat-YS-to-AMT/semcat-72-review.tsv` | **new** — 72 verified manual decisions (input to the script; keep) |
| `Sem-Cat-YS-to-AMT/EXECUTION-REPORT_semcat-fold-in_20260712.md` | **new** — this report |
| `scripts/fold_ys_semcat_into_master.py` (published as `core/fold_ys_semcat_into_master.py`) | **new** — fold-in script |
| all other files | untouched |

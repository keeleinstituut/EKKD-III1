# Katus — companion scripts and audit logs

*Created: 2026-07-13 21-16-45 · Author: Madis Jürviste · Co-Authored-By: Claude Fable 5*

Transparency companion to the doctoral dissertation (the *kappa*, Estonian
*katuspeatükk*) on occupational titles and social roles in early Estonian
lexicography — the dictionaries of Stahl (1637), Gutslaff (1648), Göseken
(1660), Vestring (early 18th c.), Helle (1732) and Hupel (1780/1818).

This folder collects, in one reviewable place, **every script used to build
and analyse the L1 dataset** and **the audit logs of all machine-assisted
(LLM) work** on it, so that reviewers and opponents can inspect what was
done by hand, what was done by script, what was done by a language model —
and how each machine contribution was checked.

The Master dataset itself (AMT-Master: 866 lemmas / 1805 attestations /
2054 linked word forms) is published alongside this folder as
`../Andmestik/AMT-Master_annotated.json`; an excerpt is printed as Annex 1
of the dissertation. The six harmonized edition JSONs are available from
the author on request.

Author: Madis Jürviste. Prepared 2026-07-13, updated 2026-08-27;
pre-defence state.

---

## Layout

| folder | contents | kappa chapters |
|---|---|---|
| `core/` | the live data pipeline: source-edition converters, id injection, Master↔edition linkage, validation/verification, counting, report builders, the APTSK/VPTS overlap diff, the LexLex viewer builder and the Annex 1 printout builders | §3.1, §3.2, §4.1, Annex 1 |
| `retrodigitization/` | OCR and conversion pipelines for the Hupel dictionaries (TEI-XML→TXT converters, web-dictionary generators, Claude-API batch OCR of the 1818 edition) and the APTSK/VPTS PDF→JSON pipeline behind the overlap analysis | ch. 2, §3.2, §4.1, publication P5 |
| `annotation/` | annotation splicing for ch. 3 (Sem-Cat + definitions) and the ÜS/Sõnaveeb semantic-category transfer that replaced the machine-proposed categories | §3.1, §3.2 |
| `analysis/` | ch. 4 analyses (semantic change, "ghost professions", §4.3 thematic tagging) and the generators of Tables 3–13 | §4.1–4.3, tables in ch. 3–4 |
| `logs/` | the audit trail of every machine-assisted change to the dataset (see below) | §3.2 |
| `SCHEMA.md` | frozen schema of the edition JSONs and the Master, incl. the column→field maps per source and the Sugu (gender) tagging policy | §3.1 |

Every file here carries a provenance stamp in git convention —
`Created:` (original modification time), `Author: Madis Jürviste`, and one
`Co-Authored-By: Claude <model>` trailer per AI
model involved (a header comment in scripts and HTML, an italic line in
Markdown, `created`/`author`/`co_authored_by` meta keys in JSON). The
Claude model named is the one active in the session(s) that produced or
substantively edited the file, verified against session logs; files
created under Claude Opus 4.8 (June 2026) and substantively edited under
Claude Fable 5 (July–August 2026) name both. Exception: the Hupel
pipelines in `retrodigitization/` (Nov 2025 – Mar 2026, before the
session-log window used for verification): their `Created:` stamps are
recovered from the original files' modification times in the archived
subprojects, and — since per-file model attribution is not recoverable
there — they carry **no** `Co-Authored-By:` trailer rather than an
unverified one. These pipelines were developed in LLM-assisted workflows
(described in publication P5); the Claude models named in the
`hupel-1818-ocr/` filenames are the OCR engines the scripts invoke, not a
statement about who wrote the scripts.

`core/` scripts are the maintained pipeline and run from the original
repository root (`uv run python scripts/<name>.py`, Python 3.12). The
`retrodigitization/`, `annotation/` and `analysis/` folders are collected
copies of scripts that originally lived inside self-contained subprojects
with their own data layouts; they are published for inspection and are not
runnable as-is from this folder. Successive versions (e.g. the
`hupel-1780-convert-early/` experiments, `DO-NOT-USE-review_processor.py`)
are included deliberately: the development history is part of the record.

## How this maps to the kappa

- **§3.1 (the dataset).** Manual excerption produced the lemma inventory;
  the `core/convert_*.py` scripts turn the six source editions and the
  Master spreadsheet into harmonized JSON (schema in `SCHEMA.md`);
  `inject_*_ids.py`, `link_master*.py` and `reverse_links.py` build the
  id-based Master↔edition linkage (tiered, precision-first);
  `validate_master.py`, `verify_all.py` and `recompute_crosssource.py`
  enforce schema, id-uniqueness and attestation-count invariants.
- **§3.2 (method).** The three-method design — manual collection,
  Python scripting, LLM assistance — corresponds to this folder's
  structure: everything scripted is here, and everything LLM-assisted is
  logged in `logs/`.
- **§4.1–4.3.** `analysis/ptk-4-analysis/` holds the semantic-change
  candidate extraction, the frequency screening against a modern lemma
  list, and the four-axis thematic tagger for §4.3 with its manual-review
  fold-in; `analysis/tabelid-3-13/` regenerates every numbered table.
- **Annex 1 and LexLex.** `core/build_annex1_*.py` produce the printed
  dataset excerpt; `core/build_viewer.py` builds LexLex (Lexicon Lexicorum
  Esthonicorum), the self-contained HTML portal used to work with the
  dataset.

## The LLM audit trail (`logs/`)

Machine-generated content entered the dataset at three points: OCR of the
source scans, the English/Estonian definitions (`DEF_en`/`DEF_et`), and
proposal-stage semantic categories (later replaced entirely by curated
ÜS/Sõnaveeb tags). The definitions are the fully logged case:

| file | what it records |
|---|---|
| `DEF-changelog_reconstructed-full_20260713.json` | **The master audit document.** All 191 definition changes from the initial LLM generation (Claude Opus, baseline snapshot 2026-06-12) to the canonical dataset, each with old/new readings and provenance: 144 corrections from the logged LLM-as-judge review pass, 13 manual author edits reconstructed by snapshot diff, 34 changes from the documented data-quality round of 2026-07-13. Caveats are stated in its `meta` block. |
| `DEF-review_changelog_20260707.json` | The logged LLM-as-judge pass itself: every machine definition critically reviewed by a newer model (Claude Fable) against the sources' attested German glosses plus DWDS/EKSS/Sõnaveeb; 144 corrected entries with per-change reasons. |
| `DEF-review_emendationes_20260707.html` | Human-readable apparatus-style rendering of those 144 corrections (old reading struck, new reading; filterable: wording 77 / meaning 46 / German citation 21). |
| `DQ-fixes-changelog_2026-07-13.md` | Data-quality round of 2026-07-13: the T1–T20 and D1–D12 fixes, incl. the 12 record deletions (886 → 870). |
| `Sugu-policy-review_2026-07-13.md` | Full audit of the gender (`Sugu`) field against the tagging policy: all 263 M and 60 N records re-checked against the attested glosses; 3 changed, borderline cases listed for decision. |
| `EXECUTION-REPORT_semcat-fold-in_20260712.md` | Replacement of the machine-proposed semantic categories with person-type tags from the ÜS/Sõnaveeb database (810 records mapped directly, 72 decided manually). |
| `linkage_report.json`, `linkage_fuzzy_report.json` | Machine-readable reports of the tiered Master↔edition linkage runs (per-tier match counts, unresolved tail). |

Since 2026-07-13 the working repository is under version control, so all
later changes to data and scripts carry a commit-level trail; the period
before that is covered by the reconstructed changelog above, built from
dated snapshots whose earliest DEF-bearing state (2026-06-12) precedes any
editing. The later steps from 870 to the canonical **866** lemmas (the
author's manual review round of 2026-07-22: renames, three merges, one
deletion, two sense narrowings) and the thematic/semantic retags of
2026-08-10 are all commit-trailed there.

## Known limitations

- The audit logs cover the `DEF`, `Sugu` and `Sem-Cat` fields and the
  linkage; manual edits to other fields before 2026-07-13 are diffable
  from archived snapshots but not itemized here.
- The changelog reconstruction cannot see edits made and reverted between
  snapshots (details in the `meta.caveats` of the reconstructed changelog).
- Re-running an edition converter regenerates that edition's uuids and
  would break existing Master↔edition links; the linkage, counting and
  report scripts are idempotent and safe to re-run.

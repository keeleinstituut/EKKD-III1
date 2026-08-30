# Sugu policy review — 2026-07-13

*Created: 2026-07-13 14-32-28 · Author: Madis Jürviste · Co-Authored-By: Claude Fable 5*

Review of the `Sugu` field of `AMT-Master_annotated.json` (870 records; 263 M + 60 N reviewed) against the kappa-bound policy of 2026-07-13. Evidence used: `Amt-Master-ID`, `DEF_et`, `DEF_en` and the six attested German gloss fields (placeholders `---`/`???`/`NULL` = no evidence).

## 1. The policy (MJ, 2026-07-13, kappa-bound)

> `Sugu: "M"` only where (a) a clear masculine marker in the Estonian word (e.g. compound-final "-mees"), or (b) a clear masculine marker in the attested German glosses, or (c) the general sense indicates M. Same criteria for `"N"` (feminine). Where BOTH M and N are present in the evidence, OR the sense allows for Ü (either gender possible), `"Ü"` applies.

Fixed points not touched: all lemmas in "-mees" = M (systematic rule D4); `junkur` = Ü (D1); `leskmees` = M, `lesknaine` = N, `lesk` = Ü, `teeäärne`/`äärne` = Ü. All verified in place.

## 2. Applied now (3 changes)

Changed only where the record's own evidence explicitly shows both genders.

1. **lapse kaela murdja**: N → **Ü**
   - Göseken-1660: `kinder Mörder` (masculine) vs. Vestring: `Ein Kinder Mörderinn`, Helle: `eine Kinder-Mörderin`, Hupel: `eine Kindermörderin` (all feminine -in).
   - One source masculine, three feminine → both genders explicitly attested.

2. **lapselootaja**: N → **Ü**
   - Vestring: `Der die Frucht abtreibt` (explicit masculine relative "Der") vs. Helle: `die die Frucht abtreibt`, Hupel: `die die Frucht abtreibt (r.)` (feminine "die").
   - DEF_et itself is gender-neutral: "loodet hävitav **inimene**, abordi tegija". Masculine and feminine reference both explicitly in the gloss text. (Weakest of the three applied — the contrast rests on the article/pronoun, not a suffix pair; easy to revert if MJ reads Vestring's "Der" as a slip.)

3. **leivavanemad**: M → **Ü**
   - Hupel: `Brodeltern, Herrschaft der man dient` — **Brodeltern** ("bread-parents", Eltern) explicitly names both genders; Vestring `Herrschafft`, Helle `Herrschaft, der man dienet` are gender-neutral collectives. No masculine marker anywhere in lemma, DEF or glosses.

Named suspect **meier** (N) was verified and NOT changed: its only evidence is Göseken `KebsWeib; beyschläfferin` — purely feminine. No source glosses the masculine steward sense (*Meier*); DEF_en explicitly "concubine, mistress (not 'dairy steward')". Stays N, policy-conformant.

## 3. Candidates for Ü — MJ decision needed (7)

M/N records where the policy verdict leans Ü but the evidence is inferential (gender-neutral sense, bare -er / bare masculine nouns, collective glosses) rather than explicit. Ordered by confidence, strongest case first.

| # | Lemma | Sugu | Evidence | Rationale |
|---|-------|------|----------|-----------|
| 1 | **ori** | M | St `Diener`; Gu `Dienst`; Ve `Ein Dienstbote Sclave`; He `der Bediente`; Hu `der Bediente, Diener, Sklav`; DEF "ori, sulane, teener" | Sense 'slave, servant' applies to both genders (the corpus itself splits it: `orjapoiss` M / `orjatüdruk` N); glosses are bare masculine/neutral nouns, no clear marker. |
| 2 | **tunnistaja** | M | All five sources: `Zeuge` / `der Zeuge`; DEF "tunnistaja (kohtus)" | Witness is a role open to either gender; `Zeuge` is a bare grammatically-masculine noun (weak evidence per policy); no Estonian marker. |
| 3 | **valetunnistaja** | M | Ve `Falscher Zeuge`; Hu `ein Falscher Zeuge` | Same as tunnistaja: bare `Zeuge`, sense allows either gender. |
| 4 | **mõisavanemad** | M | Ve `Hofes Herrschafft`; He `Herrschaft aufm Hofe`; Hu `Hofsherrschaft` | `Herrschaft` is a gender-neutral collective (referentially includes the lady of the manor); parallel to leivavanemad but without the explicit `Eltern`, so inferential only. Contrast `maavanemad`, which has `Ober Herren` and stays M. |
| 5 | **kohtuvanem** | M | St `die Obrigkeit`; Gö `Oberkeit`; Ve/He/Hu `Obrigkeit`; DEF "kohtuvanemad, ülemus, võimukandjad (mitmus)" | Glossed only with the abstract collective 'authorities'; no personal gender marker at all. M rests solely on the era's male-only offices (clause c). |
| 6 | **sillavanemad** | M | Ve `das Ordnungs gerichte` | Sole gloss names the institution (the court), not persons; M is purely inferential from male-only court membership. |
| 7 | **õppija** | M | St `Lehrer, Prediger`; Gö `prediger`; Ve `Ein Lehrmeister; Ein Lernender`; He `der Lehrer`; Hu `ein Schüler, Lernender` | The younger sense 'learner, pupil' (Vestring/Hupel) is potentially either gender; only bare masculine forms attested. Older 'Prediger' sense anchors M — hence lowest confidence. |

Systematic coverage: all 323 M/N records were reviewed. 3 changed (§2), 7 flagged above, **313 cleared as policy-conformant** (§4). Nothing skipped.

## 4. Confirmed conformant (313 records)

- **M by Estonian masculine marker** (clause a): 111 lemmas ending in "-mees" (D4 rule; e.g. kaupmees, sõjamees, peremees, töömees), plus multi-word lemmas ending/containing "mees" (asuja mees, tapja mees, vahva mees, mees oma naha sees …); "-poiss/-poeg/-isa/-härra/-isand" compounds (kokapoiss, tallipoiss, talupoeg, perepoeg, pereisa, kirikuhärra, kohtuisand, raeisand, vaekojaisand …).
- **M by German masculine marker** (clause b): Herr/Mann/Knecht/Kerl/Vater/Junge/Bube glosses — isand/härra/ülem/vanem (`Herr(en)`, Stahl `Oberheir` = Oberherr), sulane + compounds (`Knecht`), rehepapp/töömees-type `Kerl` glosses (tüma saks, vabatmees, pärismees), kirikuvanem (`Kirch-Vatter`), künnipoiss/teopoiss (`ein Junge der …`), puduvägi/trossipoiss (`tros bube`), kaarman/tüürman (`Fuhrman`, `Steurmann`), naiseropsija (`der sein Weib prügelt`), vanakõu (`Anherr`), vader (`Gevatter`).
- **M by sense** (clause c): clergy and church offices (papp, preester, piiskop, paavst, munk, abt, kaplan, praost, köster, superdent, pealevaataja, õpetaja/jutleja/ütleja/jutustaja `Prediger`), rulers/officials (keiser, kuningas, vürst, krahv, kuberner, taatholder, kantsler, pormeister, manrihter, sundija, õigusemõistja, külavanem `Dorfschulze`, vöörmünder `Vormund`), military (soldat, soldan, rüütel, tragun, kindral, ooberst, kapten, rittmeister, marssal, pealik `Hauptmann`, vahisoldat, istja/sötiku `Soldaten in Winterquartier`), craft-master titles (meister, koolmeister, tallmeister, veomeister, rahajuhataja `Münzmeister`, kunsikas `Hexenmeister`, viks tegija `ein guter Meister`), male-sense terms (pordupealine/portja `Hurer`, meestepidaja/poistepidaja/poisteteotaja `Knabenschänder`, poissmees-type bachelor terms `lediger Gesell/Kerl`, toaselts `Stuben Gesell`).
- **N by Estonian feminine marker**: 16 "-naine" (perenaine, lesknaine, külanaine …), 5 "-tüdruk" (köögitüdruk, orjatüdruk, saksatüdruk …), 8 "-emand/-proua" (kuningaemand, abtiemand, keisriproua, ülem proua …), plus pereema, peretütar, kupparimoor, vanaämm, eksinud naisterahvas.
- **N by German feminine marker**: -in glosses (kedranaine `Spinnerin`, riidepesija `die Wäscherin`, hoorasaadik/hooraperenaine `Kuplerin`, orjatüdruk `Dienerin`, emandatüdruk `Nachträterin`, prohvetiemand `Prophetin`, võla `Zauberin`, vahva naine `Heldin`), Weib/Magd/Hure glosses (amm/avanisa-amm/imetaja naine `Saugend Weib, Saug Amme`, tüdruk/ümmardaja `Magd`, kaasnaine/liignaine/meier `Kebsweib`, rootsik `Weibsperson`, karjanaine `Viehweib`).
- **N by sense**: 11 whore-terms (hoor, lits, port, pordik, turuhoor, tallatav, kiimane/mädanud/sõidetud hoor …), neitsi/neitsike `Jungfer`, nunn, paademooder `Hebamme`, leinanaine `Klageweib`, hatt `Hündin/Tiffe`, vana nõid/tuleroog `Hexe`.

## 5. Sugu distribution

| Sugu | Before | After |
|------|-------:|------:|
| Ü | 547 | **550** |
| M | 263 | **262** |
| N | 60 | **58** |
| total | 870 | 870 |

Serialization: `ensure_ascii=False, indent=2` + single trailing newline. `scripts/validate_master.py` exit 0 (all 14 checks OK). Diff against pre-edit copy confirms exactly three `Sugu` lines changed; no Comment fields touched; edition JSONs untouched.

---

## MJ decisions on the candidates (2026-07-13) — APPLIED

| lemma | decision | rationale | applied |
|---|---|---|---|
| ori | Ü | sense | M → Ü |
| tunnistaja | Ü | sense | M → Ü |
| valetunnistaja | Ü | sense | M → Ü |
| mõisavanemad | Ü | sense | M → Ü |
| kohtuvanem | M | sense: historically F not plausible | stays M |
| sillavanemad | M | sense: historically F not plausible | stays M |
| õppija | Ü | sense | M → Ü |

Final Sugu distribution after decisions: **Ü 555 / M 257 / N 58** (870 records). Validator exit 0. No open Sugu items remain; the policy paragraph for the kappa is the only outstanding D2 follow-up.

# Katuspeatükis kasutatud skriptid ja logid

*Uuendatud: 2026-08-27 · Autor: Madis Jürviste · Kasutatud keelemudel: Claude Fable 5*

Lisamaterjal MJ doktoriväitekirja katuspeatükile, mis käsitleb ametite ja sotsiaalsete rollide nimetusi vanemas eesti leksikograafias. Põhiallikateks on esimesed eesti keelt sisaldavad trükis avaldatud sõnastikud: Stahl (1637), Gutslaff (1648), Göseken (1660), Vestring (2000 [?1710–1730]), Helle (1732) ja Hupel (1780, 1818). Siinsesse repositooriumisse on kogutud skriptid, mida on kasutatud L1 andmestiku koostamiseks ja analüüsimiseks, samuti logid käsitsianalüüsi, skriptide toel ning SKMidega loodud sisu jälgimiseks.

Koondandmestik (AMT-Master: 866 märksõna / 1805 esinemust / 2054 seotud sõnakuju) on avaldatud failina `../Andmestik/AMT-Master_annotated.json`. Andmestiku väljavõte on lisatud katuspeatüki teksti lisas.

---

## Ülesehitus

| kaust | sisu | seos katuspeatükiga |
|---|---|---|
| `core/` | töös olev andmetöötlusahel: allikaväljaannete konverterid, id-de lisamine, Master JSONi ja väljaannete linkimine, valideerimine ja kontroll, loendamine, vaheanalüüsid, kattuvusvõrdlus VPTSiga, LexLexi prototüübi koostamise skript ning lisa 1 trükiväljavõtte konverteerimise skriptid | §3.1, §3.2, §4.1, lisa |
| `retrodigitization/` | Hupeli sõnaraamatute tärktuvastuse ja konverteerimise skriptid (TEI-XML→TXT-konverterid, veebisõnastike generaatorid, 1818. aasta väljaande OCRi API-masspäringuskriptid) ning kattuvusanalüüsi aluseks olev VPTSi PDF→JSON-tööahel | ptk 2, §3.2, §4.1, publikatsioon P5 |
| `annotation/` | 3. peatüki annotatsioonide ühendamine (Sem-Cat + definitsioonid) ning ÜSi/Sõnaveebi semantiliste kategooriate lisamine, millega asendati SKMi genereeritud kategooriad | §3.1, §3.2 |
| `analysis/` | 4. peatüki analüüsid (tähendusmuutused, „kummitusametid", §4.3 temaatiline märgendamine) ning tabelite 3–13 generaatorid | §4.1–4.3, ptk 3–4 tabelid |
| `logs/` | kõigi masintoega andmestikumuudatuste auditijälg (vt allpool) | §3.2 |
| `SCHEMA.md` | väljaande-JSONide ja Masteri andmeskeem, sh iga allika veergude ja väljade vastavustabelid ning sootunnuse (`Sugu`) märgendamise põhimõtted | §3.1 |

Igal siin repositooriumis talletatud failil on git'i tavapärane päritolutempel — `Created:` (faili algne muutmisaeg), `Author: Madis Jürviste` ning iga osalenud tehisintellektimudeli kohta üks rida `Co-Authored-By: Claude <mudel>` (skriptides ja HTML-failides päisekommentaarina, Markdownis kaldkirjareana, JSONis metaväljadena `created`/`author`/`co_authored_by`). Nimetatud Claude'i mudel on see, mis oli aktiivne seansis või seanssides, kus fail loodi või kus seda sisuliselt muudeti; andmed on kontrollitud seansilogide põhjal. Failid, mis on loodud Claude Opus 4.8-ga (juunis 2026) ja mida on sisuliselt muudetud Claude Fable 5-ga (juuli–august 2026), nimetavad mõlemat mudelit. Erandiks on Hupeli töövoo ahelad kaustas `retrodigitization/` (november 2025 – märts 2026, st enne kontrollimiseks kasutatud seansilogide perioodi algust): nende `Created:`-templid on taastatud algfailide muutmisaegadest arhiveeritud alamprojektides ning kuna failipõhine mudeli omistamine ei ole seal taastatav, ei sisalda need ühtegi `Co-Authored-By:` rida. Need ahelad on välja töötatud keelemudeli toega töövoogudes (kirjeldatud publikatsioonis P5); kausta `hupel-1818-ocr/` failinimedes nimetatud Claude'i mudelid on OCRi-tööriistad, mida skriptid välja kutsuvad, mitte väide skriptide autorsuse kohta.

Kausta `core/` skriptid moodustavad andmetöötlusahela ja käivituvad algse repositooriumi juurkataloogist (`uv run python scripts/<nimi>.py`, Python 3.12). Kaustad `retrodigitization/`, `annotation/` ja `analysis/` sisaldavad koopiaid skriptidest, mis algselt asusid iseseisvates, oma andmepaigutusega alamprojektides; need on avaldatud ülevaatamiseks ega ole sellest kaustast muutmata kujul käivitatavad. Järjestikused versioonid (nt katsetused kaustas `hupel-1780-convert-early/`, fail `DO-NOT-USE-review_processor.py`) on kaasatud teadlikult skriptide arengu jälgimiseks.

## Seos katuspeatükiga

- **§3.1 (andmestik).** Märksõnad on välja valitud käsitsi. Skriptid `core/convert_*.py` teisendavad kuus allikaväljaannet ja Master-faili tabeli ühtlustatud JSONiks (skeem failis `SCHEMA.md`); `inject_*_ids.py`, `link_master*.py` ja `reverse_links.py` ehitavad id-põhise Masteri JSONi ja väljaannete linkimise (astmeline, täpsus esikohal); `validate_master.py`, `verify_all.py` ja `recompute_crosssource.py` jõustavad skeemi, id-de unikaalsuse ja esinemusloendite invariandid.
- **§3.2 (meetod).** Kolme meetodi ülesehitus — käsitsi kogumine, Pythoni skriptid, keelemudeli tugi — vastab selle kausta struktuurile: kõik skriptidega tehtu on siin ning kõik keelemudeli toel tehtu on logitud kaustas `logs/`.
- **§4.1–4.3.** Kaustas `analysis/ptk-4-analysis/` on tähendusmuutuse kandidaatide eraldamine, sagedusfilter tänapäevase lemmaloendi suhtes ning §4.3 nelja teljega temaatiline märgendaja koos käsitsi ülevaatuse tulemuste sissekandmisega; `analysis/tabelid-3-13/` genereerib uuesti
  kõik nummerdatud tabelid.
- **Lisa ja LexLex.** Skriptid `core/build_annex1_*.py` koostavad andmestiku trükiväljavõtte; `core/build_viewer.py` ehitab LexLexi
  (*Lexicon Lexicorum Esthonicorum*), iseseisva HTML-portaali, mida on kasutatud andmestikuga töötamiseks.

## Keelemudelitöö auditijälg (`logs/`)

Masinloodud sisu jõudis andmestikku kolmes kohas: allikaskaneeringute OCR, inglis- ja eestikeelsed definitsioonid (`DEF_en`/`DEF_et`) ning ettepanekujärgus semantilised kategooriad (hiljem täielikult asendatud ÜSi/Sõnaveebi kureeritud märgenditega). Definitsioonid on täies mahus logitud:

| fail | mida see dokumenteerib |
|---|---|
| `DEF-changelog_reconstructed-full_20260713.json` | **Auditi põhidokument.** Kõik 191 definitsioonimuudatust alates esialgsest keelemudeli genereeritud sisust (Claude Opus, lähteseis 2026-06-12) kuni lõpliku andmestikuni, igaüks koos vana ja uue lugemi ning päritoluga: 144 parandust logitud LLM-hindaja (*LLM-as-judge*) ülevaatusvoorust, 13 autori käsitsi tehtud parandust, mis on taastatud snapshot'ide võrdluse teel, ning 34 muudatust 2026-07-13 dokumenteeritud andmekvaliteedivoorust. Reservatsioonid on esitatud faili `meta`-plokis. |
| `DEF-review_changelog_20260707.json` | Logitud LLM-hindaja ülevaatus ise: iga masindefinitsioon on kriitiliselt üle vaadatud uuema mudeliga (Claude Fable), võrreldes allikates atesteeritud saksa vastetega ning DWDSi/EKSSi/Sõnaveebiga; 144 parandatud kirjet koos muudatusepõhiste põhjendustega. |
| `DEF-review_emendationes_20260707.html` | Nende 144 paranduse inimloetav, tekstikriitilise aparaadi laadis esitus (vana lugem läbikriipsutatuna, kõrval uus; filtreeritav: sõnastus 77 / tähendus 46 / saksa tsitaat 21). |
| `DQ-fixes-changelog_2026-07-13.md` | 2026-07-13 andmekvaliteedivoor: parandused, sh 12 kirje kustutamine (886 → 870). |
| `Sugu-policy-review_2026-07-13.md` | Soovälja (`Sugu`) täielik audit märgendamispõhimõtete suhtes: kõik 263 M- ja 60 N-kirjet on atesteeritud vastete põhjal uuesti üle kontrollitud; 3 muudetud, piiripealsed juhtumid on loetletud otsustamiseks. |
| `EXECUTION-REPORT_semcat-fold-in_20260712.md` | Masina pakutud semantiliste kategooriate asendamine ÜSi/Sõnaveebi andmebaasi isikutüübimärgenditega (810 kirjet vastendatud otse, 72 otsustatud käsitsi). |
| `linkage_report.json`, `linkage_fuzzy_report.json` | Masinloetavad aruanded Master JSONi ja väljaannete astmelise linkimise käitustest (vastete arv astmete kaupa, lahendamata jääk). |

## Teadaolevad piirangud

- Auditilogid katavad väljad `DEF`, `Sugu` ja `Sem-Cat` ning linkimise; enne 2026-07-13 muudes väljades tehtud käsitsi parandused on arhiveeritud snapshot'ide põhjal võrreldavad, kuid siin ükshaaval loetlemata.
- Muudatuslogi taastamine ei näe muudatusi, mis on tehtud ja tagasi võetud kahe snapshot'i vahel (üksikasjad taastatud muudatuslogi väljas `meta.caveats`).
- Väljaandekonverteri uuesti käivitamine genereeriks selle väljaande uuid-d uuesti ja lõhuks olemasolevad Masteri↔väljaande lingid; linkimis-, loendus- ja aruandeskriptid on idempotentsed ning neid on ohutu uuesti käivitada.

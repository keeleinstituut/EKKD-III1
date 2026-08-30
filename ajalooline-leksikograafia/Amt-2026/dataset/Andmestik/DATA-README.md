# AMT-Master_annotated.json — andmestiku kirjeldus

*Created: 2026-08-28 · Author: Madis Jürviste · Co-Authored-By: Claude Fable 5*

<!-- TODO (MJ): litsents — nt CC BY 4.0? Lisada rida "Litsents: …" ja LICENSE-fail repo juurde. -->

L1 koondandmestik: **866 märksõna** (ametite ja sotsiaalsete rollide nimetused vanemas eesti leksikograafias), igaüks koos definitsioonide, kolme liigitusteljega ning kuue allikväljaande atesteeritud sõnakujude ja saksa vastetega. Seis: 2026-08-10 (v10; vt `../Skriptid/README.md` logide osa). Faili struktuur: üks juurvõti `"AMT-Master"` → 866 kirjet, igaühel identne 30-võtmeline skeem kindlas järjestuses.

## Väljad

| väli | sisu |
|---|---|
| `Amt-Master-ID` | märksõna (primaarvõti; eesti tähestikujärjestus, tühik/sidekriips enne tähti) |
| `id` | püsiv uuid7-põhine id (`am-…`) |
| `Amt-Cat` | sotsiaalne kategooria: K1 (põhielukutsed, 371), K2 (juhutegevused, 159), K3 (sotsiaalsed rollid, 318) + kombinatsioonid `K1, K2` (3), `K1, K3` (11), `K2, K3` (4) |
| `Sem-Cat` | ÜSi/Sõnaveebi isikutüübimärgendid, komadega: `in_elukutse` (369), `in_roll` (312), `in_omadus` (233), `in_tegija` (144), `esitus_tiitel` (20), `in_müt` (18), `in_sugulane` (9), `in_rahvas` (4) |
| `Teema` | temaatiline kategooria (12): OMADUS_SEOS_KUULUVUS (212), KÄSITÖÖ (200), HALDUS_VÕIM (140), MORAAL_HÄLVE (124), TEENISTUS (62), KIRIK_FUNKTSIOON (34), FEOD_MAA (31), NÕID (27), KIRIK_VAIMULIK (21), HARIDUS (8), SUGU_REPRO (5), MÜÜT (2) |
| `Sugu` | Ü (554) / M (254) / N (58) — M/N ainult selge sootunnuse korral, muidu Ü (poliitika: `../Skriptid/5-logs/Sugu-policy-review_2026-07-13.md`) |
| `DEF_en`, `DEF_et` | definitsioonid; `DEF_et` lõpus sageli atesteeritud saksa vaste `(sks …)` |
| `<Allikas>-et`, `<Allikas>-de`, `<Allikas>-id` ×6 | allika eesti sõnakuju(d), saksa vaste(d), lingitud väljaandekirjete id-loend (Stahl-1637, Gutslaff-1648, Göseken-1660, Vestring-17XX, Helle-1732, Hupel-1780-est-ger) |
| `Cross-source count` | JSON-täisarv 1–6: mitmes allikas on reaalne `-et`-atesteering (jaotus: 1×446, 2×130, 3×154, 4×70, 5×39, 6×27) |
| `Comment-1/2/3` | alati `"NULL"` — 503 sisulist kommentaari on arhiveeritud eraldi töödokumenti (vt DQ-logi otsus J1) |

## Väärtusmärgendid

- `"---"` — selles allikas atesteerimata (paariline `-id` on siis `[]`).
- `"NULL"` — väärtus puudub. Erijuht `Vestring-17XX-de`: atesteeritud Vestringis, aga allikas ei anna saksa vastet.
- `"???"` — allikas lahendamata/loetamatu (reserveeritud; praegu kasutusel pole).
- `[xN]` sõnakuju järel — kuju esineb allika kirjes N korda. Avaldatud seisus kaks lahtrit: `huckaja [x4]`, `pettis [x8]` (mõlemad `Göseken-1660-et`); tahtlikult säilitatud, vorming vastab valideerija reeglile.
- `{…}`, `[Gram]` jms — Vestringi/Gösekeni transkriptsioonidest pärinev toimetuslik märgistus, üle võetud muutmata kujul.

Täielik skeemikirjeldus ja allikate veerukaardistused: `../Skriptid/SCHEMA.md`. Terviklikkuse kontroll: `uv run python ../Skriptid/1-core/13-validate_master.py AMT-Master_annotated.json` (exit 0 = korras).

## Päritolu

Märksõnad on valitud käsitsi; definitsioonid on algselt genereeritud keelemudeliga ja seejärel läbinud logitud ülevaatus- ja parandusvoorud; semantilised kategooriad pärinevad ÜSi/Sõnaveebi kureeritud märgenditest. Täielik auditijälg: `../Skriptid/5-logs/` (vt `../Skriptid/README.md`).

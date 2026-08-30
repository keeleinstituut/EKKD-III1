# Created: 2026-07-12 14-10-30
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
# Tabel 8. L1-andmestiku temaatiline jaotus.
# Teemasildid kanoonilisest andmestikust (FEOD_MAA, nagu senises tabelis).
# Reajärjestus nagu senises tabelis: KÄSITÖÖ esimesena, ülejäänud kahanevas järjestuses.
# Käivita: uv run python tabel_08.py (andmefaili asukoht: muutuja MASTER, vaikimisi töörepositooriumi paigutus)
import json
from collections import Counter
from pathlib import Path

MASTER = Path("Katus-ALUSANDMED/json-all/AMT-Master_annotated.json")

LABELS = {
    "KÄSITÖÖ":          "KÄSITÖÖ / käsitöö, kaubandus, õppides omandatud elukutsed",
    "OMADUS_SEOS_KUULUVUS": "OMADUS_SEOS_KUULUVUS / sugulus, isikuomadused, võõramaalased, üldmõisted",
    "HALDUS_VÕIM":      "HALDUS_VÕIM / haldus, võim, sõjavägi, õigus",
    "MORAAL_HÄLVE":     "MORAAL_HÄLVE / moraalne hälve, kuritegu, patt",
    "TEENISTUS":        "TEENISTUS / teenistus, alluv töö",
    "FEOD_MAA":       "FEOD_MAA / maaga, talupoja staatusega seotud nimetused",
    "KIRIK_FUNKTSIOON": "KIRIK_FUNKTSIOON / kirikuametid, ilmikud, uskumused",
    "NÕID":             "NÕID / nõidus, ennustamine, rahvamaagia",
    "KIRIK_VAIMULIK":   "KIRIK_VAIMULIK / ordineeritud vaimulikkond",
    "HARIDUS":          "HARIDUS / haridus: õpetajad, õpilased",
    "SUGU_REPRO":       "SUGU_REPRO / sünnitus, imetamine",
    "MÜÜT":             "MÜÜT / mütoloogilised olendid",
}

rows = json.loads(MASTER.read_text(encoding="utf-8"))["AMT-Master"]
total = len(rows)
teema = Counter(e["Teema"] for e in rows)

order = ["KÄSITÖÖ"] + sorted((k for k in teema if k != "KÄSITÖÖ"), key=lambda k: -teema[k])

print("| Teema | Lemmasid | Osakaal (%) |")
print("|---|---|---|")
for k in order:
    pct = f"{teema[k] / total * 100:.1f}".replace(".", ",")
    print(f"| {LABELS.get(k, k)} | {teema[k]} | {pct} |")
print(f"| Kokku | {total} | 100 |")

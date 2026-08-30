# Created: 2026-07-12 14-10-30
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
# Tabel 3. L1-andmestiku koosseis autorite lõikes.
# Märksõna-artikleid: Stahl/Gutslaff/Göseken/Helle = senised kirjanduspõhised arvud;
# Vestring/Hupel = json-all digiteeritud väljaannete tegelik kirjete arv.
# Massiivis sõnakujusid = lingitud väljaandekirjete arv (<Source>-id loendite pikkuste summa).
# Käivita: uv run python tabel_03.py (andmefaili asukoht: muutuja MASTER, vaikimisi töörepositooriumi paigutus)
import json
from pathlib import Path

MASTER = Path("Katus-ALUSANDMED/json-all/AMT-Master_annotated.json")
PLACE = {"", "---", "???", "NULL"}

# (Autor, Suund, Aasta, JSON-võtmeprefiks, märksõna-artikleid)
SOURCES = [
    ("Stahl",    "DE-ET", "1637", "Stahl-1637",         "2309"),
    ("Gutslaff", "DE-ET", "1648", "Gutslaff-1648",      "1714"),
    ("Göseken",  "DE-ET", "1660", "Göseken-1660",       "9941"),  # MJ correction 2026-08-10 (was 9000)
    ("Vestring", "ET-DE", "17XX", "Vestring-17XX",      "6953"),
    ("Helle",    "ET-DE", "1732", "Helle-1732",         "6400"),
    ("Hupel",    "ET-DE", "1780", "Hupel-1780-est-ger", "13 732"),
]


def present(e, key):
    return isinstance(e.get(key), str) and e[key].strip() not in PLACE


rows = json.loads(MASTER.read_text(encoding="utf-8"))["AMT-Master"]
total = len(rows)

print(f"| Autor | Suund | Aasta | Märksõna-artikleid | Massiivis sõnakujusid |"
      f" Autori lemmade osakaal massiivis ({total}) |")
print("|---|---|---|---|---|---|")
for autor, suund, aasta, key, artiklid in SOURCES:
    att = sum(1 for e in rows if present(e, f"{key}-et"))
    kujud = sum(len(e.get(f"{key}-id") or []) for e in rows)
    print(f"| {autor} | {suund} | {aasta} | {artiklid} | {kujud} | {att / total * 100:.0f}% |")

# Created: 2026-07-12 14-10-30
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
# Tabel 7. Ametisõnavara osakaal sõnastikes.
# L1-lemmasid = massiivis reaalselt esindatud lemmad (platshoidjad --- / ??? / NULL välja).
# Märksõna-artikleid kokku: Stahl/Gutslaff/Göseken/Helle = senised kirjanduspõhised arvud
# (kuvatud ~ümardatuna nagu seni), Vestring/Hupel = json-all väljaannete tegelik kirjete arv.
# Ametisõnavara osakaal = lemmad / artiklid (suhe arvutatud täpse artikliarvuga).
# Käivita: uv run python tabel_07.py (andmefaili asukoht: muutuja MASTER, vaikimisi töörepositooriumi paigutus)
import json
from pathlib import Path

MASTER = Path("Katus-ALUSANDMED/json-all/AMT-Master_annotated.json")
PLACE = {"", "---", "???", "NULL"}

# (kuvanimi, JSON-võtmeprefiks, artikleid_täpne, artikleid_kuvatav)
SOURCES = [
    ("Hupel 1780 et-de", "Hupel-1780-est-ger", 13732, "13 732"),
    ("Göseken 1660",     "Göseken-1660",        9941, "9 941"),  # MJ correction 2026-08-10 (was ~9 000)
    ("Vestring ~1720",   "Vestring-17XX",       6953, "6 953"),
    ("Helle 1732",       "Helle-1732",          6400, "~6 400"),
    ("Stahl 1637",       "Stahl-1637",          2309, "~2 300"),
    ("Gutslaff 1648",    "Gutslaff-1648",       1714, "~1 700"),
]


def present(e, key):
    return isinstance(e.get(key), str) and e[key].strip() not in PLACE


rows = json.loads(MASTER.read_text(encoding="utf-8"))["AMT-Master"]
total = len(rows)

print(f"| Sõnastik | L1-andmestikus lemmasid | % kogu andmestikust ({total}) |"
      f" Märksõna-artikleid kokku | Ametisõnavara osakaal kõigist märksõna-artiklitest |")
print("|---|---|---|---|---|")
for name, key, art, art_disp in SOURCES:
    att = sum(1 for e in rows if present(e, f"{key}-et"))
    pct = f"{att / total * 100:.1f}%"                       # punktiga, nagu senises tabelis
    share = f"~{att / art * 100:.1f}%".replace(".", ",")    # komaga, nagu senises tabelis
    print(f"| {name} | {att} | {pct} | {art_disp} | {share} |")

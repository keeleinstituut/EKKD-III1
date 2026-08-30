# Created: 2026-07-12 14-10-30
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
# Tabel 10. Nõianimetused sõnastikes.
# Teema = NÕID; iga allika veerus esindatud lemmade sõnakujud master-tabeli
# <Source>-et lahtritest, tähestikuliselt; lahtrisisesed variandid eraldajaga " / ".
# Erand (toimetuslik otsus, MJ 2026-07-12): Vestringi "Att" (lemma *hatt*, Teema MORAAL_HÄLVE)
# lisatakse Vestringi ritta, sest Vestringi tõlkevaste on "Hexe, Hure (Scheltw.)" —
# nõiatähendus on allikas otseselt olemas. Teistes allikates hatt/at nõiatähendust ei kanna.
# Käivita: uv run python tabel_10.py (andmefaili asukoht: muutuja MASTER, vaikimisi töörepositooriumi paigutus)
import json
import re
from pathlib import Path

MASTER = Path("Katus-ALUSANDMED/json-all/AMT-Master_annotated.json")
PLACE = {"", "---", "???", "NULL"}

SOURCES = [
    ("Stahl 1637",    "Stahl-1637"),
    ("Gutslaff 1648", "Gutslaff-1648"),
    ("Göseken 1660",  "Göseken-1660"),
    ("Vestring 17XX", "Vestring-17XX"),
    ("Helle 1732",    "Helle-1732"),
    ("Hupel 1780",    "Hupel-1780-est-ger"),
]


def present(e, key):
    return isinstance(e.get(key), str) and e[key].strip() not in PLACE


rows = json.loads(MASTER.read_text(encoding="utf-8"))["AMT-Master"]
witch = [e for e in rows if e["Teema"] == "NÕID"]

# nõiatähendusega lisavormid väljastpoolt NÕID-teemat: {allikaveerg: [sõnakuju, ...]}
EXTRA = {"Vestring-17XX": ["Att"]}

print(f"NÕID-lemmasid massiivis kokku: {len(witch)}")
print()
print("| Sõnaraamat | Nimetusi | Sõnakujud * |")
print("|---|---|---|")
for name, key in SOURCES:
    cells = []
    for e in witch:
        if present(e, f"{key}-et"):
            # esinemuskordajad "[xN]" maha; lahtrisisesed variandid ühtlaselt kaldkriipsuga
            cell = re.sub(r"\s*\[x\d+\]", "", e[f"{key}-et"])
            parts = [p.strip() for p in re.split(r"[,;]", cell) if p.strip()]
            cells.append(" / ".join(parts))
    cells.extend(EXTRA.get(key, []))
    cells.sort(key=str.lower)
    print(f"| {name} | {len(cells)} | {', '.join(cells)} |")

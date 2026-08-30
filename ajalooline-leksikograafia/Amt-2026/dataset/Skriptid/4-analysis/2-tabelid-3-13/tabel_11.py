# Created: 2026-07-12 14-10-30
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
# Tabel 11. Naissoole osutavad sõnad varastes sõnastikes.
# Naissoole viitav = Sugu == "N". "Isikule viitavaid kokku" = allikas esindatud
# lemmade arv (kõik massiivi lemmad viitavad isikule; platshoidjad välja).
# Käivita: uv run python tabel_11.py (andmefaili asukoht: muutuja MASTER, vaikimisi töörepositooriumi paigutus)
import json
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

print("| Sõnaraamat | Naissoole viitavaid lemmasid | Isikule viitavaid kokku * |"
      " Naissoole viitavate osakaal % |")
print("|---|---|---|---|")
for name, key in SOURCES:
    att = [e for e in rows if present(e, f"{key}-et")]
    fem = sum(1 for e in att if e["Sugu"] == "N")
    pct = f"{fem / len(att) * 100:.1f}".replace(".", ",")
    print(f"| {name} | {fem} | {len(att)} | {pct} |")
print()
print("\\* Lemmade arv, mis on allikas esindatud.")

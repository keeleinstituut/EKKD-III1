# Created: 2026-07-12 14-10-30
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
# Tabel 13. Teemakategooriate ja sotsiaalsete kategooriate võrdlus.
# Ristsagedus Teema × Amt-Cat; K1/K2/K3 = puhas kategooria, Muu = kombineeritud + K0/K4.
# Reajärjestus nagu senises tabelis.
# Käivita repo juurest: uv run python scripts/tabelid-3-13/tabel_13.py
import json
from collections import Counter, defaultdict
from pathlib import Path

MASTER = Path("Katus-ALUSANDMED/json-all/AMT-Master_annotated.json")

ORDER = ["KÄSITÖÖ", "MORAAL_HÄLVE", "HALDUS_VÕIM", "TEENISTUS", "FEOD_MAA",
         "KIRIK_FUNKTSIOON", "KIRIK_VAIMULIK", "NÕID", "HARIDUS", "SUGU_REPRO",
         "MÜÜT", "OMADUS_SEOS_KUULUVUS"]

rows = json.loads(MASTER.read_text(encoding="utf-8"))["AMT-Master"]
total = len(rows)


def amt_group(e):
    a = e["Amt-Cat"].strip()
    return a if a in ("K1", "K2", "K3") else "Muu"


grid = defaultdict(Counter)
teema = Counter()
for e in rows:
    grid[e["Teema"]][amt_group(e)] += 1
    teema[e["Teema"]] += 1

cols = ("K1", "K2", "K3", "Muu")
print("| Teemakategooria | K1 | K2 | K3 | Muu | Kokku |")
print("|---|---|---|---|---|---|")
for k in ORDER:
    if not teema[k]:
        continue
    cells = " | ".join(str(grid[k][c]) for c in cols)
    print(f"| {k} | {cells} | {teema[k]} |")
tot = " | ".join(str(sum(grid[k][c] for k in grid)) for c in cols)
print(f"| Kokku | {tot} | {total} |")

# Created: 2026-07-12 14-10-30
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
# Tabel 4. L1-andmestiku koosseis esinemuste lõikes.
# Esinemus = mitmes sõnastikus lemma on esindatud (Cross-source count; platshoidjad
# --- / ??? / NULL ei loe esinemiseks). Rida "0" jäetakse välja, kui selliseid lemmasid pole.
# Käivita repo juurest: uv run python scripts/tabelid-3-13/tabel_04.py
import json
from collections import Counter
from pathlib import Path

MASTER = Path("Katus-ALUSANDMED/json-all/AMT-Master_annotated.json")

rows = json.loads(MASTER.read_text(encoding="utf-8"))["AMT-Master"]
total = len(rows)
csc = Counter(int(e["Cross-source count"]) for e in rows)

print("| Esineb N sõnastikus | Lemmasid | Osakaal |")
print("|---|---|---|")
for n in range(6, -1, -1):
    if csc[n] == 0:
        continue
    print(f"| {n} | {csc[n]} | {csc[n] / total * 100:.0f}% |")

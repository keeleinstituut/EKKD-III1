# Created: 2026-07-12 14-10-30
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
# Tabel 12. Naissoole viitavad sõnad teemade lõikes.
# Sugu == "N" lemmade jaotus Teema kaupa, kahanevas järjestuses; ainult esindatud teemad.
# Käivita: uv run python tabel_12.py (andmefaili asukoht: muutuja MASTER, vaikimisi töörepositooriumi paigutus)
import json
from collections import Counter
from pathlib import Path

MASTER = Path("Katus-ALUSANDMED/json-all/AMT-Master_annotated.json")

LABELS = {"OMADUS_SEOS_KUULUVUS": "OMADUS_SEOS_KUULUVUS / sugulus, omadused, etnilisus"}

rows = json.loads(MASTER.read_text(encoding="utf-8"))["AMT-Master"]
fem = Counter(e["Teema"] for e in rows if e["Sugu"] == "N")

print("| Teemakategooria | Naissoole viitavaid lemmasid |")
print("|---|---|")
for k, n in sorted(fem.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f"| {LABELS.get(k, k)} | {n} |")
print(f"| Kokku | {sum(fem.values())} |")

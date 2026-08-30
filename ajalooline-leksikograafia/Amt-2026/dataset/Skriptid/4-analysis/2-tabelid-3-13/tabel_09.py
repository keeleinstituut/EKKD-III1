# Created: 2026-07-12 14-10-30
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
# Tabel 9. Ametinimetuste ja sotsiaalsete rollide allikateülene püsivus
# keskmistatud allikateülese sageduse põhjal.
# KAS = keskmine Cross-source count teema lõikes; "kõigi lemmade keskmine" rida
# paigutub sorteeritud järjestusse oma väärtuse kohale.
# Käivita: uv run python tabel_09.py (andmefaili asukoht: muutuja MASTER, vaikimisi töörepositooriumi paigutus)
import json
from collections import defaultdict
from pathlib import Path

MASTER = Path("Katus-ALUSANDMED/json-all/AMT-Master_annotated.json")

rows = json.loads(MASTER.read_text(encoding="utf-8"))["AMT-Master"]
total = len(rows)

by_teema = defaultdict(list)
for e in rows:
    by_teema[e["Teema"]].append(int(e["Cross-source count"]))

table = [(sum(v) / len(v), len(v), k) for k, v in by_teema.items()]
overall = sum(int(e["Cross-source count"]) for e in rows) / total
table.append((overall, total, "kõigi lemmade keskmine"))
table.sort(key=lambda r: (-r[0], r[2]))

print("| Teema | Lemmasid | KAS  / (keskmistatud CSC) |")
print("|---|---|---|")
for avg, n, k in table:
    kas = f"{avg:.2f}".replace(".", ",")
    print(f"| {k} | {n} | {kas} |")

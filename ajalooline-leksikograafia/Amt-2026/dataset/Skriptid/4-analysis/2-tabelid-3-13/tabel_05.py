# Created: 2026-07-12 14-10-30
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
# Tabel 5. Lemmade esindatus K1–K3 kategooriate kaupa.
# Amt-Cat väärtused: puhtad K1/K2/K3, kombineeritud ("K1, K3" jne) ning K0/K4 (= Muu/määramata).
# Käivita: uv run python tabel_05.py (andmefaili asukoht: muutuja MASTER, vaikimisi töörepositooriumi paigutus)
import json
from collections import Counter
from pathlib import Path

MASTER = Path("Katus-ALUSANDMED/json-all/AMT-Master_annotated.json")

rows = json.loads(MASTER.read_text(encoding="utf-8"))["AMT-Master"]
total = len(rows)
cat = Counter(e["Amt-Cat"].strip() for e in rows)

DESC = {
    "K1": "Tuumikametid (nt arst, sepp)",
    "K2": "Juhuametid ja ametitaolised tegevused (nt abiline, eestkostja)",
    "K3": "Sotsiaalsed rollid (nt kuningas, talupoeg)",
}
COMBOS = [("K1, K2", "K1+K2"), ("K1, K3", "K1+K3"), ("K2, K3", "K2+K3"),
          ("K1, K2, K3", "K1+K2+K3")]

print("| Kategooria | Lemmade arv | Kirjeldus |")
print("|---|---|---|")
for k in ("K1", "K2", "K3"):
    print(f"| {k} | {cat[k]} | {DESC[k]} |")
for raw, label in COMBOS:
    if cat[raw]:
        print(f"| {label} | {cat[raw]} | Kombineeritud/piiripealsed |")
muu = {k: v for k, v in cat.items()
       if k not in ("K1", "K2", "K3") and k not in dict(COMBOS)}
if muu:
    muu_desc = ", ".join(f"{k} ({v})" for k, v in sorted(muu.items()))
    print(f"| Muu/määramata | {sum(muu.values())} | {muu_desc} |")
print(f"| Kokku | {total} |  |")

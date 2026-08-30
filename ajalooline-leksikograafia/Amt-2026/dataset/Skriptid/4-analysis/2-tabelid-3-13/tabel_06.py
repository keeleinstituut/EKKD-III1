# Created: 2026-07-12 14-10-30
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
# Tabel 6. L1-andmestiku jaotus semantiliste kategooriate lõikes.
# Sem-Cat = isikumärgendid ÜSi eeskujul (2026-07-12 fold-in). Mitmene märgendus:
# iga lemma loetakse igas oma kategoorias (nagu senises tabelis), seega veergude
# summa > lemmade arv. K1/K2/K3 = puhas Amt-Cat; Muu = kombineeritud + K0/K4.
# Osakaal = kategooria lemmade arv / kõik lemmad.
# Käivita: uv run python tabel_06.py (andmefaili asukoht: muutuja MASTER, vaikimisi töörepositooriumi paigutus)
import json
from collections import Counter, defaultdict
from pathlib import Path

MASTER = Path("Katus-ALUSANDMED/json-all/AMT-Master_annotated.json")

DESC = {
    "in_elukutse":    "Amet, põhitegevusala – kingsepp, kullassepp, aednik",
    "in_roll":        "Sotsiaalne roll, staatus, suhe – keiser, kodanik, vabadik",
    "in_omadus":      "Esilduv omadus / iseloomujoon – jutumees, ninakas",
    "in_tegija":      "Tegevusverbi agent – aitaja, eeljooksja",
    "inimene":        "Üldine / täpsustamata inimene – kirjatundja, kaassulane",
    "esitus_tiitel":  "Tiitlina kasutatav sõna – kuningas, krahv",
    "in_müt":         "Mütoloogiline olend – tont, lummaja",
    "in_sugulane":    "Sugulus- ja abielusuhe – abikaasa, peretütar",
    "in_rahvas":      "Etniline / paikkondlik päritolu – saks, maarahvas",
    "in_rahvas_keel": "Rahvas või keel – port",
}

rows = json.loads(MASTER.read_text(encoding="utf-8"))["AMT-Master"]
total = len(rows)


def amt_group(e):
    a = e["Amt-Cat"].strip()
    return a if a in ("K1", "K2", "K3") else "Muu"


grid = defaultdict(Counter)
multi = 0
for e in rows:
    tags = [t.strip() for t in e["Sem-Cat"].split(",") if t.strip()]
    if len(tags) > 1:
        multi += 1
    g = amt_group(e)
    for t in tags:
        grid[t][g] += 1

order = sorted(grid, key=lambda t: -sum(grid[t].values()))
cols = ("K1", "K2", "K3", "Muu")

print("| Semantiline kategooria | K1 | K2 | K3 | Muu | Kokku | Osa-kaal | Kirjeldus / näited |")
print("|---|---|---|---|---|---|---|---|")
for t in order:
    n = sum(grid[t].values())
    cells = " | ".join(str(grid[t][c]) for c in cols)
    print(f"| {t} | {cells} | {n} | {n / total * 100:.0f}% | {DESC.get(t, '')} |")
tot = {c: sum(grid[t][c] for t in grid) for c in cols}
grand = sum(tot.values())
print(f"| Kokku | {tot['K1']} | {tot['K2']} | {tot['K3']} | {tot['Muu']} | {grand} |  "
      f"| {total} lemmat + {grand - total} lisamärgendit ({multi} lemmal mitu semantilist kategooriat) |")

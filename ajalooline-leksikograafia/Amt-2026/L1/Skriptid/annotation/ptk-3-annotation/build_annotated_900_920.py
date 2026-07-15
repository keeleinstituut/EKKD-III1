# Created: 2026-06-12 15-54-33
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
"""Append hand-assigned Sem-Cat + DEF_en + DEF_et for source entries 900-920 (final 21).

Loads the existing annotated file (first 900 entries, already spliced and
verified verbatim), splices the three new labels (after "Amt-Cat") into source
entries 900-920, and writes the combined, complete 921-entry file losslessly.

Order added after "Amt-Cat":  Sem-Cat, DEF_en (English), DEF_et (distilled Estonian).
"""
import json

SRC = "Katus-ALUSANDMED/analyzed-tables/AMT-Master.json"
OUT = "Katus-DRAFTS/Ptk-3/AMT-Master_annotated.json"

NEW_KEYS = ("Sem-Cat", "DEF_en", "DEF_et")
NEW_ORDER_HEAD = ["Amt-Master-ID", "Amt-Cat", "Sem-Cat", "DEF_en", "DEF_et"]

# (expected Amt-Master-ID, Sem-Cat, DEF_en, DEF_et) — index-aligned to source rows 900..920.
ANN = [
 ("ülbe ja üleannetu inimene", "IN_OMADUS",
  "an insolent, impudent and mischievous person",
  "ülbe, jultunud ja üleannetu inimene (sks ein frecher, durchtriebener Mensch)"),
 ("üleastuja", "IN_OMADUS",
  "transgressor, lawbreaker, one who violates (a law/commandment)",
  "üleastuja, käsust/seadusest üleastuja, rikkuja (sks der Uebertreter)"),
 ("ülekaeja", "IN_ELUKUTSE, IN_ROLL:staatus",
  "overseer, supervisor; guardian",
  "ülevaataja, järelevaataja; eestkostja (sks Aufseher, Vormund)"),
 ("ülekuulaja", "IN_ELUKUTSE, IN_ROLL:staatus",
  "overseer, supervisor; guardian",
  "ülevaataja, järelevaataja; eestkostja (sks Aufseher, Vormund)"),
 ("ülekuulutaja", "IN_ELUKUTSE, IN_ROLL:staatus",
  "overseer, supervisor; guardian",
  "ülevaataja, järelevaataja; eestkostja (sks Aufseher, Vormund)"),
 ("ülem", "IN_ROLL:staatus",
  "superior, the foremost/highest one, lord",
  "ülem, ülemus, kõrgem isand (sks Oberherr, der Obere/Vornehmste)"),
 ("ülem kohtuisand", "IN_ELUKUTSE",
  "president of the court, presiding judge",
  "kohtu eesistuja, ülemkohtunik, president (sks Präsident, praeses)"),
 ("ülem proua", "IN_ROLL:staatus",
  "a noble lady, distinguished gentlewoman",
  "kõrgest seisusest proua, ülik daam (sks eine vornehme Dame)"),
 ("ülem rahva sees", "IN_ROLL:staatus",
  "the most eminent among the people, foremost of the folk",
  "rahva seas kõige tähtsam, esimene, ülik (sks der Vornehmste im Volk)"),
 ("ülempealik", "IN_ELUKUTSE",
  "supreme commander, chief captain, general",
  "ülempealik, väejuht, sõjapealik (sks Ober-Hauptmann, Heerführer)"),
 ("ülempreester", "IN_ELUKUTSE",
  "high priest",
  "ülempreester, kõrgem preester (sks Hohepriester)"),
 ("ülespidaja", "IN_OMADUS",
  "provider, sustainer, supporter (one who maintains)",
  "ülalpidaja, toitja, alalhoidja (sks der Erhalter)"),
 ("ülevaataja", "IN_ELUKUTSE, IN_ROLL:staatus",
  "overseer, supervisor; guardian",
  "ülevaataja, järelevaataja; eestkostja (sks Aufseher, Vormund)"),
 ("ülevõitja", "IN_OMADUS",
  "conqueror, victor, vanquisher",
  "ülevõitja, võitja, vallutaja (sks Überwinder, victor)"),
 ("ülla", "AINE",
  "saliva, drool (not a person-term; mislisted — see curator note)",
  "ila, sülg (pole isikunimetus; Ve eksitus) (sks der Geifer)"),
 ("ümberhulkuja", "IN_OMADUS",
  "vagrant, vagabond, tramp",
  "ümberhulkuja, hulkur, rändaja-uitaja (sks Landstreicher, Umläufer)"),
 ("ümberkaudne rahvas", "GRP_INIMENE",
  "the surrounding people, folk living round about, neighbours",
  "ümberkaudne rahvas, ümbruskonna elanikud (sks die Leute, die hier herum wohnen)"),
 ("ümberviija", "AGENT_TEGEVUS",
  "one who leads/carries around, conveyor (transfers over)",
  "ümberviija, ümberjuhtija, üleviija (sks herumführen)"),
 ("ümmardaja", "IN_ELUKUTSE",
  "maidservant, maid",
  "ümmardaja, teenija, teenijatüdruk (sks Magd, Dienstmagd)"),
 ("üte maa mees", "IN_RAHVAS",
  "compatriot, countryman, man of the same land",
  "ühemaamees, kaasmaalane, sama maa mees (sks Landsmann)"),
 ("ütleja", "IN_ELUKUTSE",
  "preacher, proclaimer",
  "jutlustaja, kuulutaja (sün jutleja) (sks Prediger)"),
]

START = 900


def splice(orig, semcat, def_en, def_et):
    new = {}
    for k, v in orig.items():          # preserve original order + values verbatim
        new[k] = v
        if k == "Amt-Cat":             # splice the three new labels right after Amt-Cat
            new["Sem-Cat"] = semcat
            new["DEF_en"] = def_en
            new["DEF_et"] = def_et
    for k, v in orig.items():          # integrity: original untouched
        assert new[k] == v
    assert list(new.keys())[:5] == NEW_ORDER_HEAD
    return new


def main():
    src = json.load(open(SRC, encoding="utf-8"))["AMT-Master"]
    existing = json.load(open(OUT, encoding="utf-8"))["AMT-Master"]

    assert len(ANN) == 21, len(ANN)
    assert len(existing) == START, f"expected {START} existing, got {len(existing)}"
    assert START + len(ANN) == len(src), f"would not reach {len(src)} total"

    # integrity: existing 0..899 are source verbatim + the three new keys
    for i in range(START):
        stripped = {k: v for k, v in existing[i].items() if k not in NEW_KEYS}
        assert stripped == src[i], f"existing entry {i} diverged from source"

    out_rows = list(existing)
    for j, (exp_id, semcat, def_en, def_et) in enumerate(ANN):
        i = START + j
        orig = src[i]
        assert orig["Amt-Master-ID"] == exp_id, f"index {i}: {orig['Amt-Master-ID']!r} != {exp_id!r}"
        out_rows.append(splice(orig, semcat, def_en, def_et))

    assert len(out_rows) == len(src), f"expected {len(src)} rows, got {len(out_rows)}"

    text = json.dumps({"AMT-Master": out_rows}, ensure_ascii=False, indent=2)
    text = text.replace("    },\n    {", "    },\n\n    {")   # blank line between entries
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"wrote {len(out_rows)} annotated entries -> {OUT} (added {len(ANN)}: {START}-{START+len(ANN)-1})")


if __name__ == "__main__":
    main()

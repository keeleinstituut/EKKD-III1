# Created: 2026-06-12 15-47-00
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
"""Append hand-assigned Sem-Cat + DEF_en + DEF_et for source entries 800-899.

Loads the existing annotated file (first 800 entries, already spliced and
verified verbatim), splices the three new labels (after "Amt-Cat") into source
entries 800-899, and writes the combined 900-entry file losslessly.

Order added after "Amt-Cat":  Sem-Cat, DEF_en (English), DEF_et (distilled Estonian).
"""
import json

SRC = "Katus-ALUSANDMED/analyzed-tables/AMT-Master.json"
OUT = "AMT-Master_annotated.json"  # in the working repo: the chapter-3 draft folder

NEW_KEYS = ("Sem-Cat", "DEF_en", "DEF_et")
NEW_ORDER_HEAD = ["Amt-Master-ID", "Amt-Cat", "Sem-Cat", "DEF_en", "DEF_et"]

# (expected Amt-Master-ID, Sem-Cat, DEF_en, DEF_et) — index-aligned to source rows 800..899.
ANN = [
 ("vahva naine", "IN_OMADUS", "heroine, valiant woman", "kangelanna, vahva naine (sks Heldin)"),
 ("vahva sõnamees", "IN_OMADUS", "eloquent speaker, debater, rhetorician", "sõnaosav rääkija, vaidleja, kõnemeister (sks Disputirer, Redekünstler, wohlberedt)"),
 ("vahva tööinimene", "IN_OMADUS", "a good, diligent, industrious worker", "tubli, töökas inimene (sks ein braver/fleissiger Arbeiter, arbeitsam)"),
 ("vaim", "IN_MÜT, IN_ELUKUTSE", "spirit, ghost; (Hupel) a foot-servant, hired laborer", "vaim, kummitus; (Hupelil) jalategija, sulane/tööline (sks Geist, Gespenst; Fußarbeiter)"),
 ("valelik", "IN_OMADUS", "liar", "valelik inimene, valetaja (sks Lügner)"),
 ("valetunnistaja", "IN_ROLL:staatus", "false witness, perjurer", "valetunnistaja, valevanduja (sks falscher Zeuge)"),
 ("valgevereline", "IN_OMADUS", "a 'white-blooded', pale/sickly person", "valgevereline, kahvatu/põdura inimene (sks ein weißblütiger Mensch)"),
 ("valitseja", "IN_ROLL:staatus", "ruler, regent, sovereign", "valitseja, riigivalitseja (sks Regent)"),
 ("valjud vanemad", "IN_ROLL:staatus", "strict masters, stern authorities/superiors", "ranged ülemused/isandad, valjud vanemad (sks strenge Herrschaften)"),
 ("vallalimees", "IN_ROLL:staatus", "a free, unmarried man; free peasant (cottar)", "vaba vallaline mees; vabadik (sks ein freier lediger Kerl, Freykerl)"),
 ("valuvõtja", "IN_ELUKUTSE", "sorcerer-healer, pain-charmer (one who draws off pain)", "valuvõtja, ravitseja-nõid, valu peletaja (sks Zauberer, Schmerzstiller)"),
 ("valvaja", "IN_ELUKUTSE", "watchman, guard, watcher", "valvaja, vaht, valvur (sks Wächter)"),
 ("vana agar tööinimene", "IN_OMADUS", "a very diligent, industrious person", "väga agar, töökas inimene (sks ein sehr fleissiger Mensch)"),
 ("vana juas", "IN_OMADUS", "an old fool", "vana narr, vana juhm (sks ein alter Narr)"),
 ("vana jätis", "IN_OMADUS", "an old, worthless, good-for-nothing person", "vana kõlvatu, väärtusetu inimene (sks ein alter nichtswürdiger Mensch)"),
 ("vana kämaras kõik", "IN_OMADUS", "an old, shrivelled, wrinkled person", "vana kortsus, kärbunud inimene (sks ein alt verschrumpelt Mensch)"),
 ("vana muld", "IN_OMADUS", "a completely worn-out, decrepit old person", "täiesti vana ja jõuetu, kulunud inimene (sks ein ganz abgelebter Mensch)"),
 ("vana mäda", "IN_OMADUS", "a completely worn-out, decrepit old person", "täiesti vana ja jõuetu, kulunud inimene (sks ein ganz abgelebter Mensch; süno vana muld)"),
 ("vana nõid", "IN_ELUKUTSE", "old witch, sorceress", "vana nõid, nõiamoor (sks Hexe)"),
 ("vana tudi", "IN", "an old man", "vana mees, vanataat (sks ein alter Mensch)"),
 ("vana tungus", "IN", "an old man", "vana mees, vanataat (sks ein alter Mensch)"),
 ("vanakõu", "IN_ROLL:sugulus", "forefather, ancestor, grandsire", "esiisa, esivanem, vanaisa; (Gö arvates ka 'pealekärkija') (sks Anherr)"),
 ("vananartsukaupmees", "IN_ELUKUTSE", "rag-and-bone man, old-clothes/secondhand dealer", "vanakraami- ehk nartsukaupmees, kaltsukaupmees (sks Trödler)"),
 ("vanaämm", "IN_ELUKUTSE", "midwife", "ämmaemand, vana-ämm (sks Hebamme)"),
 ("vande äravannutaja", "IN_ELUKUTSE", "exorcist, spirit-conjurer (one who conjures/banishes spirits)", "vaimude väljaajaja, lausuja, manaja (sks Geisterbeschwörer)"),
 ("vanem", "IN_ROLL:staatus", "superior, elder, authority; overseer", "ülem, vanem, ülemus, eestseisja (sks Oberherr, Oberkeit, der Aelteste)"),
 ("vang", "IN_ROLL:staatus", "prisoner, captive", "vang, kinnipeetu (sks ein Gefangener)"),
 ("vangihoidja", "IN_ELUKUTSE", "jailer, gaoler, prison-keeper", "vangihoidja, vangivalvur (sks Stockmeister)"),
 ("vangitu", "IN_ROLL:staatus", "prisoner, captive", "vang, kinnipeetu (sks ein Gefangener)"),
 ("vankrisepp", "IN_ELUKUTSE", "cartwright, wheelwright, wagon-maker", "vankrisepp, rattameister (sks Wagner, Rademacher)"),
 ("vapper suu", "IN_OMADUS", "chatterbox, prattler, idle talker", "lobiseja, latataja, suure suuga inimene (sks Plauderer)"),
 ("varanduse ülevaataja", "IN_ELUKUTSE", "overseer of property/wealth, steward, treasurer", "varahoidja, varanduse ülevaataja, laekur (saksakeelne vaste puudub)"),
 ("varas", "IN_OMADUS", "thief", "varas (sks Dieb)"),
 ("vardja", "IN_ELUKUTSE", "steward, master of the table, food-steward", "toidu- ehk söögimeister, kojaülem, varustaja (sks Speisemeister)"),
 ("varitseja", "IN_OMADUS", "one who lies in wait, lurker, ambusher", "varitseja, salakaela luuraja (sks ein Laurer)"),
 ("vasksepp", "IN_ELUKUTSE", "coppersmith", "vasksepp (sks Kupferschmied)"),
 ("vastaline", "IN_OMADUS", "adversary, opponent", "vastane, vastaline, vastutegija (sks Widersacher, entgegen, zuwider)"),
 ("vastane", "IN_OMADUS", "adversary, opponent", "vastane, vastaline (sks Widersacher)"),
 ("vastupanija", "IN_OMADUS", "rebel, one who resists/opposes", "vastupanija, mässaja (sks ein Rebell)"),
 ("vastuvõtja", "IN_OMADUS", "receiver of stolen goods, fence", "varastatud kauba vastuvõtja, salakauba varjaja (sks Hehler)"),
 ("vehtmeister", "IN_ELUKUTSE", "fencing-master", "vehklemismeister, vehklemisõpetaja (sks Fechtmeister)"),
 ("veisekarjane", "IN_ELUKUTSE", "cattle-herd, cowherd", "veisekarjus, karjane (sks Viehhirte)"),
 ("veisekarjus", "IN_ELUKUTSE", "cowherd", "veisekarjus, lehmakarjane (sks Kuhhirte)"),
 ("velsker", "IN_ELUKUTSE", "field-surgeon, barber-surgeon, army medic", "velsker, väliarst, habemeajaja-kirurg (sks Feldscher, Barbier)"),
 ("veltherr", "IN_ELUKUTSE", "army commander, general, field-marshal", "väejuht, sõjapealik, kindral (sks Feldherr)"),
 ("veomeister", "IN_ELUKUTSE", "weigh-master, public weigher", "kaalumeister, vaekoja meister (sks Wägemeister)"),
 ("vere tagaajaja", "IN_OMADUS", "blood-avenger, one who avenges kin", "veritasuja, verevõla kättemaksja (sks Bluträcher)"),
 ("vereajaja", "IN_OMADUS", "blood-avenger", "veritasuja, verevõla kättemaksja (sks Bluträcher)"),
 ("veskimees", "IN_ELUKUTSE", "miller", "mölder, veskimees (sks Müller)"),
 ("vihamees", "IN_OMADUS", "enemy, foe", "vihamees, vaenlane (sks Feind)"),
 ("viinaaednik", "IN_ELUKUTSE", "vine-gardener, vineyard keeper, vintner", "viinamarjaaednik, viinamäe harija (sks Weingärtner)"),
 ("viinaaiamees", "IN_ELUKUTSE", "vine-dresser, winegrower", "viinamarjakasvataja, viinaaiamees (sks Winzer)"),
 ("viinajooja", "IN_OMADUS", "wine-bibber, drunkard, heavy drinker", "viinajooja, joodik, viinasõber (sks weinsüchtig)"),
 ("viivitaja", "IN_OMADUS", "procrastinator, dawdler, one who delays", "viivitaja, kõhkleja, venitaja (sks ein Zauderer)"),
 ("viks tegija", "IN_OMADUS", "a skilful maker, a good master craftsman", "osav tegija, tubli meister (sks ein guter Meister)"),
 ("vilajas inimene", "IN_OMADUS", "a tall, lanky person (one shot up tall)", "pikaks kasvanud, sihvakas/longus inimene (sks ein lang aufgeschossener Mensch)"),
 ("vilepuhuja", "IN_ELUKUTSE", "piper, fifer, flute-player", "vilepuhuja, flöödimängija (sks Pfeifer, Flötenspieler)"),
 ("villakraasitaja", "IN_ELUKUTSE", "wool-carder, wool-beater", "villakraasija, villatöötleja (sks Wollschläger)"),
 ("villamees", "IN_ELUKUTSE", "wool-merchant, wool-dealer", "villakaupmees, villamees (sks Wollkrämer)"),
 ("vindunud inimene", "IN_OMADUS", "a thin, gaunt, scrawny person", "kõhn, kõhetu, vindunud inimene (sks ein hagerer, spitziger Mensch)"),
 ("vitsaroog", "IN_OMADUS", "one who deserves the rod/flogging; a rascal worthy of punishment", "vitsu väärt inimene, peksuväärt lurjus (sks einer der Ruthen verdient hat)"),
 ("voorimees", "IN_ELUKUTSE", "carter, carrier, wagoner", "voorimees, vedaja, küüdimees (sks Fuhrmann)"),
 ("voorkööper", "IN_ELUKUTSE", "forestaller, regrater, huckster (one who buys up to resell)", "ülesostja, vahekaupmees, harjusk (sks Vorkäufer, Höker)"),
 ("voorster", "IN_ROLL:staatus", "overseer, presider, warden, director", "eestseisja, ülevaataja, juhataja (sks Vorsteher)"),
 ("võla", "IN_ELUKUTSE", "sorceress, witch", "võlur, nõid, nõianaine (sks Zauberin, Unhold)"),
 ("võlaline", "IN_ROLL:staatus", "debtor, one in debt", "võlgnik, võlglane (sks Schuldner)"),
 ("võlglane", "IN_ROLL:staatus", "debtor", "võlgnik, võlglane (sks Schuldner)"),
 ("võltsija", "IN_OMADUS", "liar, falsifier, deceiver", "valetaja, võltsija, pettja (sks Lügner)"),
 ("võltstunnistaja", "IN_ROLL:staatus", "false witness, perjurer", "valetunnistaja, valevanduja (sks falscher Zeuge)"),
 ("võlu", "IN_ELUKUTSE", "sorcerer, sorceress, witch", "võlur, nõid, nõiamoor (sks Zauberer, Hexe)"),
 ("võtja", "AGENT_TEGEVUS", "taker, one who takes/receives", "võtja, vastuvõtja (tegijanimi) (sks ein Nehmer)"),
 ("võõra jumala teener", "IN_ROLL:ideol", "idolater, idol-worshipper, servant of a strange god", "ebajumala teenija, ebajumalakummardaja (sks Götzendiener)"),
 ("võõra usu mees", "IN_ROLL:ideol", "heretic, one of a foreign faith", "ketser, võõrausuline, usutaganeja (sks Ketzer)"),
 ("võõra vastuvõtja", "IN_OMADUS", "a hospitable person, one who takes in guests/strangers", "külalislahke inimene, võõraste vastuvõtja (sks gastfrey)"),
 ("võõramaa mees", "IN_RAHVAS", "foreigner, person from abroad", "välismaalane, võõramaalane (sks Ausländer)"),
 ("võõras", "IN_RAHVAS", "stranger, foreigner; guest", "võõras, võõramaalane; külaline (sks fremd, ein Gast)"),
 ("vägev sõjamees", "IN_OMADUS", "a mighty warrior, valiant hero", "vägev sõjamees, kangelane, vapper sõdalane (sks Held)"),
 ("vägimees", "IN_OMADUS, IN_MÜT", "a strong man, hero; (Hupel) a giant (mythological)", "vägimees, kangelane; (Hupelil) hiiglane (sks Held; Riese)"),
 ("väljakuulutaja", "IN_ELUKUTSE", "town-crier, herald, public announcer", "väljakuulutaja, hüüdja, kuulutaja (sks Ausrufer)"),
 ("väljamees", "IN", "traveller, wayfarer, one on a journey", "rändaja, teekäija, reisil olija (sks ein Reisender)"),
 ("väramees", "IN_OMADUS", "adulterer, fornicator, whoremonger", "abielurikkuja, hooraja (sks Ehebrecher, Hurer)"),
 ("väravatagune", "IN_RAHVAS", "a suburb-dweller, one living outside the (town) gate", "eeslinlane, värava-taguse elanik (sks ein Vorstädter)"),
 ("värval", "IN_ELUKUTSE", "dyer", "värval, värvija (sks Färber)"),
 ("vöörmünder", "IN_ROLL:staatus", "guardian (legal); church-warden, churchwarden-overseer", "eestkostja (vormünder); kirikuvöörmünder, kiriku eestseisja (sks Vormund, Kirchenvorsteher)"),
 ("vürst", "IN_ROLL:staatus", "prince, sovereign (duke)", "vürst, valitsejavürst, hertsog (sks Fürst, Herzog)"),
 ("walmis mees", "IN", "a fully grown, adult man", "täiskasvanud, täismees (sks ein vollkommen erwachsener Kerl)"),
 ("õige varas", "IN_OMADUS", "a real arch-thief, master-thief", "tõeline ülivaras, peavaras (sks ein Erzdieb)"),
 ("õigusemõistja", "IN_ELUKUTSE", "judge, magistrate, justice (bailiff)", "kohtunik, õigusemõistja, ametimees (sks Richter, Amtmann)"),
 ("õitsiline", "IN_ELUKUTSE", "night-herder, one who guards horses at night pasture", "õitsiline, öine hobusekarjus (õitsil olija) (saksakeelne vaste puudub)"),
 ("õlimees", "IN_ELUKUTSE", "oil-merchant, oil-dealer", "õlikaupmees, õlimüüja (sks Ölmann)"),
 ("õlletegija", "IN_ELUKUTSE", "brewer, beer-brewer", "õlletegija, õllepruulija (sks Bierbrauer)"),
 ("õnneandja", "IN_ELUKUTSE", "blessing-speaker, charm-blesser (one who pronounces blessings)", "õnnistaja, õnnesoovija, lausuja (õnne andja) (sks Segensprecher)"),
 ("õpetaja", "IN_ELUKUTSE", "teacher, instructor; (also) preacher, pastor", "õpetaja, õpetajameister; (ka) jutlustaja, kirikuõpetaja (sks Lehrer, Lehrmeister; Prediger)"),
 ("õpetaja", "IN_ELUKUTSE", "teacher, instructor; (also) preacher", "õpetaja, õpetajameister; (ka) jutlustaja (sks Lehrmeister, Prediger)"),
 ("õpilane", "IN_ROLL:staatus", "pupil, student, learner", "õpilane, õppija (sks Schüler, Lernender)"),
 ("õppija", "IN_ELUKUTSE, IN_ROLL:staatus", "(in older sources) teacher, preacher; (Vestring/Hupel) learner, pupil", "(vanemais allikais) õpetaja, jutlustaja; (Vestringil/Hupelil) õppija, õpilane (sks Lehrer, Prediger; Lernender, Schüler)"),
 ("äraandja", "IN_OMADUS", "traitor, betrayer", "äraandja, reetur (sks Verräter)"),
 ("äärne", "IN_OMADUS", "beggar (one who lies by the wayside)", "kerjus, teeäärne (tee ääres lamaja) (sks ein Bettler)"),
 ("öövaht", "IN_ELUKUTSE", "night-watchman", "öövaht, öine valvur (saksakeelne vaste puudub; vrd Nachtwächter)"),
 ("ükslane", "IN", "a single, solitary individual; one person on their own", "üksik inimene, ükslane (sks ein einzelner Mensch)"),
]

START = 800


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

    assert len(ANN) == 100, len(ANN)
    assert len(existing) == START, f"expected {START} existing, got {len(existing)}"

    # integrity: existing 0..799 are source verbatim + the three new keys
    for i in range(START):
        stripped = {k: v for k, v in existing[i].items() if k not in NEW_KEYS}
        assert stripped == src[i], f"existing entry {i} diverged from source"

    out_rows = list(existing)
    for j, (exp_id, semcat, def_en, def_et) in enumerate(ANN):
        i = START + j
        orig = src[i]
        assert orig["Amt-Master-ID"] == exp_id, f"index {i}: {orig['Amt-Master-ID']!r} != {exp_id!r}"
        out_rows.append(splice(orig, semcat, def_en, def_et))

    text = json.dumps({"AMT-Master": out_rows}, ensure_ascii=False, indent=2)
    text = text.replace("    },\n    {", "    },\n\n    {")   # blank line between entries
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"wrote {len(out_rows)} annotated entries -> {OUT} (added {len(ANN)}: {START}-{START+len(ANN)-1})")


if __name__ == "__main__":
    main()

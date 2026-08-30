# Created: 2026-06-12 15-34-54
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
"""Append hand-assigned Sem-Cat + DEF_en + DEF_et for source entries 700-799.

Loads the existing annotated file (first 700 entries, already spliced and
verified verbatim), splices the three new labels (after "Amt-Cat") into source
entries 700-799, and writes the combined 800-entry file losslessly.

Order added after "Amt-Cat":  Sem-Cat, DEF_en (English), DEF_et (distilled Estonian).
"""
import json

SRC = "Katus-ALUSANDMED/analyzed-tables/AMT-Master.json"
OUT = "AMT-Master_annotated.json"  # in the working repo: the chapter-3 draft folder

NEW_KEYS = ("Sem-Cat", "DEF_en", "DEF_et")
NEW_ORDER_HEAD = ["Amt-Master-ID", "Amt-Cat", "Sem-Cat", "DEF_en", "DEF_et"]

# (expected Amt-Master-ID, Sem-Cat, DEF_en, DEF_et) — index-aligned to source rows 700..799.
ANN = [
 ("talumees", "IN_ROLL:staatus", "peasant, peasant farmer; farmstead-holder, master of a farm", "talupoeg, talupidaja; pererahva peremees (sks Bauer, Bauerwirth, Gesinde-Wirth)"),
 ("talupoeg", "IN_ROLL:staatus", "peasant, farmer", "talupoeg, talupidaja (sks Bauer)"),
 ("talurahvas", "IN_ROLL:staatus, GRP_INIMENE", "peasant folk, peasantry; farm servants (collective)", "talurahvas, talupojad; pererahvas (sks Bauervolk, Gesindeleute)"),
 ("talutaja", "AGENT_TEGEVUS", "leader, guide (one who leads/conducts)", "juhataja, talutaja, teejuht (sks Führer, Leiter)"),
 ("tapja", "IN_OMADUS", "killer, slayer, manslayer; murderer", "tapja, mõrtsukas (sks Todschläger, Mörder)"),
 ("tapja mees", "IN_OMADUS", "killer, slayer, manslayer", "tapja, mõrtsukas (sks Todschläger)"),
 ("tapleja", "IN_OMADUS", "brawler, fighter; (also) fencer", "kakleja, võitleja, taplev inimene; (ka) vehkleja (sks Schläger, Fechter, Streiter)"),
 ("targon", "TAIM", "tarragon, the culinary herb (Artemisia dracunculus) — not a person", "estragon, maitsetaim (dracunculus esculentus); ei ole isik (sks Dragunkel)"),
 ("tark", "IN_ELUKUTSE", "soothsayer, fortune-teller, sorcerer; (also adj.) wise, clever", "ennustaja, ettekuulutaja, nõid; (ka omadussõnana) tark, arukas (sks Wahrsager, Zeichendeuter, Zauberer)"),
 ("tasuja", "IN_OMADUS", "requiter, avenger, one who repays", "tasuja, kättemaksja (sks Vergelter)"),
 ("teadmamees", "IN_ELUKUTSE", "cunning-man, wise man, one who knows (folk seer/healer); (Hupel) an acquainted person", "teadjamees, tark, oskaja (rahvatark/ravitseja); (Hupelil) tuttav inimene (sks 'was weiß', bekannter Mensch)"),
 ("teejuhataja", "IN_ELUKUTSE", "guide, one who shows the way", "teejuht, tee näitaja (sks Wegweiser; ld dux itineris)"),
 ("teekäija", "IN", "traveller, wayfarer, wanderer; pilgrim; stranger", "rändaja, teekäija, rändmees; (palve)rändur; võõras (sks Reisender, Wanderer, Pilger)"),
 ("teemees", "IN", "traveller, wayfarer", "rändaja, teekäija, teemees (sks Reisender, Wandersmann)"),
 ("teemees", "IN", "traveller, wayfarer", "rändaja, teekäija, teemees (sks Reisender, Wandersmann)"),
 ("teener", "IN_ELUKUTSE", "servant, attendant", "teener, sulane (sks Diener, Aufwärter)"),
 ("teener", "IN_ELUKUTSE", "servant", "teener (sks Diener)"),
 ("teenija", "IN_ELUKUTSE", "servant, one who serves", "teenija, teener (sks dienend)"),
 ("teeröövel", "IN_OMADUS", "highway robber, brigand; murderer", "teeröövel, maantereröövel; mõrtsukas (sks Straßenräuber, Mörder)"),
 ("teeäärne", "IN_OMADUS", "beggar (one who lies by the wayside); public whore; (also a term of abuse)", "kerjus (tee ääres lamaja); avalik hoor; (ka sõimusõna) (sks ein Bettler, eine öffentliche Hure)"),
 ("tegija", "IN_ELUKUTSE", "sorcerer, magic-worker; (also) worker, doer", "nõid, võlu (teotegija); (ka) tegija, töötegija (sks Zauberer; Arbeiter)"),
 ("teoline", "IN_ELUKUTSE", "corvée labourer, manor (estate) worker", "teoline, mõisatööline (teotegija) (sks Hofsarbeiter)"),
 ("teomees", "IN_ELUKUTSE", "corvée labourer, manor worker", "teomees, mõisatööline (sks Hofsarbeiter)"),
 ("teopoiss", "IN_ELUKUTSE", "manor work-boy (a youth able to do corvée labour)", "teopoiss, mõisatöö-poiss (juba tööks võimeline noormees) (sks ein Junge der arbeiten kann)"),
 ("teotaja", "IN_OMADUS", "slanderer, mocker, scoffer; blasphemer", "teotaja, pilkaja, laimaja (sks Lästerer, Spottvogel)"),
 ("tiisler", "IN_ELUKUTSE", "joiner, cabinetmaker", "tisler, puusepp (sks Tischler)"),
 ("timukas", "IN_ELUKUTSE", "executioner, hangman; bailiff", "timukas, hukkaja; kohtusulane (sks Henker, Scharfrichter, Büttel)"),
 ("toaselts", "IN_ROLL:staatus", "roommate, chamber-companion", "toakaaslane, toaseltsiline (sks Stubengeselle; ld contubernalis)"),
 ("tohter", "IN_ELUKUTSE", "doctor, physician", "arst, tohter (sks Doctor)"),
 ("toitja", "IN_OMADUS", "provider, nourisher, breadwinner", "toitja, ülalpidaja (sks Ernährer)"),
 ("tollipealine", "IN_ELUKUTSE", "toll-master, customs officer (who leased the tolls)", "tolliülem, tollnik (kes on tolli rentinud) (sks Zöllner)"),
 ("tolwan", "IN_OMADUS", "a stupid, foolish person; oaf", "rumal, lollakas inimene; tolvan (sks ein dummer läppischer Mensch)"),
 ("tont", "IN_MÜT, IN_ROLL:staatus", "ghost, spectre (mythological); (Göseken) a student", "tont, vaim, kummitus (müütiline); (Gösekenil) üliõpilane (sks Gespenst; Student)"),
 ("tooja", "AGENT_TEGEVUS", "bringer, one who brings something", "tooja, miski toob (tegijanimi) (sks einer der etwas bringt)"),
 ("toompapp", "IN_ELUKUTSE", "cathedral priest, dome-priest", "toomkiriku preester, toompapp (sks Dompfaffe)"),
 ("torikas", "IN_OMADUS", "a quarrelsome, contentious person; brawler", "tülinorija, riiakas inimene (sks ein Zanksüchtiger, Krakeeler)"),
 ("torine", "IN_OMADUS", "a quarrelsome, contentious person", "tülinorija, riiakas inimene (sks ein Zanksüchtiger, Krakeeler)"),
 ("trabant", "IN_ELUKUTSE", "bodyguard, halberdier, life-guard", "ihukaitsja, trabant (sks Trabant; ld satellis)"),
 ("tragun", "IN_ELUKUTSE", "dragoon, mounted soldier (Göseken); (Hupel conflates it with) tarragon, the herb", "dragoon, ratsaväelane (Gösekenil); (Hupelil segiaetud) estragon-maitsetaim (sks Dragoner; Dragunkel)"),
 ("treial", "IN_ELUKUTSE", "turner (lathe-worker)", "treial (sks Drechsler, Dreher)"),
 ("trossipoiss", "IN_ELUKUTSE", "baggage-boy, camp-servant boy", "trossipoiss, voori-/laagriteener, kannupoiss (sks Trossbube)"),
 ("trummipeksja", "IN_ELUKUTSE", "drummer", "trummilööja, trummar (sks Trommelschläger, Pauker)"),
 ("trööstja", "IN_OMADUS", "comforter, consoler", "trööstija, lohutaja (sks Tröster)"),
 ("trükja", "IN_ELUKUTSE", "printer, book-printer", "trükkija, raamatutrükkija (sks Buchdrucker)"),
 ("trükmeister", "IN_ELUKUTSE", "master printer, book-printer", "trükimeister, raamatutrükkija (sks Buchdrucker)"),
 ("tuleja", "AGENT_TEGEVUS", "newcomer, one who comes, arrival", "tulija, saabuja, tulnuk (tegijanimi) (sks ein Kommender, Ankömmling)"),
 ("tuleroog", "IN_ELUKUTSE", "a witch (one deemed worthy of burning at the stake); arch-sorceress (used as a term of abuse)", "nõid (tuleriidale väärt), nõiamoor; sõimusõnana (sks eine Hexe die des Feuers werth ist, Herz-Zauberin)"),
 ("tundja", "IN_ELUKUTSE", "soothsayer, fortune-teller", "ettekuulutaja, ennustaja, tark (sks Wahrsager)"),
 ("tunnistaja", "IN_ROLL:staatus", "witness (in court / legal sense)", "tunnistaja (kohtus) (sks Zeuge)"),
 ("tunnistusemees", "IN_ROLL:staatus", "witness", "tunnistaja, tunnistusemees (sks Zeuge)"),
 ("tunnistusmees", "IN_ROLL:staatus", "witness", "tunnistaja, tunnistusemees (sks Zeuge)"),
 ("turuhoor", "IN_OMADUS", "public whore, common harlot; arch-whore", "avalik hoor, turulits; ülihoor (sks eine öffentliche Hure, Erzhure)"),
 ("tuvimüüja", "IN_ELUKUTSE", "dove-seller, pigeon-dealer", "tuvimüüja, tuvikaupmees (sks Taubenkrämer)"),
 ("tõldsepp", "IN_ELUKUTSE", "coachmaker, cartwright, wheelwright", "tõldsepp, vankrimeister, rattameister (sks Wagner, Rademacher)"),
 ("tõlk", "IN_ELUKUTSE", "interpreter, translator", "tõlk, tõlkija (sks Dolmetscher)"),
 ("tõlkja", "IN_ELUKUTSE", "interpreter, translator", "tõlk, tõlkija (sks Dolmetscher)"),
 ("tõmbaja", "IN_OMADUS", "thief, thievish person", "varas, näppaja, vargalik inimene (sks Dieb, diebischer Mensch)"),
 ("tõmbaja inimene", "IN_OMADUS", "thief, thievish person", "varas, vargalik inimene (sks ein diebischer Mensch)"),
 ("tõrutegija", "IN_OMADUS", "ringleader, instigator (of a mob/disturbance)", "eestvedaja, õhutaja, märatsejate juht (sks Rädelführer)"),
 ("tähetundja", "IN_ELUKUTSE", "astronomer, stargazer", "astronoom, tähetundja, tähevaatleja (sks Sternseher)"),
 ("täiskasvanud mees", "IN", "a grown, adult man", "täiskasvanud mees (sks ein erwachsener Kerl)"),
 ("täkk", "LOOM", "stallion, male horse — not a person", "täkk, isane hobune; ei ole isik (sks Hengst)"),
 ("tölner", "IN_ELUKUTSE", "toll-collector, customs officer", "tölner, tollikoguja, müütnik (sks Zöllner)"),
 ("töö orjane", "IN_ROLL:staatus", "slave, bondsman (work-slave)", "(töö)ori, orjastatud inimene (sks Sklav)"),
 ("töömees", "IN_ELUKUTSE", "workman, labourer (town day-labourer)", "töömees, (linna)tööline, päeviline (sks Arbeitskerl, Arbeiter)"),
 ("tööorjane", "IN_ROLL:staatus", "slave, bondsman", "(töö)ori, orjastatud inimene (sks Sklav)"),
 ("töötegija", "IN_ELUKUTSE", "worker, labourer; (also adj.) industrious", "töötegija, tööline; (ka omadussõnana) töökas (sks Werkmann, Arbeiter; arbeitsam)"),
 ("tüdruk", "IN_ELUKUTSE", "maidservant, serving-girl; (also) girl, maiden", "teenijatüdruk, ümmardaja; (ka) tüdruk, neiu (sks Dienstmagd, Magd)"),
 ("tülitseja", "IN_OMADUS", "a quarrelsome, contentious person", "tülitseja, riiakas inimene (sks unfriedsam)"),
 ("tüma saks", "IN_OMADUS", "a stupid, simple-minded fellow", "rumal, lihtsameelne mees (siin 'saks' = Kerl, mitte sakslane) (sks ein dummer, einfältiger Kerl/Tropf)"),
 ("tündersepp", "IN_ELUKUTSE", "cooper, barrel-maker", "tündersepp, pütisepp (sks Böttcher, Tonnenmacher)"),
 ("tündrivitsutaja", "IN_ELUKUTSE", "cooper (one who hoops barrels)", "tündersepp, vaadivitsutaja (sks Fassbinder)"),
 ("tüürimees", "IN_ELUKUTSE", "helmsman, steersman", "tüürimees, roolimees (sks Steuermann)"),
 ("tüürimees", "IN_ELUKUTSE", "helmsman, steersman", "tüürimees, roolimees (sks Steuermann)"),
 ("uhtuja", "IN_ELUKUTSE", "fuller (cloth-fuller, walker)", "vanutaja, riidevanutaja (sks Walker)"),
 ("uksehoidja", "IN_ELUKUTSE", "doorkeeper, porter", "uksehoidja, väravavaht (sks Türhüter)"),
 ("uksehoidja", "IN_ELUKUTSE", "doorkeeper, porter", "uksehoidja, väravavaht (sks Türhüter)"),
 ("umbleja", "IN_ELUKUTSE", "tailor", "õmbleja, rätsep (sks Schneider)"),
 ("unistaja", "IN_OMADUS", "dreamer, day-dreamer", "unistaja, unelev inimene (sks Träumer)"),
 ("usklik", "IN_ROLL:ideol", "believer, the faithful, devout person", "usklik, usumees (sks Gläubiger)"),
 ("uue usu ning õpetuse tooja", "IN_ROLL:ideol", "innovator, bringer of a new faith and teaching; religious reformer/upstart", "uue usu ja õpetuse tooja, uuendaja (sks Neuling, Novator)"),
 ("vaataja", "AGENT_TEGEVUS", "spectator, onlooker, viewer", "vaataja, pealtvaataja (sks Zuschauer)"),
 ("vabadik, vabandetu", "IN_ROLL:staatus", "freed peasant, freedman; free peasant (not bound to corvée)", "vabadik, vabakslastu; vaba talupoeg (sks ein Freygelassener, Freybauer)"),
 ("vabandetu", "IN_ROLL:staatus", "freedman, free peasant", "vabandetu, vabakslastu, vaba talupoeg (sks ein Freygelassener, Freybauer)"),
 ("vabandik", "IN_ROLL:staatus", "freedman, free peasant", "vabandik, vabakslastu, vaba talupoeg (sks ein Freygelassener, Freybauer)"),
 ("vabatmees", "IN_ROLL:staatus", "free man; landless lodger-labourer (Lostreiber); peasant doing little manor work", "vabamees; vabadik, pop (Lostreiber); vähe mõisatööd tegev talupoeg (sks Freykerl, Lostreiber)"),
 ("vabatnaine", "IN_ROLL:staatus", "free woman; female landless lodger-labourer", "vabanaine; naissoost vabadik (sks Lostreiberin, Freyweib)"),
 ("vader", "IN_ROLL:sugulus", "godfather, godparent (sponsor at baptism)", "vader, ristiisa (ristimise vaderiks olija) (sks Gevatter; ld susceptor)"),
 ("vaekojaisand", "IN_ELUKUTSE", "public weigher, master of the weigh-house", "vaekojaisand, kaaluja (sks Wäger)"),
 ("vaekojamees", "IN_ELUKUTSE", "weigher (weigh-house man)", "vaekojamees, kaaluja (sks Wägekerl)"),
 ("vaenlane", "IN_OMADUS", "enemy, foe, adversary", "vaenlane, vihamees, vastane (sks Feind)"),
 ("vaevaja", "IN_OMADUS", "tormentor, tyrant, one who torments", "vaevaja, piinaja (sks Peiniger)"),
 ("vahemees", "IN_ELUKUTSE", "mediator, intermediary, go-between; advocate, spokesman", "vahemees, vahendaja, vahetalitaja; eestkostja (sks Mittler, Vorsprecher)"),
 ("vahetaja", "IN_ELUKUTSE", "money-changer", "(raha)vahetaja (sks Wechsler, Geldwechsler)"),
 ("vahimees", "IN_ELUKUTSE", "watchman, guard, sentry", "vahimees, valvur, vahisõdur (sks Wächter, Schildwache)"),
 ("vahisoldat", "IN_ELUKUTSE", "sentry, guard-soldier", "vahisoldat, vahisõdur (sks Schildwache)"),
 ("vaht", "IN_ELUKUTSE", "watchman, guard (also doorkeeper)", "vaht, valvur (ka uksevaht) (sks Wächter)"),
 ("vahva kõnemees", "IN_OMADUS", "a good, powerful orator; eloquent speaker", "hea, vägev kõnemees; sõnaosav rääkija (sks ein guter/starker Redner)"),
 ("vahva mees", "IN_OMADUS", "a brave, valiant man; hero; manly fellow", "vahva, vapper mees; kangelane; tubli mees (sks Held; ein braver Kerl)"),
 ("vahva mees rääkima", "IN_OMADUS", "a skilled, eloquent orator", "osav, hea kõnemees; sõnaosav rääkija (sks wohlberedt; ld facundus)"),
]

START = 700


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

    # integrity: existing 0..699 are source verbatim + the three new keys
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

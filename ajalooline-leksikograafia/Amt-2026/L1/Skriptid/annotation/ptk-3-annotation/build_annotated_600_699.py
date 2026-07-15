# Created: 2026-06-12 15-23-14
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
"""Append hand-assigned Sem-Cat + DEF_en + DEF_et for source entries 600-699.

Loads the existing annotated file (first 600 entries, already spliced and
verified verbatim), splices the three new labels (after "Amt-Cat") into source
entries 600-699, and writes the combined 700-entry file losslessly.

Order added after "Amt-Cat":  Sem-Cat, DEF_en (English), DEF_et (distilled Estonian).
"""
import json

SRC = "Katus-ALUSANDMED/analyzed-tables/AMT-Master.json"
OUT = "Katus-DRAFTS/Ptk-3/AMT-Master_annotated.json"

NEW_KEYS = ("Sem-Cat", "DEF_en", "DEF_et")
NEW_ORDER_HEAD = ["Amt-Master-ID", "Amt-Cat", "Sem-Cat", "DEF_en", "DEF_et"]

# (expected Amt-Master-ID, Sem-Cat, DEF_en, DEF_et) — index-aligned to source rows 600..699.
ANN = [
 ("riidepesija", "IN_ELUKUTSE", "launderer, washerwoman (clothes-washer); no gloss in source", "rõivapesija, pesunaine (allikagloss puudub)"),
 ("riisimees", "IN_MÜT", "giant (mythological)", "(müütiline) hiiglane (sks Riese)"),
 ("riisuja", "IN_OMADUS", "robber, plunderer, marauder", "röövel, rüüstaja, riisuja (sks Räuber, Beutemacher)"),
 ("rittmeister", "IN_ELUKUTSE", "cavalry captain (Rittmeister), cavalry officer (rank ~ captain)", "ratsaväe rittmeister, kaptenile vastav ratsaväeohvitser (sks Rittmeister)"),
 ("roa üleskandja", "IN_ELUKUTSE", "food-server, steward who serves dishes (court carver/server)", "toidu ettekandja, lauaülem (sks Truchsess, Kredenzer; ld dapifer)"),
 ("roalõikaja", "IN_ELUKUTSE", "carver (one who carves and serves meat at table)", "(liha)lõikaja lauas, ettelõikaja (sks Vorschneider; ld structor)"),
 ("roameister", "IN_ELUKUTSE", "pantry and cellar steward (master of provisions)", "toidukambri ja veinikeldri ülem, varahoidja (sks Speisemeister; ld promus)"),
 ("rootsik", "IN_RAHVAS", "Swedish woman; half-German; a woman dressed in (non-peasant) German clothes", "rootsi naine; poolsakslane; saksa moodi riietuv (mittetalupoeglik) naisterahvas (sks schwedische Weibsperson)"),
 ("ropsija", "IN_OMADUS", "brawler, one who hits/thrashes", "peksja, kakleja, ropsija (sks Schläger)"),
 ("ruunaja", "IN_ELUKUTSE", "horse-gelder, one who castrates horses", "hobuste ruunaja, kohitseja (sks einer der die Pferde wallachet)"),
 ("rõõmuandja", "IN_OMADUS", "one who brings joy, gladdener, cheerer", "rõõmu toov inimene, rõõmustaja (sks Fröhlichmacher)"),
 ("rõõmukuulutaja", "IN_ELUKUTSE", "evangelist, proclaimer of glad tidings (the gospel)", "rõõmusõnumi e evangeeliumi kuulutaja, evangelist (sks Evangelist)"),
 ("rätsep", "IN_ELUKUTSE", "tailor", "rätsep (sks Schneider)"),
 ("rätsepasulane", "IN_ELUKUTSE", "tailor's journeyman/apprentice", "rätsepasell, rätsepa sulane (sks Schneiderknecht)"),
 ("röökja", "IN_OMADUS", "shouter, bawler, crier", "röökija, kisendaja, karjuja (sks Schreier)"),
 ("röövel", "IN_OMADUS", "robber, brigand; murderer", "röövel, (tee)röövel; mõrtsukas (sks Räuber, Straßenräuber, Mörder)"),
 ("rüütel", "IN_ELUKUTSE", "rider, horseman, mounted soldier, cavalryman (later: knight)", "ratsanik, ratsasõdur; (hiljem) rüütel (sks Reiter)"),
 ("saadik", "IN_ELUKUTSE", "guide, escort, companion (also procurer/pander; forwarder)", "teejuht, saatja, kaaslane; ka kupeldaja (sks Geleitsmann, Führer, Begleiter; Kuppler)"),
 ("saatja", "IN_ELUKUTSE", "guide, escort, companion (incl. funeral escort)", "saatja, teejuht, kaaslane (ka surnusaatja) (sks Führer, Begleiter)"),
 ("sadulsepp", "IN_ELUKUTSE", "saddler, saddle-maker", "sadulsepp (sks Sattler)"),
 ("saesepp", "IN_ELUKUTSE", "saw-smith, maker of saws", "saesepp, saagimeister (sks Sägenschmied; ld serrarius)"),
 ("saia kitsäja", "???", "unclear: 'sai' = white/wheat bread, so perhaps a bread-baker/seller, but 'kitsäja' is unresolved; no gloss in source [uncertain]", "ebaselge: 'sai' viitab saiale, vahest saiaküpsetaja v -müüja, kuid 'kitsäja' jääb lahtiseks; allikagloss puudub [ebakindel]"),
 ("saks", "IN_RAHVAS", "a German", "sakslane (sks Deutscher)"),
 ("saksatüdruk", "IN_ELUKUTSE", "lady's maid, chambermaid", "toatüdruk, kammerneitsi (sks Zofe, Folgemagd)"),
 ("salaja nõuandja", "IN_ELUKUTSE", "privy councillor, confidential adviser", "salanõunik, geheimrat, salajane nõuandja (sks Geheimer Rat)"),
 ("salakoi", "IN_OMADUS", "a very secretive, tight-lipped person (jocular: 'secret moth')", "väga vaikiv, oma asju varjav inimene; naljatlev 'salakoi' (sks sehr verschwiegener Mensch)"),
 ("sant", "IN_OMADUS", "beggar; (adj.) poor, wretched, lowly", "kerjus; (omadussõnana) vaene, vilets, kehv (sks Bettler; arm, schlecht, gering)"),
 ("saunaline", "IN_ROLL:staatus", "cottager, landless peasant living in a bath-house/sauna hut", "saunik, maata talupoeg, kes elab saunas (sks Badstüber)"),
 ("saunamees", "IN_ROLL:staatus", "cottager, man living in a bath-house/sauna hut", "saunamees, saunas elav (maata) mees (sks Badstüber)"),
 ("sea karjane", "IN_ELUKUTSE", "swineherd", "seakarjus, sigade karjane (sks Schweinehirte)"),
 ("seadusandja", "IN_ROLL:staatus", "lawgiver, legislator", "seadusandja, seaduste kehtestaja (sks Gesetzgeber)"),
 ("seakarjus", "IN_ELUKUTSE", "swineherd", "seakarjus (sks Schweinehirte)"),
 ("seapoiss", "IN_ELUKUTSE", "swineherd (boy)", "seakarjus, seapoiss (sks Schweinehirte)"),
 ("seemisker", "IN_ELUKUTSE", "tawer, white-tanner (chamois-leather dresser)", "(valge)parkal, säämisknahaparkal (sks Weißgerber; ld alutarius)"),
 ("seletaja", "IN_ELUKUTSE", "interpreter, translator; expounder", "tõlk, tõlgendaja, seletaja (sks Dolmetscher, Ausleger)"),
 ("selge vana muld", "IN_OMADUS", "a frail, powerless old person (idiom: 'plain old earth')", "vana jõuetu, rauk (kõnekäänuline 'selge vana muld') (sks ein alter Ohnmächtiger)"),
 ("seltsimees", "IN_ROLL:staatus", "companion, comrade; business partner", "kaaslane, seltsiline; äripartner (sks Gefährte, Compagnon, Geselle)"),
 ("sepiline", "IN_ELUKUTSE", "smithy hand/helper; or one who has work done at the smithy", "sepapoiss, sepa abiline; v sepikojas tööd laskev inimene (sks einer der in der Schmiede arbeiten lässt)"),
 ("sepp", "IN_ELUKUTSE", "smith; craftsman, artisan", "sepp; käsitööline (sks Schmied; Handwerksmann)"),
 ("sigur", "IN_ELUKUTSE", "swineherd", "seakarjus, sigur (sks Schweinehirte)"),
 ("silla vanemad", "IN_ROLL:staatus, GRP_INIMENE", "the magistrates/elders of the rural court of order (Ordnungsgericht / sillakohus)", "sillakohtu (maakorrakohtu) vanemad, kohtumõistjad (sks das Ordnungsgericht)"),
 ("silmakirjutaja", "IN_ELUKUTSE", "illusionist, conjurer, eye-deceiver", "silmamoondaja, mustkunstnik, petukunstnik (sks Augenblender)"),
 ("silmamoondaja", "IN_ELUKUTSE", "illusionist, conjurer, eye-deceiver", "silmamoondaja, mustkunstnik (sks Augenverblender)"),
 ("silmapetja", "IN_ELUKUTSE", "juggler, conjurer, illusionist", "silmamoondaja, mustkunstnik, pettekunstnik (sks Gaukler)"),
 ("silmapistaja", "IN_ELUKUTSE", "illusionist, conjurer, eye-deceiver", "silmamoondaja, mustkunstnik (sks Augenverblender)"),
 ("soldan", "IN_ELUKUTSE", "soldier", "soldat, sõdur (sks Soldat)"),
 ("soldat", "IN_ELUKUTSE", "soldier", "soldat, sõdur (sks Soldat)"),
 ("soolapuhuja", "IN_ELUKUTSE", "salt-blower; folk sorcerer/healer (conjures by blowing on salt)", "soolapuhuja, nõid, ravitseja (soola peale puhuja) (sks Salzbläser, Zauberer)"),
 ("suiline", "IN_ELUKUTSE", "seasonal summer labourer, summer farmhand", "suiline, suvine palgatööline (sks Sommerarbeiter, Sommerling)"),
 ("sulane", "IN_ELUKUTSE", "farmhand, manservant, servant (Göseken also 'shopkeeper, stall-keeper')", "sulane, teener, talisulane; Gösekenil ka 'kaupmees, poepidaja' (sks Knecht, Diener)"),
 ("sundija", "IN_ELUKUTSE", "judge; bailiff (Vogt), criminal judge; (also) compeller, driver", "kohtunik, kohtufoogt; ka sundija, käskija (sks Richter, Gerichtsvogt)"),
 ("superdent", "IN_ELUKUTSE", "superintendent (Lutheran church official, ≈ bishop)", "superintendent (luterliku kiriku kõrgem vaimulik) (sks Superintendent)"),
 ("surnuohver", "SÜNDMUS", "offering/sacrifice for the dead; requiem, funeral rite (not a person)", "surnuohver, hingepalve, matuseohver (sündmus/talitus, mitte isik) (sks Totenopfer, Seelenmesse)"),
 ("surnusaatjad", "GRP_INIMENE", "funeral attendants, mourners, those who escort the dead (collective)", "surnusaatjad, matuselised, leinasaatjad (sks Leichenbegleiter)"),
 ("surnuvaras", "IN_ELUKUTSE", "gravedigger; corpse-bearer, undertaker", "hauakaevaja; surnumatja (sks Totengräber; ld sandapilarius)"),
 ("suured vägevad mehed", "IN_MÜT", "giants (mythological)", "mütoloogilised hiiglased (sks Riesen)"),
 ("suurpreester", "IN_ELUKUTSE", "high priest", "ülempreester, kõrge preester (sks hoher Priester)"),
 ("suutler", "IN_ELUKUTSE", "menial craftsman, low artisan; one who does dirty/rough work", "lihtkäsitööline, musta töö tegija (sks Sudler; ld cerdo)"),
 ("sõidetud hoor", "IN_OMADUS", "arch-whore, utter harlot", "ülihoor, läbi ja lõhki hoor (sks Erzhure)"),
 ("sõja peamees", "IN_ELUKUTSE", "war commander, military chief, colonel", "sõjaülem, väepealik, kolonel (sks Kriegsoberster)"),
 ("sõjamees", "IN_ELUKUTSE, IN_OMADUS", "soldier, warrior; (Hupel also) hero, valiant man", "sõjamees, sõdur, soldat; (Hupelil ka) kangelane, võitlushimuline (sks Soldat, Krieger; Held)"),
 ("sõjamees raudriidega", "IN_ELUKUTSE", "cuirassier, heavy cavalryman ('soldier in iron clothing')", "kürassir, raskeratsaväelane, raudrüüs sõjamees (sks Kürassier)"),
 ("sõjapealik", "IN_ELUKUTSE", "senior officer, high-ranking military commander", "sõjaväe ülemohvitser, väepealik (sks Oberoffizier)"),
 ("sõjasulane", "IN_ELUKUTSE", "soldier, man-at-arms, war-servant (mercenary foot-soldier)", "sõjasulane, palgasõdur, jalaväelane (sks Kriegsknecht)"),
 ("sõjategija", "IN_ELUKUTSE", "one who wages war, warrior/man of war (no gloss in source)", "sõjapidaja, sõjamees (allikagloss puudub)"),
 ("sõjavanem", "IN_ELUKUTSE", "war commander, military chief, colonel", "sõjaülem, väepealik, kolonel (sks Kriegsoberster)"),
 ("sõjaväe ülem peamees", "IN_ELUKUTSE", "supreme military commander, army leader, commander-in-chief", "sõjaväe ülemjuhataja, väepealik (sks Kriegsoberster, Heerführer)"),
 ("sõna", "IN_ELUKUTSE", "messenger, envoy, ambassador (gloss; lit. 'word')", "käskjalg, sõnumitooja, saadik (sks Botschafter)"),
 ("sõnapõlgaja", "IN_OMADUS", "disobedient person, one who scorns/defies orders", "sõnakuulmatu, allumatu inimene (sks ungehorsam)"),
 ("sõnatooja", "IN_ELUKUTSE", "messenger, courier", "sõnumitooja, käskjalg (sks Bote)"),
 ("sõnatooja", "IN_ELUKUTSE", "messenger, courier", "sõnumitooja, käskjalg (sks Bote)"),
 ("sõnavõtja", "IN_OMADUS", "obedient person, one who heeds/obeys", "sõnakuulelik, kuulekas inimene (sks gehorsam)"),
 ("sängikaeja", "IN_ELUKUTSE", "chamberlain (court official of the bedchamber/treasury)", "kammerhärra, õukonnaametnik (sks Kämmerer)"),
 ("södik", "IN_OMADUS", "glutton, voracious eater", "söödik, õgard, suur sööja (sks Fresser)"),
 ("sötik", "IN_ELUKUTSE", "executor, enforcer (one who carries out judgments/distraint)", "täideviija, kohtutäitur, nõudja (sks Exequirer)"),
 ("sötiku", "IN_ELUKUTSE, GRP_INIMENE", "soldiers in winter quarters, billeted soldiers", "talvekorteris olevad soldatid, talvituvad sõdurid (sks Soldaten in Winterquartier)"),
 ("sötiku istja", "IN_ELUKUTSE", "soldier in winter quarters, billeted soldier", "talvekorteris istuv soldat, talvituv sõdur (sks Soldaten in Winterquartier)"),
 ("sööbija", "IN_OMADUS", "gluttonous, voracious person; glutton", "õgija, ahne sööja, söödik (sks fressig)"),
 ("sööja", "IN_OMADUS", "eater, glutton, voracious eater (Helle's homonym 'Krebs' = the disease cancer, not a person)", "sööja, õgard, suur sööja (Hellel homonüüm 'vähk(tõbi)', mitte isik) (sks Fresser)"),
 ("söönud mees", "IN_OMADUS", "a strong, well-fed fellow; a man of strength", "tugev, hästi söönud (rammus) mees (sks ein Kerl der Kräfte hat)"),
 ("süüalune", "IN_ROLL:staatus", "the accused, the guilty party, defendant", "süüalune, kohtualune, süüdiolev pool (sks der Schuldige Teil)"),
 ("süüdlane", "IN_ROLL:staatus", "guilty person, culprit, the guilty party", "süüdlane, süüdiolev inimene (sks ein Schuldiger)"),
 ("süüdlik", "IN_ROLL:staatus", "defendant, the accused (in a lawsuit)", "kostja, süüalune, kohtualune (sks Beklagter)"),
 ("süütegija", "IN_ROLL:staatus", "defendant, the accused (lit. 'wrongdoer')", "kostja, süüalune (sks Beklagter)"),
 ("süütegija", "IN_ROLL:staatus", "defendant, the accused (lit. 'wrongdoer')", "kostja, süüalune (sks Beklagter)"),
 ("taatholder", "IN_ELUKUTSE", "governor, viceroy, regent (vicegerent)", "asehaldur, kuberner, asevalitseja (sks Statthalter)"),
 ("taevatähe seletaja", "IN_ELUKUTSE", "astrologer (star-interpreter)", "astroloog, tähetargutaja (sks Sterndeuter)"),
 ("taevatähe vaataja", "IN_ELUKUTSE", "astronomer, stargazer", "astronoom, tähevaatleja (sks Sternseher)"),
 ("tagaajaja", "AGENT_TEGEVUS", "pursuer, chaser; investigator, inquirer", "tagaajaja, jälitaja; juurdleja, uurija (sks Nachsetzer, Nachforscher)"),
 ("tagandismees", "IN_ROLL:staatus", "guarantor, surety (one who stands surety/vouches)", "käendaja, tagaja, käemees (sks Bürge)"),
 ("taganesmees", "IN_ROLL:staatus", "guarantor, surety", "käendaja, tagaja, käemees (sks Bürge)"),
 ("taldrikunolp", "IN_OMADUS", "parasite, sponger, freeloader (one who lives at others' expense)", "(toidu)parasiit, priileivasööja, teiste kulul elaja (sks Schmarotzer; ld gnatho)"),
 ("taldrikunolpija", "IN_OMADUS", "parasite, sponger; flatterer, sycophant, time-server", "(toidu)parasiit, priileivasööja; lipitseja, silmakirjatseja (sks Schmarotzer, Tellerlecker, Augendiener, Fuchsschwänzer)"),
 ("talguline", "IN_ROLL:staatus", "a participant/guest at a 'talkus' (communal work-bee with feasting)", "talguline, talgutel osaleja (talgu-külaline/-tööline) (sks Talkus-Gast)"),
 ("talgurahvas", "GRP_INIMENE", "the people/guests at a 'talkus' (communal work-bee); work-bee folk (collective)", "talgurahvas, talgulised (talgutel osalejad) (sks Talkus-Leute)"),
 ("talgus", "SÜNDMUS", "a 'talkus' — communal work-bee with a feast (an event, not a person)", "talgud, talgupidu, ühistöö ja sellega kaasnev söömaaeg (sündmus, mitte isik) (sks Bauernschmaus)"),
 ("talitaja", "IN_ELUKUTSE", "steward, manager, overseer; provider, arranger", "majapidaja, korraldaja, talitaja (sks Schaffer, Besteller)"),
 ("tallatav", "IN_OMADUS", "arch-whore, utter harlot (lit. 'one who is trodden')", "ülihoor, läbi ja lõhki hoor (sks Erzhure)"),
 ("tallipoiss", "IN_ELUKUTSE", "stable-boy, groom, ostler", "tallipoiss, hobusetalitaja, tallisulane (sks Stallknecht; ld equiso)"),
 ("tallmeister", "IN_ELUKUTSE", "master of the horse, equerry (head of the stables)", "tallmeister, tallide ülem (sks Stallmeister)"),
]

START = 600


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

    # integrity: existing 0..599 are source verbatim + the three new keys
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

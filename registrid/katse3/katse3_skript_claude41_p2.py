#Kood, registrite töörühma 3. katse tarvis. Eesmärk on Anthropicu mudelilt küsida vastuseid ainult ta enda treeningandmete põhjal. 
#Autor: Eleri Aedmaa
import os
import csv
import re
import time
from typing import List, Dict, Any
from anthropic import Anthropic

# --- Konfiguratsioon ---
# API klient
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("ANTHROPIC_API_KEY puudub. Palun määrake keskkonnamuutuja.")

client = Anthropic(api_key=api_key)
MODEL = "claude-opus-4-1-20250805"  # Claude 4.1 Opus
OUTPUT_FOLDER = "vastused"
FINAL_CSV = "vastused_koond.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Abifunktsioonid ---
def get_completion(prompt: str, user_msg: str) -> str:
    """Küsib Claude 4.1 Opuselt vastuse."""
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            temperature=0.1,
            messages=[
                {
                    "role": "user", 
                    "content": f"{prompt}\n\n{user_msg}"
                }
            ]
        )
        
        # Claude API tagastab vastuse sõnumite kujul
        return response.content[0].text if response.content else ""
        
    except Exception as e:
        print(f"Claude API päringu viga: {e}")
        return ""

def sanitize_filename(text: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', '_', text)[:100]

# --- PROMPT (treeningandmete-põhine) ---
def create_analysis_prompt(word: str) -> str:
    return f"""Oled eesti keele sõnaraamatu koostaja. Sinu ülesanne on analüüsida sõna „{word}" kasutust enda treeningandmetes ja otsustada, kas selle tähendustele tuleks lisada registrimärgend.

Vasta järgmistele küsimustele, tuginedes ainult enda treeningandmetele:

1. Nimeta sõna „{word}" kõik tähendused, mis su treeningandmetes esinevad. Ära erista alammõisteid erinevateks tähendusteks (näiteks „alukad" ei tähenda eraldi „aluspesu" ja „vanaema aluspükse", vaid üksnes „aluspesu").

2. Nimeta sõna „{word}" erinevate tähenduste arv.

3. Iga tähenduse juurde lisa, kas sõna on selles tähenduses sage, keskmine või vähene. Sagedusrühm vali võrdluses sõna teiste tähendustega.

4. Too iga tähenduse kohta enda treeningandmetest 5 näitelauset, kus „{word}" selles tähenduses esineb.

5. Otsusta sõna iga tähenduse kohta, kas seda kasutatakse pigem informaalsetes või neutraalsetes/formaalsetes tekstides? Kui sa ei oska eristust teha, sest see ei tule selgelt esile, siis ütle, et „ei kohaldu". Palun põhjenda oma valikut 5-10 lausega.

6. Ütle iga tähenduse juures, kui kindel sa oled oma vastuses selle kohta, kas tähendust kasutatakse informaalsetes või neutraalsetes/formaalsetes tekstides või „ei kohaldu". Vali, kas oled „väga kindel", „pigem kindel", „pigem ebakindel", „väga ebakindel".

7. Kui mõnda tähendust kasutatakse mingil viisil eripäraselt, siis vali sellele sobiv registrimärgend järgmistest:

- halvustav (vali siis, kui sõna selles tähenduses on kedagi või midagi laitev, mahategev, halvaks pidav, sõimav; näiteks ajuhälvik, debiilik, inimrämps)
- harv (vali siis, kui sõna selles tähenduses pole levinud, andmeid on vähe; näiteks ahvatama, mõistamisi, siinap). Märgend „harv" vali iga kord, kui tähendust leidub su treeningandmetes vähe.
- kõnekeelne (vali siis, kui sõna selles tähenduses on formaalsest keelekasutusest vabamasse registrisse kuuluv; näiteks igastahes, nokats, ära flippima)
- lastekeelne (vali siis, kui sõna selles tähenduses on lastekeelde kuuluv, sellele iseloomulik; näiteks jallu, kätu, nuku)
- luulekeelne (vali siis, kui sõna selles tähenduses on luulele iseloomulik, luulele omased, poeetilised väljendusvahendid; näiteks ehavalu, koidukuld, meeleheit)
- murdekeelne (vali siis, kui sõna selles tähenduses on murdes, murdekeeles kirjutatud, ei ole standardkeelne; näiteks hämmelgas, jõõrdlik, kidelema)
- rahvapärane (vali siis, kui sõna selles tähenduses on rahva seas levinud, aga pole ametlik termin, tihti näiteks kuude, taimede, loomade, haiguste, sugulaste nimetused; näiteks heinakuu, jooksva, männiseen)
- stiilitundlik (vali siis, kui sõna selles tähenduses on neutraalsest sõnastusest stiililiselt millegi poolest nähtavalt markeeritud, peene stiilitajuga, kõrgstiilsem; näiteks armastet, inimesepoeg, modern)
- vananenud (vali siis, kui sõna selles tähenduses on iganenud, aegunud; näiteks automobiil, aeroplaan, drogist)
- vulgaarne (vali siis, kui sõna selles tähenduses on labane, jäme, tahumatu; näiteks hoorapoeg, koinima, perse saatma)

Iga valiku korral põhjenda 5-10 lausega, miks just see märgend sobib. Igal informaalsel tähendusel peab olema vähemalt üks märgend. Kui sobib mitu, too mitu. Neutraalsele/formaalsele ja „ei kohaldu" tähendusele lisa märgend ainult siis, kui see tundub treeningandmete põhjal vajalik.

OLULINE: Pärast küsimustele vastamist anna oma vastused TÄPSELT järgmises struktureeritud formaadis parsimiseks. Kasuta erinevat eraldajat (§§§) iga tähenduse andmete vahel:

--- STRUKTUREERITUD VASTUS ALGAB ---
TÄHENDUSED: tähendus1§§§tähendus2§§§tähendus3
TÄHENDUSTE-ARV: 3
SAGEDUSED: tähendus1-sage§§§tähendus2-keskmine§§§tähendus3-vähene
NÄITED: tähendus1-laus1|laus2|laus3|laus4|laus5§§§tähendus2-laus1|laus2|laus3|laus4|laus5§§§tähendus3-laus1|laus2|laus3|laus4|laus5
REGISTRID: tähendus1-neutraalsetes-formaalsetes§§§tähendus2-informaalsetes§§§tähendus3-ei kohaldu
REGISTRI-PÕHJENDUSED: tähendus1-... 5–10 lauset ...§§§tähendus2-... 5–10 lauset ...§§§tähendus3-... 5–10 lauset ...
REGISTRI-KINDLUS: tähendus1-väga kindel§§§tähendus2-pigem kindel§§§tähendus3-pigem ebakindel
MÄRGENDID: tähendus1-ei-kohaldu§§§tähendus2-kõnekeelne,rahvapärane§§§tähendus3-harv
MÄRGENDITE-PÕHJENDUSED: tähendus1-ei-kohaldu§§§tähendus2-... 5–10 lauset ...§§§tähendus3-... 5–10 lauset ...
--- STRUKTUREERITUD VASTUS LÕPEB ---"""

# --- Parser ---
def parse_analysis_response(txt: str, word: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    try:
        m = re.search(r'--- STRUKTUREERITUD VASTUS ALGAB ---(.*?)--- STRUKTUREERITUD VASTUS LÕPEB ---',
                      txt, re.DOTALL)
        structured = m.group(1) if m else txt
        data: Dict[str, str] = {}
        for line in structured.splitlines():
            line = line.strip()
            if ':' in line and not line.startswith('http'):
                k, v = line.split(':', 1)
                data[k.strip()] = v.strip()

        # Tähendused
        meanings = [t.strip() for t in data.get('TÄHENDUSED', '').split('§§§') if t.strip()]

        # Sagedused
        freq = []
        for item in data.get('SAGEDUSED', '').split('§§§'):
            if item.strip():
                freq.append(item.split('-', 1)[1].strip() if '-' in item else item.strip())

        # Näited
        examples = []
        for item in data.get('NÄITED', '').split('§§§'):
            if item.strip():
                ex = item.split('-', 1)[1].strip() if '-' in item else item.strip()
                examples.append(ex.replace('|', ' | '))

        # Registrid
        registers = []
        for item in data.get('REGISTRID', '').split('§§§'):
            if item.strip():
                registers.append(item.split('-', 1)[1].strip() if '-' in item else item.strip())

        # Registri põhjendused
        reg_just = []
        for item in data.get('REGISTRI-PÕHJENDUSED', '').split('§§§'):
            if item.strip():
                reg_just.append(item.split('-', 1)[1].strip() if '-' in item else item.strip())

        # Registri kindlus
        reg_conf = []
        for item in data.get('REGISTRI-KINDLUS', '').split('§§§'):
            if item.strip():
                reg_conf.append(item.split('-', 1)[1].strip() if '-' in item else item.strip())

        # Märgendid
        tags = []
        for item in data.get('MÄRGENDID', '').split('§§§'):
            if item.strip():
                tags.append(item.split('-', 1)[1].strip() if '-' in item else item.strip())

        # Märgendite põhjendused
        tag_just = []
        for item in data.get('MÄRGENDITE-PÕHJENDUSED', '').split('§§§'):
            if item.strip():
                tag_just.append(item.split('-', 1)[1].strip() if '-' in item else item.strip())

        total_meanings = data.get('TÄHENDUSTE-ARV', str(len(meanings))) or str(len(meanings))

        for i, meaning in enumerate(meanings):
            fval = freq[i] if i < len(freq) else "ei määratletud"
            exval = examples[i] if i < len(examples) else "ei leitud"
            reg = registers[i] if i < len(registers) else "ei määratletud"
            rj = reg_just[i] if i < len(reg_just) else "ei määratletud"
            rc = reg_conf[i] if i < len(reg_conf) else "pigem ebakindel"
            tag_text = tags[i] if i < len(tags) else "ei-kohaldu"
            tj = tag_just[i] if i < len(tag_just) else "ei-kohaldu"

            tag_list = []
            if tag_text and tag_text != "ei-kohaldu":
                tag_list = [m.strip() for m in tag_text.split(',') if m.strip()]

            if not tag_list:
                results.append({
                    "Sõna": word,
                    "Tähenduse nr": i + 1,
                    "Tähendus": meaning,
                    "Tähenduste arv kokku": total_meanings,
                    "Sagedus": fval,
                    "Näited": exval,
                    "Tekstiregister": reg,
                    "Registri põhjendus": rj,
                    "Registri kindlus": rc,
                    "Registrimärk": "ei-kohaldu",
                    "Märgendi põhjendus": "ei-kohaldu"
                })
            else:
                for mtag in tag_list:
                    results.append({
                        "Sõna": word,
                        "Tähenduse nr": i + 1,
                        "Tähendus": meaning,
                        "Tähenduste arv kokku": total_meanings,
                        "Sagedus": fval,
                        "Näited": exval,
                        "Tekstiregister": reg,
                        "Registri põhjendus": rj,
                        "Registri kindlus": rc,
                        "Registrimärk": mtag,
                        "Märgendi põhjendus": (tj if tj != "ei-kohaldu" else "ei leitud")
                    })

            # Lühilog
            print(f"   Tähendus {i+1}: {meaning}")
            print(f"      Sagedus: {fval}")
            print(f"      Register: {reg}  ({rc})")
            print(f"      Reg.põhjendus: {rj[:120]}{'...' if len(rj) > 120 else ''}")
            print(f"      Märgend(id): {tag_text}")
            if tj != "ei-kohaldu":
                print(f"      Märgendi põhjendus: {tj[:120]}{'...' if len(tj) > 120 else ''}")

    except Exception as e:
        print(f"   Parsimise viga: {e}")
        results.append({
            "Sõna": word,
            "Tähenduse nr": 1,
            "Tähendus": "parsimise viga",
            "Tähenduste arv kokku": 0,
            "Sagedus": "ei määratletud",
            "Näited": "ei leitud",
            "Tekstiregister": "ei määratletud",
            "Registri põhjendus": "ei määratletud",
            "Registri kindlus": "pigem ebakindel",
            "Registrimärk": "ei-kohaldu",
            "Märgendi põhjendus": "ei-kohaldu"
        })

    if not results:
        results.append({
            "Sõna": word,
            "Tähenduse nr": 1,
            "Tähendus": "ei määratletud",
            "Tähenduste arv kokku": 0,
            "Sagedus": "ei määratletud",
            "Näited": "ei leitud",
            "Tekstiregister": "ei määratletud",
            "Registri põhjendus": "ei määratletud",
            "Registri kindlus": "pigem ebakindel",
            "Registrimärk": "ei-kohaldu",
            "Märgendi põhjendus": "ei-kohaldu"
        })
    return results

# --- Töötlemine (ilma kontekstifailideta) ---
def process_word_analysis(word: str):
    prompt = create_analysis_prompt(word)
    user_msg = f"Analüüsi sõna \"{word}\" ainult oma treeningandmete põhjal ja tagasta täpselt nõutud struktuur."

    try:
        reply = get_completion(prompt, user_msg)

        # logi ja salvesta toorvastus
        print("\n" + "="*80)
        print(f"MUDELI VASTUS sõnale '{word}':")
        print("="*80)
        print(reply)
        print("="*80)

        safe_word = sanitize_filename(word)
        out_path = os.path.join(OUTPUT_FOLDER, f"{safe_word}_analysis.txt")
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(reply)

        parsed = parse_analysis_response(reply, word)

        # lühikokkuvõte
        print(f"\nPARSITUD TULEMUS:")
        print(f"   Ridu kokku: {len(parsed)}")
        for r in parsed:
            print(f"   {r['Tähenduse nr']}. {r['Tähendus'][:60]}{'...' if len(r['Tähendus']) > 60 else ''}")
            print(f"      {r['Sagedus']} | {r['Tekstiregister']} ({r['Registri kindlus']}) | {r['Registrimärk']}")

        print(f"{word} — Analüüs lõpetatud\n")
        return parsed

    except Exception as e:
        print(f"Viga sõnaga {word}: {e}")
        return None

# --- Põhiprogramm ---
def main():
    all_rows: List[Dict[str, Any]] = []

    # Loeme sisendfaili (tab-eraldaja). Kui päis „Sõna", jäta vahele.
    with open("sisend.txt", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        first_row = next(reader, None)
        if first_row and len(first_row) == 1 and first_row[0].strip().lower() in ("sõna", "sona", "word"):
            words = [row[0].strip() for row in reader if row and row[0].strip()]
        else:
            words = []
            if first_row and len(first_row) >= 1 and first_row[0].strip():
                words.append(first_row[0].strip())
            for row in reader:
                if row and row[0].strip():
                    words.append(row[0].strip())

    for i, word in enumerate(words, 1):
        print(f"\n{'='*60}")
        print(f"ANALÜÜSIN ({i}/{len(words)}): '{word}'")
        print(f"{'='*60}")

        result = process_word_analysis(word)
        if result:
            all_rows.extend(result)
        else:
            print("Lisame tühja rea järjekorra säilitamiseks")
            all_rows.append({
                "Sõna": word,
                "Tähenduse nr": 1,
                "Tähendus": "töötlemata",
                "Tähenduste arv kokku": 0,
                "Sagedus": "ei saadaval",
                "Näited": "ei saadaval",
                "Tekstiregister": "ei määratletud",
                "Registri põhjendus": "ei saadaval",
                "Registri kindlus": "pigem ebakindel",
                "Registrimärk": "ei kohaldu",
                "Märgendi põhjendus": "ei saadaval"
            })

        time.sleep(0.5)

    # CSV
    fieldnames = [
        "Sõna", "Tähenduse nr", "Tähendus", "Tähenduste arv kokku",
        "Sagedus", "Näited", "Tekstiregister", "Registri põhjendus",
        "Registri kindlus", "Registrimärk", "Märgendi põhjendus"
    ]
    with open(FINAL_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";", quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\nLõplik fail salvestatud: {FINAL_CSV}")
    print(f"Kokku analüüsitud ridu: {len(all_rows)}")

    # Stat
    uniq = len(set(r["Sõna"] for r in all_rows))
    processed = len(set(r["Sõna"] for r in all_rows if r["Tähendus"] != "töötlemata"))
    unprocessed = uniq - processed
    avg_meanings = (len(all_rows) / uniq) if uniq > 0 else 0.0
    print(f"\nAnalüüsi statistika:")
    print(f"  Kokku sõnu: {uniq}")
    print(f"  Edukalt töödeldud sõnu: {processed}")
    print(f"  Töötlemata sõnu: {unprocessed}")
    print(f"  Keskmine tähendusi sõna kohta: {avg_meanings:.1f}")

if __name__ == "__main__":
    main()
#Kood, registrite töörühma 3. katse tarvis. Eesmärk on OpenAI mudelile kaasa anda korpusest andmed, mida ta analüüsima peab.
#Kui kontekst on liiga suur, siis see vektoriseeritakse.
#Autor: Eleri Aedmaa
import os
import csv
import openai
import pickle
import faiss
import tiktoken
import pandas as pd
import re
import time
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# --- Konfiguratsioon ---
client = openai.OpenAI()
MODEL = "gpt-4o"
EMBED_MODEL = SentenceTransformer("intfloat/multilingual-e5-base")
DATA_FOLDER = "contexts"
OUTPUT_FOLDER = "vastused"
FINAL_CSV = "vastused_koond.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs("vector_cache", exist_ok=True)

# --- Abi funktsioonid ---
def tokenize_length(text: str, model=MODEL):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def build_faiss_index(chunks: List[str]):
    passages = [f"passage: {chunk}" for chunk in chunks]
    embeddings = EMBED_MODEL.encode(passages, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def save_index(index, chunks, path):
    faiss.write_index(index, f"{path}.faiss")
    with open(f"{path}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

def load_index(path):
    index = faiss.read_index(f"{path}.faiss")
    with open(f"{path}_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def ensure_index_exists(word: str, full_lines: List[str]):
    index_path = os.path.join("vector_cache", word)
    if os.path.exists(index_path + ".faiss") and os.path.exists(index_path + "_chunks.pkl"):
        return load_index(index_path)
    index, _ = build_faiss_index(full_lines)
    save_index(index, full_lines, index_path)
    return index, full_lines

def get_relevant_chunks_max(query: str, chunks: List[str], index, max_k=None):
    query_vec = EMBED_MODEL.encode([f"query: {query}"], show_progress_bar=False)
    D, I = index.search(query_vec, len(chunks))
    sorted_chunks = [chunks[i] for i in I[0]]
    if max_k:
        return sorted_chunks[:max_k]
    return sorted_chunks

def get_completion(prompt: str, context: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": context}
        ],
        max_tokens=16000,
        temperature=0.1
    )
    return response.choices[0].message.content

def sanitize_filename(text):
    return re.sub(r'[<>:"/\\|?*]', '_', text)[:100]

# --- PROMPT (täpselt sinu sõnastus) ---
def create_analysis_prompt(word: str):
    return f"""Oled eesti keele sõnaraamatu koostaja. Sinu ülesanne on analüüsida sõna „{word}" kasutust etteantud tekstimaterjalis ja otsustada, kas selle tähendustele tuleks lisada registrimärgend.

Vasta järgmistele küsimustele, tuginedes ainult etteantud materjalile:

1. Nimeta sõna „{word}" kõik tähendused, mida etteantud tekstides näed. Ära erista alammõisteid erinevateks tähendusteks (näiteks „alukad" ei tähenda eraldi „aluspesu" ja „vanaema aluspükse", vaid üksnes „aluspesu").

2. Nimeta sõna „{word}" erinevate tähenduste arv.

3. Iga tähenduse juurde lisa, kas sõna on selles tähenduses sage, keskmine või vähene. Sagedusrühm vali võrdluses sõna teiste tähendustega.

4. Too iga tähenduse kohta etteantud materjalist 5 näitelauset, kus „{word}" selles tähenduses esineb.

5. Otsusta sõna iga tähenduse kohta, kas seda kasutatakse pigem informaalsetes või neutraalsetes/formaalsetes tekstides? Kui sa ei oska eristust teha, sest see ei tule selgelt esile, siis ütle, et „ei kohaldu". Palun põhjenda oma valikut 5-10 lausega.

6. Ütle iga tähenduse juures, kui kindel sa oled oma vastuses selle kohta, kas tähendust kasutatakse informaalsetes või neutraalsetes/formaalsetes tekstides või „ei kohaldu“. Vali, kas oled „väga kindel“, „pigem kindel“, „pigem ebakindel“, „väga ebakindel“.

7. Kui mõnda tähendust kasutatakse tekstides mingil viisil eripäraselt, siis vali sellele sobiv registrimärgend järgmistest:

- halvustav (vali siis, kui sõna selles tähenduses on kedagi või midagi laitev, mahategev, halvaks pidav, sõimav; näiteks ajuhälvik, debiilik, inimrämps)
- harv (vali siis, kui sõna selles tähenduses pole levinud, andmeid on vähe; näiteks ahvatama, mõistamisi, siinap). Märgend „harv" vali iga kord, kui tähendust leidub etteantud tekstimaterjalis vähe.
- kõnekeelne (vali siis, kui sõna selles tähenduses on formaalsest keelekasutusest vabamasse registrisse kuuluv; näiteks igastahes, nokats, ära flippima)
- lastekeelne (vali siis, kui sõna selles tähenduses on lastekeelde kuuluv, sellele iseloomulik; näiteks jallu, kätu, nuku)
- luulekeelne (vali siis, kui sõna selles tähenduses on luulele iseloomulik, luulele omased, poeetilised väljendusvahendid; näiteks ehavalu, koidukuld, meeleheit)
- murdekeelne (vali siis, kui sõna selles tähenduses on murdes, murdekeeles kirjutatud, ei ole standardkeelne; näiteks hämmelgas, jõõrdlik, kidelema)
- rahvapärane (vali siis, kui sõna selles tähenduses on rahva seas levinud, aga pole ametlik termin, tihti näiteks kuude, taimede, loomade, haiguste, sugulaste nimetused; näiteks heinakuu, jooksva, männiseen)
- stiilitundlik (vali siis, kui sõna selles tähenduses on neutraalsest sõnastusest stiililiselt millegi poolest nähtavalt markeeritud, peene stiilitajuga, kõrgstiilsem; näiteks armastet, inimesepoeg, modern)
- vananenud (vali siis, kui sõna selles tähenduses on iganenud, aegunud; näiteks automobiil, aeroplaan, drogist)
- vulgaarne (vali siis, kui sõna selles tähenduses on labane, jäme, tahumatu; näiteks hoorapoeg, koinima, perse saatma)

Iga valiku korral põhjenda 5-10 lausega, miks just see märgend sobib. Igal informaalsel tähendusel peab olema vähemalt üks märgend. Kui sobib mitu, too mitu. Neutraalsele/formaalsele ja „ei kohaldu“ tähendusele lisa märgend ainult siis, kui see tundub tekstimaterjali põhjal vajalik.

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

# --- Parandatud parsimise funktsioon ---
def parse_analysis_response(txt: str, word: str) -> List[Dict[str, Any]]:
    """
    Tagastab listi, kus iga element on üks tähendus koos kõigi andmetega
    """
    results = []
    try:
        structured_match = re.search(
            r'--- STRUKTUREERITUD VASTUS ALGAB ---(.*?)--- STRUKTUREERITUD VASTUS LÕPEB ---',
            txt, re.DOTALL
        )
        structured_text = structured_match.group(1) if structured_match else txt
        lines = structured_text.split('\n')
        data = {}
        for line in lines:
            line = line.strip()
            if ':' in line and not line.startswith('http'):
                key, value = line.split(':', 1)
                data[key.strip()] = value.strip()

        # Tähendused
        meanings = []
        if 'TÄHENDUSED' in data:
            meanings = [t.strip() for t in data['TÄHENDUSED'].split('§§§') if t.strip()]

        # Sagedused
        freq = []
        if 'SAGEDUSED' in data:
            for item in data['SAGEDUSED'].split('§§§'):
                if '-' in item:
                    freq.append(item.split('-', 1)[1].strip())
                else:
                    freq.append(item.strip())

        # Näited (5 lauset, kuid võtame kõik, mis antud)
        examples = []
        if 'NÄITED' in data:
            for item in data['NÄITED'].split('§§§'):
                if '-' in item:
                    ex = item.split('-', 1)[1].strip().replace('|', ' | ')
                    examples.append(ex)
                else:
                    examples.append(item.strip())

        # Registrid
        registers = []
        if 'REGISTRID' in data:
            for item in data['REGISTRID'].split('§§§'):
                registers.append(item.split('-', 1)[1].strip() if '-' in item else item.strip())

        # Registri põhjendused
        reg_just = []
        if 'REGISTRI-PÕHJENDUSED' in data:
            for item in data['REGISTRI-PÕHJENDUSED'].split('§§§'):
                reg_just.append(item.split('-', 1)[1].strip() if '-' in item else item.strip())

        # Registri kindlus
        reg_conf = []
        if 'REGISTRI-KINDLUS' in data:
            for item in data['REGISTRI-KINDLUS'].split('§§§'):
                reg_conf.append(item.split('-', 1)[1].strip() if '-' in item else item.strip())

        # Märgendid
        tags = []
        if 'MÄRGENDID' in data:
            for item in data['MÄRGENDID'].split('§§§'):
                tags.append(item.split('-', 1)[1].strip() if '-' in item else item.strip())

        # Märgendite põhjendused
        tag_just = []
        if 'MÄRGENDITE-PÕHJENDUSED' in data:
            for item in data['MÄRGENDITE-PÕHJENDUSED'].split('§§§'):
                tag_just.append(item.split('-', 1)[1].strip() if '-' in item else item.strip())

        # Koosta read
        total_meanings = data.get('TÄHENDUSTE-ARV', str(len(meanings)))
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

            if not tag_list or tag_text == "ei-kohaldu":
                results.append({
                    "Sõna": word,
                    "Tähenduse nr": i + 1,
                    "Tähendus": meaning,
                    "Tähenduste arv kokku": total_meanings,
                    "Sagedus": fval,                          # sage / keskmine / vähene
                    "Näited": exval,                          # 5 lauset, kui on
                    "Tekstiregister": reg,                    # informaalsetes / neutraalsetes-formaalsetes / ei kohaldu
                    "Registri põhjendus": rj,                 # 5–10 lauset
                    "Registri kindlus": rc,                   # väga kindel / pigem kindel / pigem ebakindel / väga ebakindel
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

            # Konsooli lühilogid
            print(f"   ✅ Tähendus {i+1}: {meaning}")
            print(f"      📈 Sagedus: {fval}")
            print(f"      📊 Register: {reg}  ({rc})")
            print(f"      🔍 Reg.põhjendus: {rj[:100]}{'...' if len(rj) > 100 else ''}")
            print(f"      🏷️ Märgend(id): {tag_text}")
            if tj != "ei-kohaldu":
                print(f"      📝 Märgendi põhjendus: {tj[:100]}{'...' if len(tj) > 100 else ''}")

    except Exception as e:
        print(f"   ⚠️ Parsimise viga: {e}")
        import traceback
        traceback.print_exc()
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

# --- Töötlemine ---
def process_word_analysis(word: str):
    context_path = os.path.join(DATA_FOLDER, f"{word}_full_context_only.txt")
    if not os.path.exists(context_path):
        print(f"⛔ Puudub kontekstifail: {context_path}")
        return None

    with open(context_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    full_text = "\n".join(lines)
    if tokenize_length(full_text) < 120000:
        context = full_text
        print(f"📄 Kasutan täielikku konteksti ({len(lines)} rida)")
    else:
        print(f"ℹ️ Fail on suur – kasutatakse embedding-põhist lõiguvalikut ({word})")
        index, chunks = ensure_index_exists(word, lines)
        relevant_chunks = get_relevant_chunks_max(word, chunks, index, max_k=150)
        context = "\n---\n".join(relevant_chunks)
        print(f"📄 Kasutan {len(relevant_chunks)} kõige relevantsemat lõiku")

    prompt = create_analysis_prompt(word)

    try:
        reply = get_completion(prompt, context)

        print("\n" + "="*80)
        print(f"🤖 MUDELI VASTUS sõnale '{word}':")
        print("="*80)
        print(reply)
        print("="*80)

        safe_word = sanitize_filename(word)
        out_path = os.path.join(OUTPUT_FOLDER, f"{safe_word}_analysis.txt")
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(reply)

        parsed_results = parse_analysis_response(reply, word)

        print(f"\n📊 PARSITUD TULEMUS:")
        print(f"   📝 Tähendusi kokku (read): {len(parsed_results)}")
        for result in parsed_results:
            print(f"   {result['Tähenduse nr']}. {result['Tähendus'][:50]}{'...' if len(result['Tähendus']) > 50 else ''}")
            print(f"      📈 Sagedus: {result['Sagedus']}")
            print(f"      📋 Register: {result['Tekstiregister']} ({result['Registri kindlus']})")
            print(f"      🏷️ Märgend: {result['Registrimärk']}")

        print(f"✅ {word} — Analüüs lõpetatud\n")
        return parsed_results

    except Exception as e:
        print(f"❌ Viga sõnaga {word}: {e}")
        return None

# --- Põhiprogramm ---
def main():
    all_rows = []

    # Loeme sisendfaili
    with open("sisend.txt", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        first_row = next(reader, None)
        # Kui on päis, proovi tuvastada; muidu loe kõiki
        if first_row and len(first_row) == 1 and first_row[0].lower() in ("sõna", "sona", "word"):
            words = [row[0].strip() for row in reader if len(row) >= 1 and row[0].strip()]
        else:
            words = []
            if first_row and len(first_row) >= 1 and first_row[0].strip():
                words.append(first_row[0].strip())
            for row in reader:
                if len(row) >= 1 and row[0].strip():
                    words.append(row[0].strip())

    for i, word in enumerate(words, 1):
        print(f"\n{'='*60}")
        print(f"📝 ANALÜÜSIN ({i}/{len(words)}): '{word}'")
        print(f"{'='*60}")

        result = process_word_analysis(word)
        if result:
            all_rows.extend(result)
        else:
            print(f"⚠️ Lisame tühja rea järjekorra säilitamiseks")
            all_rows.append({
                "Sõna": word,
                "Tähenduse nr": 1,
                "Tähendus": "töötlemata",
                "Tähenduste arv kokku": 0,
                "Sagedus": "kontekstifail puudub",
                "Näited": "ei saadaval",
                "Tekstiregister": "ei määratletud",
                "Registri põhjendus": "ei saadaval",
                "Registri kindlus": "pigem ebakindel",
                "Registrimärk": "ei kohaldu",
                "Märgendi põhjendus": "ei saadaval"
            })

        time.sleep(0.5)

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

    print(f"\n✅ Lõplik fail salvestatud: {FINAL_CSV}")
    print(f"📊 Kokku analüüsitud ridu: {len(all_rows)}")

    # Statistika
    unikaalsed_sõnad = len(set(row["Sõna"] for row in all_rows))
    toötletud_sõnad = len(set(row["Sõna"] for row in all_rows if row["Tähendus"] != "töötlemata"))
    toötlemata_sõnad = unikaalsed_sõnad - toötletud_sõnad
    keskmine_tähendusi = len(all_rows) / unikaalsed_sõnad if unikaalsed_sõnad > 0 else 0

    print(f"\n📈 Analüüsi statistika:")
    print(f"  Kokku sõnu: {unikaalsed_sõnad}")
    print(f"  Edukalt töödeldud sõnu: {toötletud_sõnad}")
    print(f"  Töötlemata sõnu (puuduvad failid): {toötlemata_sõnad}")
    print(f"  Keskmine tähendusi sõna kohta: {keskmine_tähendusi:.1f}")

if __name__ == "__main__":
    main()

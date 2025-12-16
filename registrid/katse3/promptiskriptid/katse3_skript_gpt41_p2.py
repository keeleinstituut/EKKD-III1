#Kood, registrite töörühma 3. katse tarvis. Eesmärk on OpenAI mudelile kaasa anda korpusest andmed, mida ta analüüsima peab. Igale analüüsitavale sõnale antakse kaasa ka tähendus.
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
MODEL = "gpt-4.1-2025-04-14"
EMBED_MODEL = SentenceTransformer("intfloat/multilingual-e5-base")
DATA_FOLDER = "contexts"
OUTPUT_FOLDER = "vastused"
FINAL_CSV = "vastused_koond.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs("vector_cache", exist_ok=True)

# --- Abi funktsioonid ---
def tokenize_length(text: str):
    enc = tiktoken.get_encoding("cl100k_base")
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

# --- Prompt sõna ja tähenduse analüüsimiseks ---
def create_definition_analysis_prompt(word: str, definition: str):
    return f"""Oled eesti keele sõnaraamatu koostaja. Sinu ülesanne on hinnata, kas sõnale „{word}" tuleb tähenduses „{definition}" lisada registrimärgend. Vasta ainult etteantud konteksti põhjal.

Vasta järgmistele küsimustele:

1. Otsusta sõna „{word}" tähenduse „{definition}" kohta, kas seda kasutatakse pigem informaalsetes või neutraalsetes/formaalsetes tekstides? Kui sa ei oska eristust teha või see ei tule selgelt esile, siis ütle, et „ei kohaldu". Palun põhjenda oma valikut 5-10 lausega.

2. Kui valid „ei kohaldu", siis ja ainult siis vaata enda treeningandmetesse ja otsusta selle põhjal, kas seda kasutatakse pigem informaalsetes või neutraalsetes/formaalsetes tekstides. Palun põhjenda oma valikut 5-10 lausega.

3. Nimeta sõna „{word}" erinevate tähenduste arv.

4. Iga tähenduse juurde lisa, kas sõna on selles tähenduses sage, keskmine või harv. Sagedusrühm vali võrdluses sõna teiste tähendustega.

5. Too 3 näidet etteantud materjalist, kus sõna „{word}" esineb just selles tähenduses. Kui näiteid on vähem, too nii palju, kui leidub.

6. Kui valisid, et sõna selles tähenduses esineb pigem *informaalsetes* tekstides, siis:
   - Millise registrimärgendeist sellele tähendusele lisaksid? (vali vähemalt üks, võid valida mitu):
     • halvustav (näiteks ajuhälvik, debiilik, inimrämps)
     • harv (näiteks ahvatama, mõistamisi, siinap)
     • kõnekeelne (näiteks igastahes, nokats, ära flippima)
     • lastekeelne (näiteks jänku, kätu, nuku)
     • luulekeelne (näiteks ehavalu, koidukuld, meeleheit)
     • murdekeelne (näiteks hämmelgas, jõõrdlik, kidelema)
     • rahvakeelne (näiteks heinakuu, viinakuu, männiseen)
     • stiilitundlik (näiteks armastet, kirjutet, seitung)
     • unarsõna (näiteks absurdum, ööp)
     • vananenud (näiteks automobiil, drogist)
     • vulgaarne (näiteks hoorapoeg, koinima, munn)
   - Märgend „harv" vali iga kord, kui tähendust leidub etteantud tekstimaterjalis vähe
   - Põhjenda iga märgendivalikut 5-10 lausega.

OLULINE: Pärast küsimustele vastamist anna oma vastused TÄPSELT järgmises struktureeritud formaadis parsimiseks:

--- STRUKTUREERITUD VASTUS ALGAB ---
SÕNA: {word}
TÄHENDUS: {definition}
TEKSTIREGISTER: informaalsetes/neutraalsetes-formaalsetes/ei-kohaldu
REGISTRI-PÕHJENDUS: [5-10 lauseline põhjendus]
TREENINGANDMETE-PÕHJENDUS: [5-10 lauseline põhjendus või ei-kohaldu]
TÄHENDUSTE-ARV: [number]
SAGEDUS: sage/keskmine/harv
NÄITED: Näide 1|Näide 2|Näide 3
REGISTRIMÄRK: halvustav,kõnekeelne või ei-kohaldu
MÄRGENDI-PÕHJENDUS: [5-10 lauseline põhjendus iga märgendi kohta või ei-kohaldu]
--- STRUKTUREERITUD VASTUS LÕPEB ---"""

# --- Parsimise funktsioon ---
def parse_definition_analysis_response(txt: str, word: str, definition: str) -> Dict[str, Any]:
    """
    Tagastab ühe tähenduse analüüsi tulemuse
    """
    result = {
        "Sõna": word,
        "Tähendus": definition,
        "Tekstiregister": "ei määratletud",
        "Registri põhjendus": "ei määratletud",
        "Treeningandmete põhjendus": "ei-kohaldu",
        "Tähenduste arv kokku": 0,
        "Sagedus": "ei määratletud",
        "Näited": "ei leitud",
        "Registrimärk": "ei-kohaldu",
        "Märgendi põhjendus": "ei-kohaldu"
    }
    
    try:
        # Otsime struktureeritud vastust märgendite vahelt
        structured_match = re.search(r'--- STRUKTUREERITUD VASTUS ALGAB ---(.*?)--- STRUKTUREERITUD VASTUS LÕPEB ---', txt, re.DOTALL)
        
        if not structured_match:
            print("   ⚠️ Struktureeritud vastust ei leitud, parsime kogu teksti")
            structured_text = txt
        else:
            structured_text = structured_match.group(1)
        
        lines = structured_text.split('\n')
        data = {}
        
        for line in lines:
            line = line.strip()
            if ':' in line and not line.startswith('http'):
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                data[key] = value
        
        # Parsime andmed
        if 'SÕNA' in data:
            result["Sõna"] = data['SÕNA']
        
        if 'TÄHENDUS' in data:
            result["Tähendus"] = data['TÄHENDUS']
        
        if 'TEKSTIREGISTER' in data:
            result["Tekstiregister"] = data['TEKSTIREGISTER']
        
        if 'REGISTRI-PÕHJENDUS' in data:
            result["Registri põhjendus"] = data['REGISTRI-PÕHJENDUS']
        
        if 'TREENINGANDMETE-PÕHJENDUS' in data:
            result["Treeningandmete põhjendus"] = data['TREENINGANDMETE-PÕHJENDUS']
        
        if 'TÄHENDUSTE-ARV' in data:
            result["Tähenduste arv kokku"] = data['TÄHENDUSTE-ARV']
        
        if 'SAGEDUS' in data:
            result["Sagedus"] = data['SAGEDUS']
        
        if 'NÄITED' in data:
            result["Näited"] = data['NÄITED'].replace('|', ' | ')
        
        if 'REGISTRIMÄRK' in data:
            result["Registrimärk"] = data['REGISTRIMÄRK']
        
        if 'MÄRGENDI-PÕHJENDUS' in data:
            result["Märgendi põhjendus"] = data['MÄRGENDI-PÕHJENDUS']
        
        print(f"   ✅ Tähendus: {result['Tähendus'][:50]}{'...' if len(result['Tähendus']) > 50 else ''}")
        print(f"      📊 Register: {result['Tekstiregister']}")
        print(f"      🔍 Registri põhjendus: {result['Registri põhjendus'][:100]}{'...' if len(result['Registri põhjendus']) > 100 else ''}")
        print(f"      🏷️ Märgend: {result['Registrimärk']}")
        if result["Märgendi põhjendus"] != "ei-kohaldu":
            print(f"      📝 Märgendi põhjendus: {result['Märgendi põhjendus'][:100]}{'...' if len(result['Märgendi põhjendus']) > 100 else ''}")
    
    except Exception as e:
        print(f"   ⚠️ Parsimise viga: {e}")
        import traceback
        traceback.print_exc()
        result["Tähendus"] = "parsimise viga"
    
    return result

# --- Sõna ja tähenduse töötlemise funktsioon ---
def process_definition_analysis(word: str, definition: str):
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
        # Otsime relevantset sisu nii sõna kui tähenduse põhjal
        query = f"{word} {definition}"
        relevant_chunks = get_relevant_chunks_max(query, chunks, index, max_k=150)
        context = "\n---\n".join(relevant_chunks)
        print(f"📄 Kasutan {len(relevant_chunks)} kõige relevantsemast lõiku")

    prompt = create_definition_analysis_prompt(word, definition)

    try:
        reply = get_completion(prompt, context)
        
        # Prindime mudeli toorvastuse
        print("\n" + "="*80)
        print(f"🤖 MUDELI VASTUS sõnale '{word}' tähenduses '{definition[:50]}{'...' if len(definition) > 50 else ''}':")
        print("="*80)
        print(reply)
        print("="*80)
        
        # Salvesta toorvastus faili
        safe_word = sanitize_filename(word)
        safe_definition = sanitize_filename(definition)
        out_path = os.path.join(OUTPUT_FOLDER, f"{safe_word}_{safe_definition}_analysis.txt")
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(reply)
        
        # Parsime vastuse
        parsed_result = parse_definition_analysis_response(reply, word, definition)
        
        print(f"✅ {word} (tähendus: {definition[:30]}{'...' if len(definition) > 30 else ''}) — Analüüs lõpetatud\n")
        
        return parsed_result

    except Exception as e:
        print(f"❌ Viga sõnaga {word}, tähendus {definition}: {e}")
        return None

# --- Põhiprogramm ---
def main():
    all_rows = []
    
    # Loeme sisendandmeid (eeldame, et fail sisaldab veerge: sõna, tähendus)
    input_file = "sisend.tsv"  # Muuda faili nime vastavalt vajadusele
    
    if not os.path.exists(input_file):
        print(f"⛔ Sisend fail '{input_file}' puudub!")
        print("Palun loo fail järgmise struktuuriga:")
        print("sõna<TAB>tähendus")
        print("kits<TAB>koduloom")
        print("kits<TAB>Hiina sodiaagimärk")
        return
    
    with open(input_file, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader, None)  # jäta päis vahele, kui on
        word_definitions = []
        for row in reader:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                word_definitions.append((row[0].strip(), row[1].strip()))
    
    for i, (word, definition) in enumerate(word_definitions, 1):
        print(f"\n{'='*60}")
        print(f"📝 ANALÜÜSIN ({i}/{len(word_definitions)}): '{word}' - '{definition[:50]}{'...' if len(definition) > 50 else ''}'")
        print(f"{'='*60}")
        
        result = process_definition_analysis(word, definition)
        if result:
            all_rows.append(result)
        else:
            # Lisa tühi rida järjekorra säilitamiseks
            print(f"⚠️ Lisame tühja rea järjekorra säilitamiseks")
            all_rows.append({
                "Sõna": word,
                "Tähendus": definition,
                "Tekstiregister": "ei määratletud",
                "Registri põhjendus": "kontekstifail puudub",
                "Treeningandmete põhjendus": "ei saadaval",
                "Tähenduste arv kokku": 0,
                "Sagedus": "ei määratletud",
                "Näited": "ei saadaval",
                "Registrimärk": "ei kohaldu",
                "Märgendi põhjendus": "ei saadaval"
            })
        
        time.sleep(0.5)  # Väike paus

    # Salvesta CSV
    fieldnames = [
        "Sõna", "Tähendus", "Tekstiregister", "Registri põhjendus", 
        "Treeningandmete põhjendus", "Tähenduste arv kokku", "Sagedus", 
        "Näited", "Registrimärk", "Märgendi põhjendus"
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
    toötletud_read = len([row for row in all_rows if row["Tähendus"] != "parsimise viga"])
    toötlemata_read = len(all_rows) - toötletud_read

    print(f"\n📈 Analüüsi statistika:")
    print(f"  Kokku sõnu: {unikaalsed_sõnad}")
    print(f"  Kokku tähendusi: {len(all_rows)}")
    print(f"  Edukalt töödeldud tähendusi: {toötletud_read}")
    print(f"  Töötlemata tähendusi: {toötlemata_read}")

if __name__ == "__main__":
    main()

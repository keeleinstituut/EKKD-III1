#Kood registrite töörühma 3. katse tarvis. Eesmärk on OpenAI mudelile kaasa anda korpusest andmed, mida ta analüüsima peab. Sõna antakse ilma tähenduseta.
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
from typing import List
from sentence_transformers import SentenceTransformer

# --- Konfiguratsioon ---
client = openai.OpenAI()
MODEL = "gpt-4o"
EMBED_MODEL = SentenceTransformer("intfloat/multilingual-e5-base")
DATA_FOLDER = "contexts"
OUTPUT_FOLDER = "vastused_v2"
FINAL_CSV = "vastused_koond_v2.csv"

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
        max_tokens=2000,
        temperature=0.1
    )
    return response.choices[0].message.content

def sanitize_filename(text):
    return re.sub(r'[<>:"/\\|?*]', '_', text)[:100]

# --- Uus prompt sõna analüüsimiseks ---
def create_analysis_prompt(word: str):
    return f"""Oled eesti keele sõnaraamatu koostaja. Sinu ülesanne on analüüsida sõna „{word}" kasutust etteantud tekstimaterjalis ja otsustada, kas selle tähendustele tuleks lisada registrimärgend.

Vasta järgmistele küsimustele, tuginedes ainult etteantud materjalile. Ole lühike ja konkreetne:

1. Nimeta sõna „{word}" kõik tähendused, mida etteantud tekstides näed.
2. Iga tähenduse juurde lisa, mitmes lauses sõna selles tähenduses esineb.
3. Too iga tähenduse kohta etteantud materjalist kuni 10 näitelauset. Kui neid on andmetes vähem, siis too nii palju, kui leidub.
4. Otsusta sõna iga tähenduse kohta, kas seda kasutatakse pigem informaalsetes või neutraalsetes/formaalsetes tekstides? Kui sa ei oska eristust teha või see ei tule selgelt esile, siis ütle, et „ei kohaldu". Palun põhjenda oma valikut.
5. Kui mõnda tähendust kasutatakse pigem informaalsetes tekstides, siis vali sellele sobiv registrimärgend järgmistest:
• halvustav
• harv
• kõnekeelne
• lastekeelne
• luulekeelne
• murdekeelne
• rahvakeelne
• stiilitundlik
• unarsõna
• vananenud
• vulgaarne

Iga valiku korral põhjenda, miks just see märgend sobib. Igal informaalsel tähendusel peab olema vähemalt üks märgend. Kui sobib mitu, too mitu.

OLULINE: Pärast küsimustele vastamist anna oma vastused TÄPSELT järgmises struktureeritud formaadis:

VASTUS||TÄHENDUSED: [tähendus1; tähendus2; tähendus3]||ESINEMISED: [tähendus1: X lauses; tähendus2: Y lauses]||NÄITED: [TÄHENDUS1: näide1 | näide2 | näide3; TÄHENDUS2: näide1 | näide2]||REGISTRID: [TÄHENDUS1: informaalsetes/neutraalsetes-formaalsetes/ei-kohaldu; TÄHENDUS2: informaalsetes/neutraalsetes-formaalsetes/ei-kohaldu]||PÕHJENDUSED: [TÄHENDUS1: põhjendus1; TÄHENDUS2: põhjendus2]||MÄRGENDID: [TÄHENDUS1: märgend1, märgend2 või ei-kohaldu; TÄHENDUS2: märgend1 või ei-kohaldu]||MÄRGENDITE-PÕHJENDUS: [TÄHENDUS1: märgend1: põhjendus1, märgend2: põhjendus2; TÄHENDUS2: ei-kohaldu]||LÕPP"""

# --- Uus parsimise funktsioon ---
def parse_analysis_response(txt, word):
    result = {
        "Sõna": word,
        "Tähendused": "",
        "Esinemiste arv": "",
        "Näited": "",
        "Tekstiregistrid": "",
        "Registri põhjendused": "",
        "Registrimärgendid": "",
        "Märgendite põhjendused": ""
    }
    
    # Otsime struktureeritud vastust
    structured_match = re.search(r'VASTUS\|\|(.*?)\|\|LÕPP', txt, re.DOTALL)
    
    if structured_match:
        print("   ✅ Struktureeritud vastus leitud")
        structured_text = structured_match.group(1)
        parts = structured_text.split('||')
        
        for part in parts:
            part = part.strip()
            
            if part.startswith('TÄHENDUSED:'):
                tahendused_text = part.replace('TÄHENDUSED:', '').strip()
                if tahendused_text:
                    tahendused = [t.strip() for t in tahendused_text.split(';') if t.strip()]
                    result["Tähendused"] = ' | '.join([f"{i+1}. {t}" for i, t in enumerate(tahendused)])
                    
            elif part.startswith('ESINEMISED:'):
                esinemised_text = part.replace('ESINEMISED:', '').strip()
                result["Esinemiste arv"] = esinemised_text
                
            elif part.startswith('NÄITED:'):
                naited_text = part.replace('NÄITED:', '').strip()
                result["Näited"] = naited_text
                
            elif part.startswith('REGISTRID:'):
                registrid_text = part.replace('REGISTRID:', '').strip()
                result["Tekstiregistrid"] = registrid_text
                
            elif part.startswith('PÕHJENDUSED:'):
                pohjendused_text = part.replace('PÕHJENDUSED:', '').strip()
                result["Registri põhjendused"] = pohjendused_text
                
            elif part.startswith('MÄRGENDID:'):
                margendid_text = part.replace('MÄRGENDID:', '').strip()
                result["Registrimärgendid"] = margendid_text
                
            elif part.startswith('MÄRGENDITE-PÕHJENDUS:'):
                margendite_pohjendused_text = part.replace('MÄRGENDITE-PÕHJENDUS:', '').strip()
                result["Märgendite põhjendused"] = margendite_pohjendused_text
    
    else:
        print("   ⚠️ Struktureeritud vastust ei leitud, kasutame vaba teksti parsimist")
        
        # Vaba teksti parsimise loogika (lihtsustatud versioon)
        # Otsime tähendusi
        tahendused_match = re.search(r'1\.\s*([^2]*?)(?=2\.|$)', txt, re.DOTALL)
        if tahendused_match:
            tahendused_text = tahendused_match.group(1).strip()
            # Lihtne heuristika - otsime loetelu märke
            tahendused_lines = [line.strip() for line in tahendused_text.split('\n') if line.strip()]
            tahendused = []
            for line in tahendused_lines:
                if re.match(r'^[•\-\*]\s*|^\d+\.\s*|^[a-z]\)\s*', line):
                    tahendused.append(re.sub(r'^[•\-\*]\s*|^\d+\.\s*|^[a-z]\)\s*', '', line))
            if tahendused:
                result["Tähendused"] = ' | '.join([f"{i+1}. {t}" for i, t in enumerate(tahendused)])
        
        # Muud väljad jätame tühjaks või märgime "vaba tekst"
        result["Esinemiste arv"] = "vaba tekst - vt toorvastust"
        result["Näited"] = "vaba tekst - vt toorvastust"
        result["Tekstiregistrid"] = "vaba tekst - vt toorvastust"
        result["Registri põhjendused"] = "vaba tekst - vt toorvastust"
        result["Registrimärgendid"] = "vaba tekst - vt toorvastust"
        result["Märgendite põhjendused"] = "vaba tekst - vt toorvastust"
    
    # Vaikeväärtused tühjadele väljadele
    for key in result:
        if not result[key]:
            result[key] = "ei määratletud"
    
    return result

# --- Sõna töötlemise funktsioon ---
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
        print(f"📄 Kasutan {len(relevant_chunks)} kõige relevantsemast lõiku")

    prompt = create_analysis_prompt(word)

    try:
        reply = get_completion(prompt, context)
        
        # Prindime mudeli toorvastuse
        print("\n" + "="*80)
        print(f"🤖 MUDELI VASTUS sõnale '{word}':")
        print("="*80)
        print(reply)
        print("="*80)
        
        # Salvesta toorvastus faili
        safe_word = sanitize_filename(word)
        out_path = os.path.join(OUTPUT_FOLDER, f"{safe_word}_analysis.txt")
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(reply)
        
        # Parsime vastuse
        parsed_result = parse_analysis_response(reply, word)
        
        # Prindime parsimise tulemuse
        print(f"\n📊 PARSITUD TULEMUS:")
        print(f"   📝 Tähendused: {parsed_result['Tähendused'][:100]}{'...' if len(parsed_result['Tähendused']) > 100 else ''}")
        print(f"   📊 Esinemised: {parsed_result['Esinemiste arv'][:100]}{'...' if len(parsed_result['Esinemiste arv']) > 100 else ''}")
        print(f"   📋 Registrid: {parsed_result['Tekstiregistrid'][:100]}{'...' if len(parsed_result['Tekstiregistrid']) > 100 else ''}")
        print(f"   🏷️ Märgendid: {parsed_result['Registrimärgendid'][:100]}{'...' if len(parsed_result['Registrimärgendid']) > 100 else ''}")
        
        print(f"✅ {word} — Analüüs lõpetatud\n")
        
        return parsed_result

    except Exception as e:
        print(f"❌ Viga sõnaga {word}: {e}")
        return None

# --- Põhiprogramm ---
def main():
    all_rows = []
    
    # Loeme sisend_2.tsv faili (ainult sõnad)
    with open("sisend_2.tsv", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # jäta päis vahele, kui on
        words = [row[0].strip() for row in reader if len(row) >= 1 and row[0].strip()]
    
    for i, word in enumerate(words, 1):
        print(f"\n{'='*60}")
        print(f"📝 ANALÜÜSIN ({i}/{len(words)}): '{word}'")
        print(f"{'='*60}")
        
        result = process_word_analysis(word)
        if result:
            all_rows.append(result)
        else:
            # Lisa tühi rida järjekorra säilitamiseks
            print(f"⚠️ Lisame tühja rea järjekorra säilitamiseks")
            all_rows.append({
                "Sõna": word,
                "Tähendused": "töötlemata",
                "Esinemiste arv": "kontekstifail puudub",
                "Näited": "ei saadaval",
                "Tekstiregistrid": "ei määratletud",
                "Registri põhjendused": "ei saadaval",
                "Registrimärgendid": "ei kohaldu",
                "Märgendite põhjendused": "ei saadaval"
            })
        
        time.sleep(0.5)  # Väike paus

    # Salvesta CSV
    fieldnames = [
        "Sõna", "Tähendused", "Esinemiste arv", "Näited", 
        "Tekstiregistrid", "Registri põhjendused", 
        "Registrimärgendid", "Märgendite põhjendused"
    ]

    with open(FINAL_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";", quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\n✅ Lõplik fail salvestatud: {FINAL_CSV}")
    print(f"📊 Kokku analüüsitud {len(all_rows)} sõna")

    # Statistika
    toötletud = len([row for row in all_rows if row["Tähendused"] != "töötlemata"])
    toötlemata = len(all_rows) - toötletud

    print(f"\n📈 Analüüsi statistika:")
    print(f"  Edukalt töödeldud: {toötletud}")
    print(f"  Töötlemata (puuduvad failid): {toötlemata}")

if __name__ == "__main__":
    main()
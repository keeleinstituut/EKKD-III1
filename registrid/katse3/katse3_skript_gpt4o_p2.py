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
from typing import List
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
        max_tokens=1500,
        temperature=0.1
    )
    return response.choices[0].message.content

def sanitize_filename(text):
    return re.sub(r'[<>:"/\\|?*]', '_', text)[:100]

# --- Prompt ---
def create_prompt(word: str, definition: str):
    return f"""Oled eesti keele sõnaraamatu koostaja. Sinu ülesanne on hinnata, kas sõnale „{word}" tuleb tähenduses „{definition}" lisada registrimärgend. Vasta ainult etteantud konteksti põhjal ja hoia vastused lühikesed ning konkreetsed.

Vasta järgmistele küsimustele:

1. Otsusta sõna „{word}" tähenduse „{definition}" kohta, kas seda kasutatakse pigem informaalsetes või neutraalsetes/formaalsetes tekstides? Kui sa ei oska eristust teha või see ei tule selgelt esile, siis ütle, et „ei kohaldu". Palun põhjenda oma valikut.

2. Too kuni 10 näidet antud materjalist, kus sõna „{word}" esineb just selles tähenduses. Kui näiteid on vähem, too nii palju, kui leidub.

3. Kui valisid, et sõna selles tähenduses esineb pigem *informaalsetes* tekstides, siis:
• Millise registrimärgendeist sellele tähendusele lisaksid? (vali vähemalt üks, võid valida mitu):
• halvustav, harv, kõnekeelne, lastekeelne, luulekeelne, murdekeelne, rahvakeelne, stiilitundlik, unarsõna, vananenud, vulgaarne
• Põhjenda iga märgendivalikut lühidalt.

OLULINE: Pärast küsimustele vastamist anna oma vastused TÄPSELT järgmises struktureeritud formaadis:

VASTUS||TEKSTIREGISTER: [informaalsetes/neutraalsetes-formaalsetes/ei-kohaldu]||PÕHJENDUS: [lühike põhjendus]||NÄITED: [näide1; näide2; näide3]||REGISTRIMÄRGENDID: [märgend1, märgend2 või ei-kohaldu]||MÄRGENDITE-PÕHJENDUS: [märgend1: põhjendus1; märgend2: põhjendus2 või ei-kohaldu]||LÕPP"""

# --- Hübriid parsimise funktsioon ---
def parse_response(txt, word, definition):
    result = {
        "Sõna": word,
        "Tähendus": definition,
        "Tekstiregister": "",
        "Põhjendus": "",
        "Näited (kuni 10)": "",
        "Registrimärgend(id)": "",
        "Märgendite põhjendus": ""
    }
    
    # Otsime struktureeritud vastust
    structured_match = re.search(r'VASTUS\|\|(.*?)\|\|LÕPP', txt, re.DOTALL)
    
    if structured_match:
        # Struktureeritud vastus leitud
        structured_text = structured_match.group(1)
        parts = structured_text.split('||')
        
        for part in parts:
            part = part.strip()
            
            if part.startswith('TEKSTIREGISTER:'):
                register = part.replace('TEKSTIREGISTER:', '').strip()
                if 'informaalsetes' in register.lower():
                    result["Tekstiregister"] = "informaalsetes"
                elif 'neutraalsetes' in register.lower() or 'formaalsetes' in register.lower():
                    result["Tekstiregister"] = "neutraalsetes/formaalsetes"
                elif 'ei-kohaldu' in register.lower() or 'ei kohaldu' in register.lower():
                    result["Tekstiregister"] = "ei kohaldu"
                else:
                    result["Tekstiregister"] = register
                    
            elif part.startswith('PÕHJENDUS:'):
                result["Põhjendus"] = part.replace('PÕHJENDUS:', '').strip()
                
            elif part.startswith('NÄITED:'):
                naited_text = part.replace('NÄITED:', '').strip()
                if naited_text and 'puuduvad' not in naited_text.lower():
                    examples = [ex.strip().strip('"\'„"') for ex in naited_text.split(';') if ex.strip()]
                    if examples:
                        numbered_examples = []
                        for i, example in enumerate(examples[:10], 1):
                            numbered_examples.append(f"{i}. {example}")
                        result["Näited (kuni 10)"] = ' | '.join(numbered_examples)
                    else:
                        result["Näited (kuni 10)"] = "näited puuduvad"
                else:
                    result["Näited (kuni 10)"] = "näited puuduvad"
                    
            elif part.startswith('REGISTRIMÄRGENDID:'):
                margendid_text = part.replace('REGISTRIMÄRGENDID:', '').strip()
                if 'ei-kohaldu' in margendid_text.lower() or 'ei kohaldu' in margendid_text.lower():
                    result["Registrimärgend(id)"] = "ei kohaldu"
                else:
                    margendid = [m.strip() for m in margendid_text.split(',') if m.strip()]
                    result["Registrimärgend(id)"] = ", ".join(margendid) if margendid else "ei kohaldu"
                    
            elif part.startswith('MÄRGENDITE-PÕHJENDUS:'):
                pohjendused_text = part.replace('MÄRGENDITE-PÕHJENDUS:', '').strip()
                if 'ei-kohaldu' in pohjendused_text.lower() or 'ei kohaldu' in pohjendused_text.lower():
                    result["Märgendite põhjendus"] = "ei kohaldu"
                else:
                    result["Märgendite põhjendus"] = pohjendused_text
    
    else:
        # Struktureeritud vastust ei leitud, kasutame vana parsimisloogikat
        print("   ⚠️ Struktureeritud vastust ei leitud, kasutame vaba teksti parsimist")
        
        # 1. Tekstiregister
        first_sentence_match = re.search(r'1\.\s*[^\.]*?(informaalsetes|neutraalsetes|formaalsetes|ei kohaldu)[^\.]*\.', txt, re.IGNORECASE)
        if first_sentence_match:
            register_word = first_sentence_match.group(1).lower()
            if 'informaalsetes' in register_word:
                result["Tekstiregister"] = "informaalsetes"
            elif 'neutraalsetes' in register_word or 'formaalsetes' in register_word:
                result["Tekstiregister"] = "neutraalsetes/formaalsetes"
            elif 'ei kohaldu' in register_word:
                result["Tekstiregister"] = "ei kohaldu"
        
        # 2. Põhjendus
        pohjendus_match = re.search(r'1\.[^\.]*\.\s*([^2]*?)(?=2\.|$)', txt, re.DOTALL)
        if pohjendus_match:
            pohjendus_text = pohjendus_match.group(1).strip()
            pohjendus_text = re.sub(r'Palun põhjenda oma valikut\.?\s*[-–]?\s*', '', pohjendus_text, flags=re.IGNORECASE)
            pohjendus_text = re.sub(r'\s+', ' ', pohjendus_text)
            result["Põhjendus"] = pohjendus_text.strip()
        
        # 3. Näited  
        naited_match = re.search(r'2\.\s*([^3]*?)(?=3\.|VASTUS|$)', txt, re.DOTALL)
        if naited_match:
            naited_text = naited_match.group(1).strip()
            if 'puudub' not in naited_text.lower() and 'ei saa' not in naited_text.lower():
                quotes_pattern = r'[„"\'"]([^„"\']*?)[„"\'"]\s*[-–]?'
                quotes_matches = re.findall(quotes_pattern, naited_text)
                if quotes_matches:
                    numbered_examples = []
                    for i, example in enumerate(quotes_matches[:10], 1):
                        numbered_examples.append(f"{i}. {example}")
                    result["Näited (kuni 10)"] = ' | '.join(numbered_examples)
                else:
                    result["Näited (kuni 10)"] = "näited puuduvad"
            else:
                result["Näited (kuni 10)"] = "näited puuduvad"
    
    # Vaikeväärtused
    if not result["Tekstiregister"]:
        result["Tekstiregister"] = "ei määratletud"
    if not result["Põhjendus"]:
        result["Põhjendus"] = "põhjendus puudub"
    if not result["Näited (kuni 10)"]:
        result["Näited (kuni 10)"] = "näited puuduvad"
    if not result["Registrimärgend(id)"]:
        result["Registrimärgend(id)"] = "ei kohaldu"
    if not result["Märgendite põhjendus"]:
        result["Märgendite põhjendus"] = "ei kohaldu"
    
    return result

# --- Sõna töötlemise funktsioon---
def process_word(word: str, definition: str):
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
        relevant_chunks = get_relevant_chunks_max(word, chunks, index, max_k=100)
        context = "\n---\n".join(relevant_chunks)
        print(f"📄 Kasutan {len(relevant_chunks)} kõige relevantsemast lõiku")

    prompt = create_prompt(word, definition)

    try:
        reply = get_completion(prompt, context)
        
        # Prindime mudeli toorvastuse
        print("\n" + "="*80)
        print(f"🤖 MUDELI VASTUS sõnale '{word}' ({definition}):")
        print("="*80)
        print(reply)
        print("="*80)
        
        # Salvesta toorvastus faili (su stiilis)
        safe_word = sanitize_filename(word)
        safe_def = sanitize_filename(definition)
        out_path = os.path.join(OUTPUT_FOLDER, f"{safe_word}__{safe_def}.txt")
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(reply)
        
        # Parsime vastuse
        parsed_result = parse_response(reply, word, definition)
        
        # Prindime parsimise tulemuse
        print(f"\n📊 PARSITUD TULEMUS:")
        print(f"   📝 Tekstiregister: {parsed_result['Tekstiregister']}")
        print(f"   💭 Põhjendus: {parsed_result['Põhjendus'][:100]}{'...' if len(parsed_result['Põhjendus']) > 100 else ''}")
        print(f"   📋 Näiteid leitud: {len(parsed_result['Näited (kuni 10)'].split('|')) if parsed_result['Näited (kuni 10)'] != 'näited puuduvad' else 0}")
        print(f"   🏷️ Registrimärgendid: {parsed_result['Registrimärgend(id)']}")
        if parsed_result['Märgendite põhjendus'] != "ei kohaldu":
            print(f"   ❓ Märgendite põhjendused:")
            for pohjendus in parsed_result['Märgendite põhjendus'].split(';'):
                if pohjendus.strip():
                    print(f"      • {pohjendus.strip()}")
                    
        # Hoiatus kui märgendeid on aga põhjendusi ei ole
        margendid = [m.strip() for m in parsed_result['Registrimärgend(id)'].split(',') if m.strip() != 'ei kohaldu']
        if margendid and parsed_result['Märgendite põhjendus'] == "ei kohaldu":
            print(f"   ⚠️ HOIATUS: Märgendid valitud ({margendid}) aga põhjendused puuduvad!")
        elif margendid:
            pohjenduste_arv = len([p for p in parsed_result['Märgendite põhjendus'].split(';') if p.strip()])
            if len(margendid) != pohjenduste_arv:
                print(f"   ⚠️ HOIATUS: Märgendeid {len(margendid)}, aga põhjendusi {pohjenduste_arv}")
        
        print(f"✅ {word} — Töötlemine lõpetatud\n")
        
        return parsed_result

    except Exception as e:
        print(f"❌ Viga sõnaga {word}: {e}")
        return None

# --- Põhiprogramm ---
def main():
    all_rows = []
    
    with open("sisend.tsv", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        next(reader)  # jäta päis vahele
        entries = [(row[0].strip(), row[1].strip()) for row in reader if len(row) >= 2]
    
    for i, (word, definition) in enumerate(entries, 1):
        print(f"\n{'='*60}")
        print(f"📝 TÖÖTLEN ({i}/{len(entries)}): '{word}' - '{definition}'")
        print(f"{'='*60}")
        
        result = process_word(word, definition)
        if result:
            all_rows.append(result)
        else:
            # Lisa tühi rida järjekorra säilitamiseks
            print(f"⚠️ Lisame tühja rea järjekorra säilitamiseks")
            all_rows.append({
                "Sõna": word,
                "Tähendus": definition,
                "Tekstiregister": "töötlemata",
                "Põhjendus": "kontekstifail puudub",
                "Näited (kuni 10)": "näited puuduvad",
                "Registrimärgend(id)": "ei kohaldu",
                "Märgendite põhjendus": "ei kohaldu"
            })
        
        time.sleep(0.5)  # Väike paus

    # Salvesta CSV
    fieldnames = [
        "Sõna", "Tähendus", "Tekstiregister", "Põhjendus",
        "Näited (kuni 10)", "Registrimärgend(id)", "Märgendite põhjendus"
    ]

    with open(FINAL_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";", quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\n✅ Lõplik fail salvestatud: {FINAL_CSV}")
    print(f"📊 Kokku töödeldud {len(all_rows)} kirjet")

    # Statistika
    tekstiregister_stats = {}
    for row in all_rows:
        reg = row["Tekstiregister"]
        tekstiregister_stats[reg] = tekstiregister_stats.get(reg, 0) + 1

    print("\n📈 Tekstiregistri statistika:")
    for reg, count in tekstiregister_stats.items():
        print(f"  {reg}: {count}")

if __name__ == "__main__":
    main()
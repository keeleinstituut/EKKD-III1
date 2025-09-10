#Kood, registrite töörühma 3. katse tarvis. Eesmärk on Google'i mudelile kaasa anda korpusest andmed, mida ta analüüsima peab. Igale analüüsitavale sõnale antakse kaasa ka tähendus.
#Kui kontekst on liiga suur, siis see vektoriseeritakse.
#Autor: Eleri Aedmaa
import os
import csv
import google.generativeai as genai
import pickle
import faiss
import time
import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from google.api_core import exceptions

# --- Konfiguratsioon ---
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("VIGA: GOOGLE_API_KEY keskkonnamuutja pole seadistatud.")
    exit()

# Mudelite ja parameetrite seadistus
MODEL_NAME = "gemini-2.5-pro"
EMBED_MODEL = SentenceTransformer("intfloat/multilingual-e5-base")

DATA_FOLDER = "contexts"
OUTPUT_FOLDER = "vastused"
FINAL_CSV = "vastused_koond.csv"
VECTOR_CACHE_FOLDER = "vector_cache"

# Kontrollide seadistus
MAX_CONTEXT_LINES = 10000
NUM_RELEVANT_CHUNKS = 150

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(VECTOR_CACHE_FOLDER, exist_ok=True)

# --- VEKTOR-FUNKTSIOONID ---


def build_faiss_index(chunks: List[str]):
    passages = [f"passage: {chunk}" for chunk in chunks]
    embeddings = EMBED_MODEL.encode(passages, show_progress_bar=True)
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
    index_path = os.path.join(VECTOR_CACHE_FOLDER, sanitize_filename(word))
    if os.path.exists(index_path + ".faiss") and os.path.exists(index_path + "_chunks.pkl"):
        print("   -> Leidsin vahemälust valmis vektoriindeksi, laen selle.")
        return load_index(index_path)
    
    print("   -> Vektoriindeksit ei leitud, loon uue (see võib võtta aega)...")
    index, _ = build_faiss_index(full_lines)
    save_index(index, full_lines, index_path)
    print("   -> Uus indeks loodud ja salvestatud vahemällu.")
    return index, full_lines

def get_relevant_chunks_max(query: str, chunks: List[str], index, max_k=150):
    query_vec = EMBED_MODEL.encode([f"query: {query}"], show_progress_bar=False)
    distances, indices = index.search(query_vec, min(len(chunks), max_k + 10))
    sorted_chunks = [chunks[i] for i in indices[0]]
    unique_chunks = list(dict.fromkeys(sorted_chunks))
    return unique_chunks[:max_k]

# --- ÜLEJÄÄNUD ABIFUNKTSIOONID, PROMPTID JA PARSERID ---


def get_completion(prompt: str, context: str) -> str:
    """
    Saadab päringu Gemini mudelile ja tagastab vastuse tekstina.
    """
    generation_config = {
        "temperature": 0.1,
        "max_output_tokens": 16000,
    }

    # Loome mudeli, edastades süsteemiprompti ja genereerimise seaded
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=prompt,
        generation_config=generation_config
    )
    
    # Saadame konteksti kui kasutaja sõnumi
    response = model.generate_content(context)
    
    # Kontrollime, kas vastus blokeeriti turvakaalutlustel
    if not response.parts:
         print("⚠️ Mudeli vastus oli tühi või blokeeritud.")
         return "MUDELI_VASTUS_BLOKEERITUD"
         
    return response.text

def sanitize_filename(text):
    return re.sub(r'[<>:"/\\|?*]', '_', text)[:100]

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

def parse_definition_analysis_response(txt: str, word: str, definition: str) -> Dict[str, Any]:
    """
    Tagastab ühe tähenduse analüüsi tulemuse.
    Funktsioon ei vaja muudatusi, kuna see töötleb mudeli tekstilist väljundit.
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
            print("     ⚠️ Struktureeritud vastust ei leitud, parsime kogu teksti")
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
        
        print(f"     ✅ Tähendus: {result['Tähendus'][:50]}{'...' if len(result['Tähendus']) > 50 else ''}")
        print(f"        📊 Register: {result['Tekstiregister']}")
        print(f"        🔍 Registri põhjendus: {result['Registri põhjendus'][:100]}{'...' if len(result['Registri põhjendus']) > 100 else ''}")
        print(f"        🏷️  Märgend: {result['Registrimärk']}")
        if result["Märgendi põhjendus"] != "ei-kohaldu":
            print(f"        📝 Märgendi põhjendus: {result['Märgendi põhjendus'][:100]}{'...' if len(result['Märgendi põhjendus']) > 100 else ''}")
    
    except Exception as e:
        print(f"     ⚠️ Parsimise viga: {e}")
        import traceback
        traceback.print_exc()
        result["Tähendus"] = "parsimise viga"
    
    return result

# --- SÕNA JA TÄHENDUSE TÖÖTLEMISE FUNKTSIOON ---
def process_definition_analysis(word: str, definition: str):
    context_path = os.path.join(DATA_FOLDER, f"{word}_full_context_only.txt")
    if not os.path.exists(context_path):
        print(f"⛔ Puudub kontekstifail: {context_path}")
        return None

    with open(context_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # --- KONTEKSTI VALIKU LOOGIKA ---
    if len(lines) > MAX_CONTEXT_LINES:
        print(f"⚠️  Kontekstifail on suur ({len(lines)} rida). Kasutan vektorotsingut.")
        index, chunks = ensure_index_exists(word, lines)
        
        # Loome spetsiifilise päringu nii sõna kui tähendusega
        query = f"{word} {definition}"
        print(f"   -> Otsin relevantseid lõike päringuga: '{query[:100]}...'")

        relevant_chunks = get_relevant_chunks_max(query, chunks, index, max_k=NUM_RELEVANT_CHUNKS)
        context = "\n---\n".join(relevant_chunks)
        print(f"📄 Kasutan kontekstina {len(relevant_chunks)} kõige relevantsemat lõiku.")
    else:
        print(f"📄 Fail on piisavalt väike ({len(lines)} rida). Kasutan täielikku konteksti.")
        context = "\n".join(lines)

    prompt = create_definition_analysis_prompt(word, definition)

    # Vigade käsitlemise ja korduskatsete loogika
    max_retries = 5
    retry_delay = 60
    for attempt in range(max_retries):
        try:
            reply = get_completion(prompt, context)
            
            # Ülejäänud funktsioon on sama...
            print("\n" + "="*80)
            print(f"🤖 MUDELI VASTUS sõnale '{word}' tähenduses '{definition[:50]}{'...' if len(definition) > 50 else ''}':")
            print("="*80)
            print(reply)
            print("="*80)
            
            safe_word = sanitize_filename(word)
            safe_definition = sanitize_filename(definition)
            out_path = os.path.join(OUTPUT_FOLDER, f"{safe_word}_{safe_definition}_analysis.txt")
            with open(out_path, "w", encoding="utf-8") as out_f:
                out_f.write(reply)
            
            parsed_result = parse_definition_analysis_response(reply, word, definition)
            
            print(f"✅ {word} (tähendus: {definition[:30]}{'...' if len(definition) > 30 else ''}) — Analüüs lõpetatud\n")
            return parsed_result

        except exceptions.ResourceExhausted as e:
            print(f"⚠️ Viga 429: Limiit ületatud. Ootan {retry_delay} sekundit. Katse {attempt + 1}/{max_retries}.")
            time.sleep(retry_delay)
            retry_delay *= 2
        
        except Exception as e:
            print(f"❌ Ootamatu viga sõnaga {word}, tähendus {definition}: {e}")
            return None
            
    print(f"❌ Sõna '{word}' töötlemine ebaõnnestus pärast {max_retries} katset. Liigun edasi.")
    return None


# --- Põhiprogramm ---
def main():
    # See funktsioon ei vaja muudatusi
    all_rows = []
    
    input_file = "sisend.tsv"
    
    if not os.path.exists(input_file):
        print(f"⛔ Sisend fail '{input_file}' puudub!")
        return
    
    with open(input_file, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        try:
            next(reader, None)
        except StopIteration:
            pass
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
            print(f"⚠️ Lisame tühja rea järjekorra säilitamiseks")
            all_rows.append({
                "Sõna": word, "Tähendus": definition, "Tekstiregister": "ei määratletud",
                "Registri põhjendus": "viga töötlemisel", "Treeningandmete põhjendus": "ei saadaval",
                "Tähenduste arv kokku": 0, "Sagedus": "ei määratletud",
                "Näited": "ei saadaval", "Registrimärk": "ei kohaldu",
                "Märgendi põhjendus": "ei saadaval"
            })
        
        time.sleep(0.5)

    fieldnames = [
        "Sõna", "Tähendus", "Tekstiregister", "Registri põhjendus", 
        "Treeningandmete põhjendus", "Tähenduste arv kokku", "Sagedus", 
        "Näited", "Registrimärk", "Märgendi põhjendus"
    ]

    with open(FINAL_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";", quoting=csv.QUOTE_ALL)
        writer.writeheader()
        if all_rows:
            writer.writerows(all_rows)

    print(f"\n✅ Lõplik fail salvestatud: {FINAL_CSV}")

if __name__ == "__main__":
    main()
#Kood registrite töörühma 3. katse tarvis. Eesmärk on Anthropicu mudelile kaasa anda korpusest andmed, mida ta analüüsima peab. Sõna antakse ilma tähenduseta.
#Kui kontekst on liiga suur, siis see vektoriseeritakse.
#Autor: Eleri Aedmaa

import os
import csv
import anthropic
import pickle
import faiss
import tiktoken
import pandas as pd
import re
import time
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# --- Konfiguratsioon ---
client = anthropic.Anthropic()
MODEL = "claude-3-7-sonnet-20250219"  # Claude Sonnet 3.7
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
    response = client.messages.create(
        model=MODEL,
        system=prompt,
        messages=[
            {"role": "user", "content": context}
        ],
        max_tokens=16000,
        temperature=0.1
    )
    return response.content[0].text

def sanitize_filename(text):
    return re.sub(r'[<>:"/\\|?*]', '_', text)[:100]

# --- Prompt sõna analüüsimiseks ---
def create_analysis_prompt(word: str):
    return f"""Oled eesti keele sõnaraamatu koostaja. Sinu ülesanne on analüüsida sõna „{word}" kasutust etteantud tekstimaterjalis ja otsustada, kas selle tähendustele tuleks lisada registrimärgend.

Vasta järgmistele küsimustele, tuginedes ainult etteantud materjalile:

1. Nimeta sõna „{word}" kõik tähendused, mida etteantud tekstides näed. Ära erista alammõisteid erinevateks tähendusteks (näiteks „alukad" ei tähenda eraldi „aluspesu" ja „vanaema aluspükse", vaid üksnes „aluspesu").

2. Nimeta sõna „{word}" erinevate tähenduste arv.

3. Iga tähenduse juurde lisa, kas sõna on selles tähenduses sage, keskmine või harv. Sagedusrühm vali võrdluses sõna teiste tähendustega.

4. Too iga tähenduse kohta etteantud materjalist 3 näitelauset. Kui neid on andmetes vähem, siis too nii palju, kui leidub.

5. Otsusta sõna iga tähenduse kohta, kas seda kasutatakse pigem informaalsetes või neutraalsetes/formaalsetes tekstides? Kui sa ei oska eristust teha või see ei tule selgelt esile, siis ütle, et „ei kohaldu". Palun põhjenda oma valikut 5-10 lausega.

6. Kui valid „ei kohaldu", siis ja ainult siis vaata enda treeningandmetesse ja otsusta selle põhjal. Palun põhjenda oma valikut 5-10 lausega.

7. Kui mõnda tähendust kasutatakse pigem informaalsetes tekstides, siis vali sellele sobiv registrimärgend järgmistest:
• halvustav (näiteks ajuhälvik, debiilik, inimrämps)
• harv (näiteks ahvatama, mõistamisi, siinap). Märgend „harv" vali iga kord, kui tähendust leidub etteantud tekstimaterjalis vähe.
• kõnekeelne (näiteks igastahes, nokats, ära flippima)
• lastekeelne (näiteks jänku, kätu, nuku)
• luulekeelne (näiteks ehavalu, koidukuld, meeleheit)
• murdekeelne (näiteks hämmelgas, jõõrdlik, kidelema)
• rahvakeelne (näiteks heinakuu, viinakuu, männiseen)
• stiilitundlik (näiteks armastet, kirjutet, seitung)
• unarsõna (näiteks absurdum, ööp)
• vananenud (näiteks automobiil, drogist)
• vulgaarne (näiteks hoorapoeg, koinima, munn)

Iga valiku korral põhjenda 5-10 lausega, miks just see märgend sobib. Igal informaalsel tähendusel peab olema vähemalt üks märgend. Kui sobib mitu, too mitu.

OLULINE: Pärast küsimustele vastamist anna oma vastused TÄPSELT järgmises struktureeritud formaadis parsimiseks. Kasuta erinevat eraldajat (§§§) iga tähenduse andmete vahel:

--- STRUKTUREERITUD VASTUS ALGAB ---
TÄHENDUSED: kodukits§§§Hiina sodiaagimärk§§§halvustav termin politseinikule
TÄHENDUSTE-ARV: 3
SAGEDUSED: kodukits-sage§§§Hiina sodiaagimärk-keskmine§§§halvustav termin politseinikule-harv
NÄITED: kodukits-Kits on väike lehmake|Kitsed on intelligentsed loomad|Kits talub äärmuslikke tingimusi§§§Hiina sodiaagimärk-Kits on sodiaagi kaheksas loom|Kits on sitke ja järjekindel§§§halvustav termin politseinikule-Miks politseinikud kitsed on|Kriminaalidele tulebki kitse panna
REGISTRID: kodukits-neutraalsetes-formaalsetes§§§Hiina sodiaagimärk-neutraalsetes-formaalsetes§§§halvustav termin politseinikule-informaalsetes
REGISTRI-PÕHJENDUSED: kodukits-Kodukitse tähendus on neutraalne ja informatiivne. Seda kasutatakse loomakasvatus- ja põllumajandusalases kirjanduses. Terminoloogia on objektiivne ja faktiline. Sõnakasutus on ametlik ja teaduslik. Kontekst on tavaliselt hariduslik või informatsioonialane§§§Hiina sodiaagimärk-Horoskoobid on neutraalsed kultuuritekstid. Need kuuluvad populaarkultuuri valdkonda. Stiil on kirjeldav ja informeeriv. Kasutus on üldlevinud meedias. Puudub emotsionaalne värvitud sõnakasutus§§§halvustav termin politseinikule-Kasutatakse vaenulikus või kriitilises kontekstis. Sõnakasutus on emotsionaalselt laetud. Esineb tavaliselt konfliktides või protestides. Väljendab negatiivset suhtumist võimuesindajatesse. Kontekst on sageli poliitiliselt pingeline
MÄRGENDID: kodukits-ei-kohaldu§§§Hiina sodiaagimärk-ei-kohaldu§§§halvustav termin politseinikule-halvustav
MÄRGENDITE-PÕHJENDUSED: kodukits-ei-kohaldu§§§Hiina sodiaagimärk-ei-kohaldu§§§halvustav termin politseinikule-Selgelt negatiivne suhtumine politseinike suhtes. Sõna kannab halvustavat konnotatsiooni. Kasutatakse derogatiivses mõttes. Väljendab põlgust või vihkamist. Eesmärk on alandada ja maha teha. Kuulub konflikti- ja protestikeele hulka
--- STRUKTUREERITUD VASTUS LÕPEB ---"""

# --- Parandatud parsimise funktsioon ---
def parse_analysis_response(txt: str, word: str) -> List[Dict[str, Any]]:
    """
    Tagastab listi, kus iga element on üks tähendus koos kõigi andmetega
    """
    results = []
    
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
        
        # Parsime tähendused (kasutame §§§ eraldajat)
        tähendused = []
        if 'TÄHENDUSED' in data:
            tähendused = [t.strip() for t in data['TÄHENDUSED'].split('§§§') if t.strip()]
        
        # Parsime sagedused
        sagedused = []
        if 'SAGEDUSED' in data:
            sagedused_raw = data['SAGEDUSED'].split('§§§')
            for item in sagedused_raw:
                if '-' in item:
                    sagedused.append(item.split('-', 1)[1].strip())
                else:
                    sagedused.append(item.strip())
        
        # Parsime registrid
        registrid = []
        if 'REGISTRID' in data:
            registrid_raw = data['REGISTRID'].split('§§§')
            for item in registrid_raw:
                if '-' in item:
                    registrid.append(item.split('-', 1)[1].strip())
                else:
                    registrid.append(item.strip())
        
        # Parsime registri põhjendused (nüüd säilivad pikad tekstid)
        registri_põhjendused = []
        if 'REGISTRI-PÕHJENDUSED' in data:
            põhjendused_raw = data['REGISTRI-PÕHJENDUSED'].split('§§§')
            for item in põhjendused_raw:
                if '-' in item:
                    registri_põhjendused.append(item.split('-', 1)[1].strip())
                else:
                    registri_põhjendused.append(item.strip())
        
        # Parsime märgendid
        märgendid = []
        if 'MÄRGENDID' in data:
            märgendid_raw = data['MÄRGENDID'].split('§§§')
            for item in märgendid_raw:
                if '-' in item:
                    märgendid.append(item.split('-', 1)[1].strip())
                else:
                    märgendid.append(item.strip())
        
        # Parsime märgendite põhjendused (nüüd säilivad pikad tekstid)
        märgendite_põhjendused = []
        if 'MÄRGENDITE-PÕHJENDUSED' in data:
            märgendite_põhjendused_raw = data['MÄRGENDITE-PÕHJENDUSED'].split('§§§')
            for item in märgendite_põhjendused_raw:
                if '-' in item:
                    märgendite_põhjendused.append(item.split('-', 1)[1].strip())
                else:
                    märgendite_põhjendused.append(item.strip())
        
        # Parsime näited
        näited = []
        if 'NÄITED' in data:
            näited_raw = data['NÄITED'].split('§§§')
            for item in näited_raw:
                if '-' in item:
                    näited_part = item.split('-', 1)[1].strip()
                    näited.append(näited_part.replace('|', ' | '))
                else:
                    näited.append(item.strip())
        
        # Loome iga tähenduse jaoks eraldi kirje
        for i, tähendus in enumerate(tähendused):
            # Võtame andmed järjekorra alusel, aga kontrollime pikkust
            sagedus = sagedused[i] if i < len(sagedused) else "ei määratletud"
            näited_text = näited[i] if i < len(näited) else "ei leitud"
            register = registrid[i] if i < len(registrid) else "ei määratletud"
            registri_põhjendus = registri_põhjendused[i] if i < len(registri_põhjendused) else "ei määratletud"
            märgendid_text = märgendid[i] if i < len(märgendid) else "ei-kohaldu"
            märgendite_põhjendus = märgendite_põhjendused[i] if i < len(märgendite_põhjendused) else "ei-kohaldu"
            
            # Jagame märgendid komaga (kui on mitu)
            märgendid_list = []
            if märgendid_text and märgendid_text != "ei-kohaldu":
                märgendid_list = [m.strip() for m in märgendid_text.split(',') if m.strip()]
            
            # Kui märgendeid pole või on "ei-kohaldu", teeme ühe rea
            if not märgendid_list or märgendid_text == "ei-kohaldu":
                result = {
                    "Sõna": word,
                    "Tähenduse nr": i + 1,
                    "Tähendus": tähendus,
                    "Tähenduste arv kokku": data.get('TÄHENDUSTE-ARV', str(len(tähendused))),
                    "Sagedus": sagedus,
                    "Näited": näited_text,
                    "Tekstiregister": register,
                    "Registri põhjendus": registri_põhjendus,
                    "Treeningandmete põhjendus": "ei-kohaldu",
                    "Registrimärk": "ei-kohaldu",
                    "Märgendi põhjendus": "ei-kohaldu"
                }
                results.append(result)
            else:
                # Iga märgendi jaoks eraldi rida
                for j, märgend in enumerate(märgendid_list):
                    # Kui on mitu märgendit, kasutame sama põhjendust kõigi jaoks
                    # (kuna märgendite põhjendused on tihti ühe tähenduse kohta koos)
                    märgendi_põhjendus = märgendite_põhjendus if märgendite_põhjendus != "ei-kohaldu" else "ei leitud"
                    
                    result = {
                        "Sõna": word,
                        "Tähenduse nr": i + 1,
                        "Tähendus": tähendus,
                        "Tähenduste arv kokku": data.get('TÄHENDUSTE-ARV', str(len(tähendused))),
                        "Sagedus": sagedus,
                        "Näited": näited_text,
                        "Tekstiregister": register,
                        "Registri põhjendus": registri_põhjendus,
                        "Treeningandmete põhjendus": "ei-kohaldu",
                        "Registrimärk": märgend.strip(),
                        "Märgendi põhjendus": märgendi_põhjendus
                    }
                    results.append(result)
            
            print(f"   ✅ Tähendus {i+1}: {tähendus}")
            print(f"      📈 Sagedus: {sagedus}")
            print(f"      📊 Register: {register}")
            print(f"      🔍 Registri põhjendus: {registri_põhjendus[:100]}{'...' if len(registri_põhjendus) > 100 else ''}")
            print(f"      🏷️ Märgendid: {märgendid_text}")
            if märgendite_põhjendus != "ei-kohaldu":
                print(f"      📝 Märgendi põhjendus: {märgendite_põhjendus[:100]}{'...' if len(märgendite_põhjendus) > 100 else ''}")
    
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
            "Treeningandmete põhjendus": "ei-kohaldu",
            "Registrimärk": "ei-kohaldu",
            "Märgendi põhjendus": "ei-kohaldu"
        })
    
    # Kui mingil põhjusel ei ole tulemusi, lisa vähemalt üks tühi kirje
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
            "Treeningandmete põhjendus": "ei-kohaldu",
            "Registrimärk": "ei-kohaldu",
            "Märgendi põhjendus": "ei-kohaldu"
        })
    
    return results

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
        parsed_results = parse_analysis_response(reply, word)
        
        # Prindime parsimise tulemuse
        print(f"\n📊 PARSITUD TULEMUS:")
        print(f"   📝 Tähendusi kokku: {len(parsed_results)}")
        for result in parsed_results:
            print(f"   {result['Tähenduse nr']}. {result['Tähendus'][:50]}{'...' if len(result['Tähendus']) > 50 else ''}")
            print(f"      📈 Sagedus: {result['Sagedus']}")
            print(f"      📋 Register: {result['Tekstiregister']}")
            print(f"      🏷️ Märgend: {result['Registrimärk']}")
        
        print(f"✅ {word} — Analüüs lõpetatud\n")
        
        return parsed_results

    except Exception as e:
        print(f"❌ Viga sõnaga {word}: {e}")
        return None

# --- Põhiprogramm ---
def main():
    all_rows = []
    
    # Loeme sisend_2.tsv faili
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
            all_rows.extend(result)  # Lisame kõik tähendused
        else:
            # Lisa tühi rida järjekorra säilitamiseks
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
                "Treeningandmete põhjendus": "ei saadaval",
                "Registrimärk": "ei kohaldu",
                "Märgendi põhjendus": "ei saadaval"
            })
        
        time.sleep(0.5)  # Väike paus

    # Salvesta CSV
    fieldnames = [
        "Sõna", "Tähenduse nr", "Tähendus", "Tähenduste arv kokku", "Sagedus", 
        "Näited", "Tekstiregister", "Registri põhjendus", "Treeningandmete põhjendus",
        "Registrimärk", "Märgendi põhjendus"
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
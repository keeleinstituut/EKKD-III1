#Kood, mis palub keelemudelil (Claude Opus 4.1) leida ÜSis esitatud tähendusele vasted eelnevalt mudeli antud tähendustele.
#Autor: Eleri Aedmaa


import anthropic
import csv
import os
from pathlib import Path

# API võti (pane oma võti siia või kasuta keskkonna muutujat)
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "key_here")


def read_csv(filepath, encoding='utf-8'):
    """Loeb CSV faili"""
    # Proovin erinevaid kodeeringuid
    encodings = [encoding, 'utf-8', 'cp1252', 'windows-1252', 'iso-8859-1', 'latin1']
    
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                # Proovin erinevaid eraldajaid
                sample = f.read(1024)
                f.seek(0)
                
                # Määran eraldaja (; või ,)
                delimiter = ';' if ';' in sample else ','
                
                reader = csv.DictReader(f, delimiter=delimiter)
                data = list(reader)
                
                # Eemaldan BOM märgi, kui on
                if data and list(data[0].keys())[0].startswith('\ufeff'):
                    first_key = list(data[0].keys())[0]
                    new_key = first_key.replace('\ufeff', '')
                    for row in data:
                        row[new_key] = row.pop(first_key)
                
                print(f"  ✓ Loetud {filepath} kodeeringuga: {enc}, eraldaja: '{delimiter}'")
                return data
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    # Kui ükski ei tööta, proovi utf-8 errors='ignore'
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        delimiter = ';' if ';' in f.read(1024) else ','
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delimiter)
        data = list(reader)
        print(f"  ⚠ Loetud {filepath} UTF-8-ga (mõned märgid vahele jäetud), eraldaja: '{delimiter}'")
        return data

def find_matches_with_claude(pohifail_path, teise_faili_path, output_path, teise_faili_veerg='_1'):
    """
    Leiab põhifaili sõnadele vasted teisest failist kasutades Claude API-d
    
    Args:
        pohifail_path: põhifaili tee (YS_t2hendused.csv)
        teise_faili_path: teise faili tee (kus otsida vasteid)
        output_path: väljundfaili tee
        teise_faili_veerg: veeru nimi, kust otsida vasteid (default: '_1')
    """
    
    # Loen failid
    pohifail = read_csv(pohifail_path)
    teine_fail = read_csv(teise_faili_path)
    
    # DEBUG: Näitan veergude nimesid
    if pohifail:
        print(f"  Põhifaili veerud: {list(pohifail[0].keys())}")
    if teine_fail:
        print(f"  Teise faili veerud: {list(teine_fail[0].keys())}")
    
    # Eemaldan päise rea teisest failist (kui on)
    if teine_fail and teine_fail[0].get('', '') == 'katsesõna':
        teine_fail = teine_fail[1:]
    
    client = anthropic.Anthropic(api_key=API_KEY)
    
    results = []
    total = len(pohifail)
    
    for idx, row in enumerate(pohifail, 1):
        word = row['word']
        definition = row['definition']
        
        print(f"[{idx}/{total}] Töötlen: {word}")
        
        # Otsin teisest failist sõna järgi (esimene veerg peab matchima 100%)
        matching_rows = [r for r in teine_fail if r.get('katsesõna', '').strip() == word or r.get('', '').strip() == word]
        
        # DEBUG: Näitan, mitu vastet leiti
        if idx <= 3:  # Näitan esimeste 3 kohta detaile
            print(f"  DEBUG: Leitud {len(matching_rows)} rida teisest failist")
        
        if not matching_rows:
            results.append({
                'word': word,
                'definition': definition,
                'vaste': 'ei leidu'
            })
            continue
        
        # Kogun kõik võimalikud tähendused teisest failist
        candidates = [r.get(teise_faili_veerg, '') for r in matching_rows if r.get(teise_faili_veerg)]
        
        # DEBUG
        if idx <= 3:
            print(f"  DEBUG: Kandidaate: {len(candidates)}")
            if candidates:
                print(f"  DEBUG: Esimene kandidaat: {candidates[0][:100]}...")
        
        if not candidates:
            results.append({
                'word': word,
                'definition': definition,
                'vaste': 'ei leidu'
            })
            continue
        
        # Kui on ainult üks kandidaat, võta see kohe
        if len(candidates) == 1:
            results.append({
                'word': word,
                'definition': definition,
                'vaste': candidates[0]
            })
            print(f"  ✓ Ainult üks kandidaat, võetud automaatselt")
            continue
        
        # Küsin Claude'ilt parimat vastet
        prompt = f"""Oled andmekontrollija. Sul on sõna "{word}" tähendusega "{definition}"

Kandidaadid teisest failist:
{chr(10).join(f"{i+1}. {c}" for i, c in enumerate(candidates))}

OLULINE: Vasta AINULT kandidaadi numbriga või sõnaga "puudub"!

Vali kandidaat, mis on tähenduslikult kõige lähedasem sõna esitatud tähendusele.
Kui ükski ei sobi (sarnasus alla 30%), kirjuta "puudub".

Vasta formaat:
[number või "puudub"]

Näiteks: 2
Või: puudub"""

        try:
            message = client.messages.create(
                model="claude-opus-4-1-20250805",
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            response = message.content[0].text.strip()
            
            # DEBUG
            if idx <= 3:
                print(f"  DEBUG: Claude vastus: {response}")
            
            # Parsin vastust - võtan AINULT numbri või "puudub"
            vaste_result = "ei leidu"
            
            # Eemaldan kõik mittevajalikud read ja märgid
            response_clean = response.replace('VASTE:', '').replace('[', '').replace(']', '').strip()
            
            # Võtan esimese rea
            first_line = response_clean.split('\n')[0].strip()
            
            if first_line.lower() in ['puudub', 'ei leidu', 'ei']:
                vaste_result = "ei leidu"
            elif first_line.isdigit():
                idx_vaste = int(first_line) - 1
                if 0 <= idx_vaste < len(candidates):
                    # VÕTAN TÄPSE ORIGINAALTEKSTI
                    vaste_result = candidates[idx_vaste]
                else:
                    vaste_result = "ei leidu"
            else:
                # Kui vastus pole number ega "puudub", provin leida numbrit tekstist
                import re
                numbers = re.findall(r'\d+', first_line)
                if numbers:
                    idx_vaste = int(numbers[0]) - 1
                    if 0 <= idx_vaste < len(candidates):
                        vaste_result = candidates[idx_vaste]
            
            results.append({
                'word': word,
                'definition': definition,
                'vaste': vaste_result
            })
            
        except Exception as e:
            print(f"  VIGA: {e}")
            results.append({
                'word': word,
                'definition': definition,
                'vaste': f'VIGA: {str(e)}'
            })
    
    # Salvestan tulemused
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['word', 'definition', 'vaste']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Statistika
    leitud = sum(1 for r in results if r['vaste'] != 'ei leidu' and not r['vaste'].startswith('VIGA'))
    
    print(f"\n=== VALMIS ===")
    print(f"Kokku: {len(results)} rida")
    print(f"Vasteid leitud: {leitud}")
    print(f"Tulemused salvestatud: {output_path}")



if __name__ == "__main__":
    # Kasutamine
    find_matches_with_claude(
        pohifail_path="YS_t2hendused.csv",
        teise_faili_path="claude_t2h.csv",
        output_path="tulemused_claude_api.csv",
        teise_faili_veerg="Tähendus"  # Muuda seda vastavalt veeru nimele
    )
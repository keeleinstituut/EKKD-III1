#EKKD-III1 tähenduste töörühma päringukood, et promptida Anthropicu mudeleid.

#Küsimus puudutab erinevat tüüpi definitsioonide vormistamist. 
#Autor: Eleri Aedmaa

import anthropic
import csv
import pandas as pd

def read_inputs_from_file(file_path):
    try:
        return pd.read_csv(file_path, delimiter='\t', usecols=['Katsesõna'])
    except Exception as e:
        raise ValueError(f"Sisendfaili lugemisel tekkis viga: {e}")

def get_response_for_input(api_key, word, meaning):
    client = anthropic.Client(api_key=api_key)
    response = client.messages.create(
        model="claude-3-opus-latest", #muuda vajadusel
        system=f"Sa oled eesti keele sõnaraamatu koostaja. Kirjuta sõnale '{word}' kolm definitsiooni, kus seletad, mida see sõna eesti keeles tähendab. "
               "Esimene seletus kirjuta samas stiilis nagu definitsioon ÜS, "
               "teine seletus kirjuta samas stiilis nagu definitsioon ÕS ja "
               "kolmas seletus kirjuta samas stiilis nagu definitsioon KÕ. Järgi näidete stiili:\n\n"
               "**huumor**\n"
               "definitsioon ÜS - heatahtlik nali, koomiliste elunähtuste, sündmuste või inimeste puuduste ja nõrkuste heatahtlik naeruvääristamine\n"
               "definitsioon ÕS - heatahtlik nali\n"
               "definitsioon KÕ - see, kui miski on naljakas\n\n"
               "**kaste**\n"
               "definitsioon ÜS - vedel või poolvedel lisand, mis muudab toidu mahlasemaks ja maitsvamaks\n"
               "definitsioon ÕS - soust\n"
               "definitsioon KÕ - paks vedelik, mida süüakse koos muu toiduga\n\n"
               "**erinev**\n"
               "definitsioon ÜS - mingil viisil teistest, muust eristuv\n"
               "definitsioon ÕS - muust eristuv\n"
               "definitsioon KÕ - (kellegagi, millegagi või omavahel võrreldes) teistsugune\n\n"
               "**eeldama**\n"
               "definitsioon ÜS - eeltingimusena mingit omadust, oskust vm asjaolu vajama või ootama\n"
               "definitsioon ÕS - eeldusena nõudma\n"
               "definitsioon KÕ - tingimusena millegi olemasolu vajama, eeldusena nõudma\n\n"
               "**juures**\n"
               "definitsioon ÜS - kellegi või millegi vahetus läheduses, hästi lähedal\n"
               "definitsioon ÕS - vahetus läheduses\n"
               "definitsioon KÕ - (millegi) lähedal\n\n"
               "Vastus peab olema järgmisel kujul:\n"
               "definitsioon ÜS - [ÜS stiilis seletus]\n"
               "definitsioon ÕS - [ÕS stiilis seletus]\n"
               "definitsioon KÕ - [KÕ stiilis seletus]",
        messages=[{
            "role": "user",
            "content": f"Sõna: {word}"
        }],
        temperature=0.2,
        max_tokens=4096
    )
    print(f"Claude vastus sõnale '{word}': {response.content}")
    return response.content

def process_response(api_key, word):
    try:
        response = get_response_for_input(api_key, word, None)
        text = response[0].text if isinstance(response, list) else response
        lines = text.split('\n')
        
        us_def = ""
        os_def = ""
        ko_def = ""
        
        for line in lines:
            if line.startswith("definitsioon ÜS -"):
                us_def = line.replace("definitsioon ÜS -", "").strip()
            elif line.startswith("definitsioon ÕS -"):
                os_def = line.replace("definitsioon ÕS -", "").strip()
            elif line.startswith("definitsioon KÕ -"):
                ko_def = line.replace("definitsioon KÕ -", "").strip()
                
        return pd.Series([us_def, os_def, ko_def])
    except Exception as e:
        return pd.Series(["Viga", f"Töötlemise viga: {e}", ""])

def main():
    api_key = ""
    input_file_path = 'ÜSÕSKÕ_sisend.csv'
    output_file_path = 'ÜSÕSKÕ_väljund_claude3.csv'
    
    try:
        user_inputs = read_inputs_from_file(input_file_path)
        
        if 'Katsesõna' not in user_inputs.columns:
            raise ValueError("Sisendfail peab sisaldama veergu 'Katsesõna'.")
            
        user_inputs[['ÜS_definitsioon', 'ÕS_definitsioon', 'KÕ_definitsioon']] = user_inputs.apply(
            lambda row: process_response(api_key, row['Katsesõna']),
            axis=1
        )
        
        user_inputs.to_csv(output_file_path, index=False, quoting=csv.QUOTE_MINIMAL, sep='\t')
        print(f"Tulemused salvestatud faili: {output_file_path}")
    except Exception as e:
        print(f"Tekkis viga: {e}")

if __name__ == "__main__":
    main()

#Kood EKKD-III1 registrite töörühma teise katse päringute tegemiseks Anthropicu mudelilt Claude 3.7 Sonnet mõtlemise režiimis.
#Autor: Eleri Aedmaa

import pandas as pd
import csv
import requests
import time
import json

def read_inputs_from_file(file_path):
    try:
        return pd.read_csv(file_path, delimiter='\t', usecols=['Katsesõna', 'Tähendus'])
    except Exception as e:
        raise ValueError(f"Sisendfaili lugemisel tekkis viga: {e}")

def get_response_for_input(api_key, word, meaning):
    # Otsene HTTP päring Anthropic API-sse
    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    # Lisame väikese ajalise viivituse, et vältida API limiitide ületamist
    time.sleep(1)
    
    data = {
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": 32000,
        "temperature": 1,
        "thinking": {
            "type": "enabled", 
            "budget_tokens": 16000
        },
        "system": f"Oled eesti keele sõnaraamatu koostaja ja pead otsustama, kas sõnale/väljendile on vaja lisada registrimärgend. "
            f"Kas eesti(keelset) sõna '{word}' tähenduses '{meaning}' kasutatakse pigem [informaalsetes, neutraalsetes/formaalsetes] tekstides? "
            "Kui sa ei oska eristust teha või see ei tule selgelt esile, siis ütle, et 'ei kohaldu'. "
            "Informaalsed tekstid on teiste seas näiteks blogid, foorumid, kommentaariumid, chativestlused, sotsiaalmeedia tekstid, trükivigasid täis tekstid, vahel ka raamatutegelaste otsekõne. "
            "Palun põhjenda oma valikut. Lähtu vastates ainult oma treeningandmetest, mitte välisotsingutest ja välistest andmebaasidest (sh EKI sõnastikest)."
            "Vastus peab olema täpselt järgmisel kujul ilma lisakommentaarideta:\n"
            "Kasutus: [informaalsetes / neutraalsetes/formaalsetes / ei kohaldu]\n"
            "Põhjendus: [Selgitus kasutuse kohta]",
        "messages": [{
            "role": "user",
            "content": f"Sõna: {word}\nTähendus: {meaning}"
        }]
    }
    
    try:
        response = requests.post("https://api.anthropic.com/v1/messages", 
                                headers=headers, 
                                json=data)
        
        if response.status_code != 200:
            print(f"API viga: {response.status_code} - {response.text}")
            return None, None
        
        # Salvesta vastus
        full_response = response.json()
        
        thinking_text = None
        response_text = None
        
        # Eraldame thinking ja teksti vastuse
        if 'content' in full_response and isinstance(full_response['content'], list):
            for content_item in full_response['content']:
                if content_item.get('type') == 'thinking' and 'thinking' in content_item:
                    thinking_text = content_item['thinking']
                elif content_item.get('type') == 'text' and 'text' in content_item:
                    response_text = content_item['text']
        
        return response_text, thinking_text
    except Exception as e:
        print(f"API päringu viga: {e}")
        time.sleep(5)  # Ootame natuke enne uuesti proovimist
        return None, None

def parse_structured_response(response_text):
    """Parsib struktureeritud vastuse väljad"""
    if not response_text:
        return "Määramata", "Põhjendus puudub"
    
    usage = "Määramata"
    explanation = "Põhjendus puudub"
    
    lines = response_text.strip().split('\n')
    for line in lines:
        if line.lower().startswith("kasutus:"):
            usage = line.replace("Kasutus:", "", 1).strip()
        elif line.lower().startswith("põhjendus:"):
            explanation = line.replace("Põhjendus:", "", 1).strip()
    
    return usage, explanation

def main():
    api_key = ""
    input_file_path = 'ekkd_i_k6nek_6s_think.csv'
    output_file_path = 'ekkd_i_k6nek_6s_väljund_claude3.7sonnet_thinking.csv'
    
    try:
        user_inputs = read_inputs_from_file(input_file_path)
        print(f"Laeti {len(user_inputs)} rida andmeid")
        
        # Loome tühja dataframe'i tulemustele
        result_df = user_inputs.copy()
        result_df['Kasutus'] = ""
        result_df['Põhjendus'] = ""
        result_df['mõttekäik'] = ""
        
        # Töötleme iga rida eraldi
        for idx, row in user_inputs.iterrows():
            if idx % 5 == 0:
                print(f"Töötleme rida {idx+1}/{len(user_inputs)}")
            
            word = row['Katsesõna']
            meaning = row['Tähendus']
            
            try:
                response_text, thinking_text = get_response_for_input(api_key, word, meaning)
                
                # Parsime struktureeritud vastuse
                usage, explanation = parse_structured_response(response_text)
                
                # Salvestame tulemused
                result_df.at[idx, 'Kasutus'] = usage
                result_df.at[idx, 'Põhjendus'] = explanation
                
                # Teisendame mõttekäigu ühele reale, asendades reavahetused
                one_line_thinking = thinking_text.replace('\n', ' ') if thinking_text else ""
                result_df.at[idx, 'mõttekäik'] = one_line_thinking
                
                # Viivitus on juba lisatud get_response_for_input funktsioonis
                
            except Exception as e:
                print(f"Viga rea {idx} töötlemisel: {e}")
                result_df.at[idx, 'Kasutus'] = "Viga"
                result_df.at[idx, 'Põhjendus'] = f"{e}"
                result_df.at[idx, 'mõttekäik'] = ""
            
            # Vahesalvestus iga 10 rea järel
            if idx % 10 == 0 and idx > 0:
                result_df.to_csv(f"{output_file_path}.temp", index=False, quoting=csv.QUOTE_MINIMAL, sep='\t')
                print(f"Vahesalvestus tehtud: {idx+1}/{len(user_inputs)}")
        
        # Lõplik salvestus
        result_df.to_csv(output_file_path, index=False, quoting=csv.QUOTE_MINIMAL, sep='\t')
        print(f"Tulemused salvestatud faili: {output_file_path}")
    except Exception as e:
        print(f"Tekkis viga: {e}")
        
if __name__ == "__main__":
    main()
#Kood EKKD-III1 registrite töörühma teise katse päringute tegemiseks Google'i mudelilt Gemini 1.5 Pro.
#Autor: Eleri Aedmaa

import os
import csv
import time
import pandas as pd
import google.generativeai as genai
from google.api_core import exceptions

# Seadista API võti (ära pane seda otse koodi sisse produktsioonis!)
os.environ["GOOGLE_API_KEY"] = ""  # Asenda oma API võtmega
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def read_inputs_from_file(file_path):
    try:
        return pd.read_csv(file_path, delimiter='\t', usecols=['Katsesõna', 'Tähendus'])
    except Exception as e:
        raise ValueError(f"Sisendfaili lugemisel tekkis viga: {e}")

def get_response_for_input(word, meaning):
    model = genai.GenerativeModel("models/gemini-1.5-pro") # Kasuta soovitud mudelit

    prompt = f"""Oled eesti keele sõnaraamatu koostaja, kelle ülesandeks on määrata, kas sõnale või väljendile tuleks lisada registrimärgend.
            Kas eesti(keelset) sõna '{word}' tähenduses '{meaning}' kasutatakse pigem [informaalsetes, neutraalsetes/formaalsetes] registrites? 
            Kui sa ei oska eristust teha või see ei tule selgelt esile, siis ütle, et 'ei kohaldu'. 
            Informaalsed registrid on teiste seas näiteks blogid, foorumid, kommentaariumid, chativestlused, sotsiaalmeedia tekstid, trükivigasid täis tekstid, vahel ka raamatutegelaste otsekõne.
            Palun põhjenda oma valikut. Lähtu vastates ainult oma treeningandmetest, mitte välisotsingutest ja andmebaasidest. 
            Kui sõna '{word}' kasutatakse pigem informaalsetes registrites, siis mis on sõna '{word}' neutraalsed/formaalsed sünonüümid eesti keeles? 
            Kui sõna kasutatakse pigem neutraalsetes/formaalsetes registrites, siis vasta 'ei kohaldu'. 
            Vastus peab olema järgmisel kujul:
            Kasutus: [informaalsetes / neutraalsetes/formaalsetes / ei kohaldu]
            Põhjendus: [Selgitus kasutuse kohta]
            Sünonüümid: [Sünonüümid või 'ei kohaldu']"""

    response = model.generate_content(prompt)
    return response.text

def process_response(word, meaning):
    retries = 0
    max_retries = 5
    retry_delay = 60  # Algne viivitus sekundites

    while retries < max_retries:
        try:
            time.sleep(retries * retry_delay) # Viivitus enne päringut
            response = get_response_for_input(word, meaning)
            text = response
            lines = text.split('\n')
            
            category = ""
            explanation = ""
            synonyms = ""
            
            for line in lines:
                if line.startswith("Kasutus:"):
                    category = line.replace("Kasutus:", "").strip()
                elif line.startswith("Põhjendus:"):
                    explanation = line.replace("Põhjendus:", "").strip()
                elif line.startswith("Sünonüümid:"):
                    synonyms = line.replace("Sünonüümid:", "").strip()
                    
            return pd.Series([category, explanation, synonyms])
        except exceptions.ResourceExhausted:
            retries += 1
            if retries < max_retries:
              print(f"Kvoot on ületatud. Ootan {retries * retry_delay} sekundit ja proovin uuesti (katse {retries}/{max_retries}).")
            else:
              print(f"Kvoot on ületatud mitu korda. Loobun sõnast '{word}'.")
              return pd.Series(["Viga", f"Kvoodi viga: Liiga palju katseid", ""])

        except Exception as e:
            print(f"Viga: {e}")
            return pd.Series(["Viga", f"Töötlemise viga: {e}", ""])
    

def main():
    input_file_path = 'katse2_sisend2.csv'
    output_file_path = 'katse2_prompt3_väljund_gemini1.5pro.csv'

    try:
        user_inputs = read_inputs_from_file(input_file_path)
        
        if 'Katsesõna' not in user_inputs.columns or 'Tähendus' not in user_inputs.columns:
            raise ValueError("Sisendfail peab sisaldama veerge 'Katsesõna' ja 'Tähendus'.")
        user_inputs[['vastus', 'põhjendus', 'sünonüümid']] = user_inputs.apply(
            lambda row: process_response(row['Katsesõna'], row['Tähendus']),
            axis=1
        )

        user_inputs.to_csv(output_file_path, index=False, quoting=csv.QUOTE_MINIMAL, sep='\t')
        print(f"Tulemused salvestatud faili: {output_file_path}")
    except Exception as e:
        print(f"Tekkis viga: {e}")
        
if __name__ == "__main__":
    main()
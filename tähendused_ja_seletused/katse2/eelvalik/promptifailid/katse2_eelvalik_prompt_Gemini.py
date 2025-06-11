#EKKD-III1 tähenduste töörühma teise katse päringukood, et promptida Google'i mudeleid.
#Mudelit juhendatakse genereerima 5 näite põhjal ÜSi stiilis definitsioon. 
#Autor: Eleri Aedmaa

import os
import csv
import time
import pandas as pd
import google.generativeai as genai
from google.api_core import exceptions

# API võtme seadistamine
os.environ["GOOGLE_API_KEY"] = ""  # Asenda oma API võtmega
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def read_inputs_from_file(file_path):
    try:
        return pd.read_csv(file_path, delimiter='\t', usecols=['Katsesõna'])
    except Exception as e:
        raise ValueError(f"Sisendfaili lugemisel tekkis viga: {e}")

def get_response_for_input(word):
    model = genai.GenerativeModel("models/gemini-2.5-pro-preview-03-25") #mudeli määramine

    prompt = f"""Sa oled eesti keele sõnaraamatu koostaja. Kirjuta sõnale '{word}' ÜSi stiilis definitsioon, kus seletad, mida see sõna eesti keeles tähendab. Lähtu allolevatest näidetest.

**huumor**
ÜSi definitsioon - heatahtlik nali, koomiliste elunähtuste, sündmuste või inimeste puuduste ja nõrkuste heatahtlik naeruvääristamine

**kaste**
ÜSi definitsioon - vedel või poolvedel lisand, mis muudab toidu mahlasemaks ja maitsvamaks

**erinev**
ÜSi definitsioon - mingil viisil teistest, muust eristuv

**eeldama**
ÜSi definitsioon - eeltingimusena mingit omadust, oskust vm asjaolu vajama või ootama

**juures**
ÜSi definitsioon - kellegi või millegi vahetus läheduses, hästi lähedal

Vastus peab olema järgmisel kujul:
{word} - [ÜSi definitsioon]
"""

    response = model.generate_content(prompt)
    return response.text

def process_response(word):
    retries = 0
    max_retries = 5
    retry_delay = 60

    while retries < max_retries:
        try:
            time.sleep(retries * retry_delay)
            response = get_response_for_input(word)
            lines = response.strip().split('\n')

            for line in lines:
                if '-' in line:
                    parts = line.split('-', 1)
                    if len(parts) == 2:
                        return pd.Series([parts[1].strip()])
            return pd.Series(["Määramata definitsioon"])
        except exceptions.ResourceExhausted:
            retries += 1
            if retries < max_retries:
                print(f"Kvoot ületatud. Ootan {retries * retry_delay} s ja proovin uuesti (katse {retries}/{max_retries}).")
            else:
                print(f"Kvoot ületatud mitu korda. Loobun sõnast '{word}'.")
                return pd.Series(["Kvoodi viga"])
        except Exception as e:
            print(f"Viga: {e}")
            return pd.Series([f"Töötlemise viga: {e}"])

def main():
    input_file_path = 'katse2_eelvalik_sisend.csv'
    output_file_path = 'katse2_eelvalik_väljund_gemini25pro.csv'

    try:
        user_inputs = read_inputs_from_file(input_file_path)

        if 'Katsesõna' not in user_inputs.columns:
            raise ValueError("Sisendfail peab sisaldama veergu 'Katsesõna'.")

        user_inputs[['ÜS_definitsioon']] = user_inputs.apply(
            lambda row: process_response(row['Katsesõna']),
            axis=1
        )

        user_inputs.to_csv(output_file_path, index=False, quoting=csv.QUOTE_MINIMAL, sep='\t')
        print(f"Tulemused salvestatud faili: {output_file_path}")
    except Exception as e:
        print(f"Tekkis viga: {e}")

if __name__ == "__main__":
    main()

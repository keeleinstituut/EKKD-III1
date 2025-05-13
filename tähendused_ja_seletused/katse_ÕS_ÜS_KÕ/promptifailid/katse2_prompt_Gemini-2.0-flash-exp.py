#EKKD-III1 tähenduste töörühma päringukood, et promptida Google Gemini 2.0 Flash mudelit.
#Küsimus puudutab erinevat tüüpi definitsioonide vormistamist. 
#Autor: Eleri Aedmaa

import os
import csv
import time
import pandas as pd
import google.generativeai as genai
from google.api_core import exceptions

os.environ["GOOGLE_API_KEY"] = ""
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def read_inputs_from_file(file_path):
    try:
        return pd.read_csv(file_path, delimiter='\t', usecols=['Katsesõna'])
    except Exception as e:
        raise ValueError(f"Sisendfaili lugemisel tekkis viga: {e}")

def get_response_for_input(word):
    model = genai.GenerativeModel("models/gemini-2.0-flash-exp")

    prompt = f"""Kirjuta sõnale '{word}' kolm definitsiooni, kus seletad, mida see sõna eesti keeles tähendab. 
        Esimene seletus kirjuta samas stiilis nagu definitsioon ÜS, 
        teine seletus kirjuta samas stiilis nagu definitsioon ÕS ja 
        kolmas seletus kirjuta samas stiilis nagu definitsioon KÕ. Järgi näidete stiili:

        **huumor**
        definitsioon ÜS - heatahtlik nali, koomiliste elunähtuste, sündmuste või inimeste puuduste ja nõrkuste heatahtlik naeruvääristamine
        definitsioon ÕS - heatahtlik nali
        definitsioon KÕ - see, kui miski on naljakas

        **kaste**
        definitsioon ÜS - vedel või poolvedel lisand, mis muudab toidu mahlasemaks ja maitsvamaks
        definitsioon ÕS - soust
        definitsioon KÕ - paks vedelik, mida süüakse koos muu toiduga

        **erinev**
        definitsioon ÜS - mingil viisil teistest, muust eristuv
        definitsioon ÕS - muust eristuv
        definitsioon KÕ - (kellegagi, millegagi või omavahel võrreldes) teistsugune

        **eeldama**
        definitsioon ÜS - eeltingimusena mingit omadust, oskust vm asjaolu vajama või ootama
        definitsioon ÕS - eeldusena nõudma
        definitsioon KÕ - tingimusena millegi olemasolu vajama, eeldusena nõudma

        **juures**
        definitsioon ÜS - kellegi või millegi vahetus läheduses, hästi lähedal
        definitsioon ÕS - vahetus läheduses
        definitsioon KÕ - (millegi) lähedal

        Vastus peab olema järgmisel kujul:
        definitsioon ÜS - [ÜS stiilis seletus]
        definitsioon ÕS - [ÕS stiilis seletus]
        definitsioon KÕ - [KÕ stiilis seletus]"""

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
            text = response
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
    output_file_path = 'katse2_väljund_gemini2.0flash-exp_II.csv'

    try:
        user_inputs = read_inputs_from_file(input_file_path)
        
        if 'Katsesõna' not in user_inputs.columns:
            raise ValueError("Sisendfail peab sisaldama veergu 'Katsesõna'.")
            
        user_inputs[['ÜS_definitsioon', 'ÕS_definitsioon', 'KÕ_definitsioon']] = user_inputs.apply(
            lambda row: process_response(row['Katsesõna']),
            axis=1
        )

        user_inputs.to_csv(output_file_path, index=False, quoting=csv.QUOTE_MINIMAL, sep='\t')
        print(f"Tulemused salvestatud faili: {output_file_path}")
    except Exception as e:
        print(f"Tekkis viga: {e}")
        
if __name__ == "__main__":
    main()
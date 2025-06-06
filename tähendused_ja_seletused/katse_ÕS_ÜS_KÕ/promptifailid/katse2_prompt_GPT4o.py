#EKKD-III1 tähenduste töörühma päringukood, et promptida OpenAI GPT-4o mudelit.
#Küsimus puudutab erinevat tüüpi definitsioonide vormistamist. 
#Autor: Eleri Aedmaa
import openai
import csv
import pandas as pd
import os

# OpenAI API võtme määramine
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY keskkonnamuutuja puudub. Palun määra see enne skripti käivitamist.")

# OpenAI kliendi initsialiseerimine
client = openai.OpenAI(api_key=api_key)

# Failist sisendi lugemine
def read_inputs_from_file(file_path):
    try:
        return pd.read_csv(file_path, delimiter='\t', usecols=['Katsesõna'])
    except Exception as e:
        raise ValueError(f"Sisendfaili lugemisel tekkis viga: {e}")

# OpenAI mudelilt vastuse pärimine
def get_response_for_input(word):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Sa oled eesti keele sõnaraamatu koostaja. Kirjuta sõnale '{word}' kolm definitsiooni, kus seletad, mida see sõna eesti keeles tähendab. "
                    "Esimene seletus kirjuta samas stiilis nagu definitsioon ÜS, teine seletus kirjuta samas stiilis nagu definitsioon ÕS "
                    "ja kolmas seletus kirjuta samas stiilis nagu definitsioon KÕ. "
                    "Vastus peab olema järgmisel kujul:\n"
                    "definitsioon ÜS - [ÜS-stiilis seletus]\n"
                    "definitsioon ÕS - [ÕS-stiilis seletus]\n"
                    "definitsioon KÕ - [KÕ-stiilis seletus]\n"
                    "Näited:\n"
                    "huumor\n"
                    "definitsioon ÜS - heatahtlik nali, koomiliste elunähtuste, sündmuste või inimeste puuduste ja nõrkuste heatahtlik naeruvääristamine\n"
                    "definitsioon ÕS - heatahtlik nali\n"
                    "definitsioon KÕ - see, kui miski on naljakas\n"
                    "kaste\n"
                    "definitsioon ÜS - vedel või poolvedel lisand, mis muudab toidu mahlasemaks ja maitsvamaks\n"
                    "definitsioon ÕS - soust\n"
                    "definitsioon KÕ - paks vedelik, mida süüakse koos muu toiduga\n"
                    "erinev\n"
                    "definitsioon ÜS - mingil viisil teistest, muust eristuv\n"
                    "definitsioon ÕS - muust eristuv\n"
                    "definitsioon KÕ - (kellegagi, millegagi või omavahel võrreldes) teistsugune\n"
                    "eeldama\n"
                    "definitsioon ÜS - eeltingimusena mingit omadust, oskust vm asjaolu vajama või ootama\n"
                    "definitsioon ÕS - eeldusena nõudma\n"
                    "definitsioon KÕ - tingimusena millegi olemasolu vajama, eeldusena nõudma\n"
                    "juures\n"
                    "definitsioon ÜS - kellegi või millegi vahetus läheduses, hästi lähedal\n"
                    "definitsioon ÕS - vahetus läheduses\n"
                    "definitsioon KÕ - (millegi) lähedal"
                )
            },
            {
                "role": "user",
                "content": f"Sõna: {word}"
            }
        ],
        temperature=0.2,
        max_tokens=512,
        top_p=1
    )
    return response.choices[0].message.content

# Parandatud funktsioon vastuste töötlemiseks
def process_response(word):
    try:
        response = get_response_for_input(word)
        lines = response.split('\n')

        # Default values
        us_def = ""
        os_def = ""
        ko_def = ""

        # Parse the structured response
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

# Peafunktsioon
def main():
    # Sisendfaili tee
    input_file_path = 'katse2_sisend.csv'  # Fail peab sisaldama "Katsesõna" veergu
    output_file_path = 'katse2_väljund_gpt4o.csv'

    try:
        user_inputs = read_inputs_from_file(input_file_path)

        # Kontrolli, et vajalik veerg on olemas
        if 'Katsesõna' not in user_inputs.columns:
            raise ValueError("Sisendfail peab sisaldama veergu 'Katsesõna'.")

        user_inputs[['definitsioon ÜS', 'definitsioon ÕS', 'definitsioon KÕ']] = user_inputs.apply(
            lambda row: process_response(row['Katsesõna']),
            axis=1
        )

        # Salvesta tulemused faili
        user_inputs.to_csv(output_file_path, index=False, quoting=csv.QUOTE_NONNUMERIC, sep='\t')
        print(f"Tulemused salvestatud faili: {output_file_path}")
    except Exception as e:
        print(f"Tekkis viga: {e}")

if __name__ == "__main__":
    main()

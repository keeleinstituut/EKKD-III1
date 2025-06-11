#EKKD-III1 tähenduste töörühma teise katse päringukood, et promptida OpenAI mudeleid.
#Mudelit juhendatakse genereerima 5 näite põhjal ÜSi stiilis definitsioon. 
#Autor: Eleri Aedmaa

import os
import csv
import time
import pandas as pd
import openai

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
        model="gpt-4.1-2025-04-14",
        messages=[
            {
                "role": "system",
                "content": (
                    "Sa oled eesti keele sõnaraamatu koostaja. Kirjuta sõnale ÜSi stiilis definitsioon, kus seletad, mida see sõna eesti keeles tähendab. Lähtu allolevate näidete stiilist.\n"
                    "**huumor**\n"
                    "ÜSi definitsioon - heatahtlik nali, koomiliste elunähtuste, sündmuste või inimeste puuduste ja nõrkuste heatahtlik naeruvääristamine\n"
                    "**kaste**\n"
                    "ÜSi definitsioon - vedel või poolvedel lisand, mis muudab toidu mahlasemaks ja maitsvamaks\n"
                    "**erinev**\n"
                    "ÜSi definitsioon - mingil viisil teistest, muust eristuv\n"
                    "**eeldama**\n"
                    "ÜSi definitsioon - eeltingimusena mingit omadust, oskust vm asjaolu vajama või ootama\n"
                    "**juures**\n"
                    "ÜSi definitsioon - kellegi või millegi vahetus läheduses, hästi lähedal\n"
                    "Vastus peab olema järgmisel kujul:\n"
                    f"{word} - [ÜSi definitsioon]"
                )
            },
            {
                "role": "user",
                "content": f"Sõna: {word}"
            }
        ],
        temperature=0.2,
        max_tokens=1000,
        top_p=1
    )
    return response.choices[0].message.content

# Vastuse töötlemine
def process_response(word):
    try:
        response = get_response_for_input(word)
        lines = response.strip().split('\n')

        for line in lines:
            if '-' in line:
                parts = line.split('-', 1)
                if len(parts) == 2:
                    return pd.Series([parts[1].strip()])

        return pd.Series(["Määramata definitsioon"])
    except Exception as e:
        return pd.Series([f"Töötlemise viga: {e}"])

# Peafunktsioon
def main():
    input_file_path = 'katse2_eelvalik_sisend.csv'
    output_file_path = 'katse2_eelvalik_valjund_gpt41.csv'

    try:
        user_inputs = read_inputs_from_file(input_file_path)

        if 'Katsesõna' not in user_inputs.columns:
            raise ValueError("Sisendfail peab sisaldama veergu 'Katsesõna'.")

        user_inputs[['ÜS_definitsioon']] = user_inputs.apply(
            lambda row: process_response(row['Katsesõna']),
            axis=1
        )

        user_inputs.to_csv(output_file_path, index=False, quoting=csv.QUOTE_NONNUMERIC, sep='\t')
        print(f"Tulemused salvestatud faili: {output_file_path}")
    except Exception as e:
        print(f"Tekkis viga: {e}")

if __name__ == "__main__":
    main()

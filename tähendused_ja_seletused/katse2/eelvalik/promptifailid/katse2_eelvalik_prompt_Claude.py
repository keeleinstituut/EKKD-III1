#EKKD-III1 tähenduste töörühma teise katse päringukood, et promptida Anthropicu mudeleid.
#Mudelit juhendatakse genereerima 5 näite põhjal ÜSi stiilis definitsioon. 
#Autor: Eleri Aedmaa

import anthropic
import csv
import pandas as pd

def read_inputs_from_file(file_path):
    try:
        return pd.read_csv(file_path, delimiter='\t', usecols=['Katsesõna'])
    except Exception as e:
        raise ValueError(f"Sisendfaili lugemisel tekkis viga: {e}")

def get_response_for_input(api_key, word):
    client = anthropic.Client(api_key=api_key)
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219", 
        system=f"Sa oled eesti keele sõnaraamatu koostaja. Kirjuta sõnale '{word}' ÜSi stiilis definitsioon, kus seletad, mida see sõna eesti keeles tähendab. Lähtu allolevate näidetest. "
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
               "sõna - [ÜSi definitsioon]",
        messages=[{
            "role": "user",
            "content": f"Sõna: {word}"
        }],
        max_tokens=32000,
        thinking: {
            type: enabled, 
            budget_tokens: 16000
        }
    )
    print(f"Claude vastus sõnale '{word}': {response.content}")
    return response.content

def process_response(api_key, word):
    try:
        response = get_response_for_input(api_key, word)
        text = response[0].text if isinstance(response, list) else response
        lines = text.split('\n')

        os_def = ""
        for line in lines:
            if '-' in line:
                parts = line.split('-', 1)
                if len(parts) == 2:
                    os_def = parts[1].strip()
                    break
                
        return pd.Series([os_def])
    except Exception as e:
        return pd.Series([f"Töötlemise viga: {e}"])

def main():
    api_key = ""
    input_file_path = 'katse2_eelvalik_sisend.csv'
    output_file_path = 'katse2_think_eelvalik_valjund_claude37.csv'

    try:
        user_inputs = read_inputs_from_file(input_file_path)

        if 'Katsesõna' not in user_inputs.columns:
            raise ValueError("Sisendfail peab sisaldama veergu 'Katsesõna'.")

        user_inputs[['ÜS_definitsioon']] = user_inputs.apply(
            lambda row: process_response(api_key, row['Katsesõna']),
            axis=1
        )

        user_inputs.to_csv(output_file_path, index=False, quoting=csv.QUOTE_MINIMAL, sep='\t')
        print(f"Tulemused salvestatud faili: {output_file_path}")
    except Exception as e:
        print(f"Tekkis viga: {e}")

if __name__ == "__main__":
    main()

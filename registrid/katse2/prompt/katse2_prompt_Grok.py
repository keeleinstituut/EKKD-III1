#Kood EKKD-III1 registrite töörühma teise katse päringute tegemiseks Xi mudelilt Grok-3.
#Autor: Eleri Aedmaa

import os
import csv
import pandas as pd
from openai import OpenAI

def read_inputs_from_file(file_path):
    try:
        return pd.read_csv(file_path, delimiter='\t', usecols=['Katsesõna', 'Tähendus'], encoding='utf-8')
    except Exception as e:
        raise ValueError(f"Sisendfaili lugemisel tekkis viga: {e}")

def get_response_for_input(client, word, meaning):
    try:
        response = client.chat.completions.create(
            model="grok-3",
            messages=[
                {"role": "system", "content": (
                    "Roll: Oled eesti keele sõnaraamatu koostaja, kelle ülesandeks on määrata, "
                    "kas sõnale või väljendile tuleks lisada registrimärgend. "
                    f"Sõna '{word}', tähenduses '{meaning}', kas kasutatakse pigem "
                    "[informaalsetes / neutraalsetes / formaalsetes] tekstides? "
                    "Kui pole selge, vasta 'ei kohaldu'. Põhjenda valikut. "
                    "Vastus peab olema:\n"
                    "Kasutus: [informaalsetes / neutraalsetes / formaalsetes / ei kohaldu]\n"
                    "Põhjendus: [Selgitus]"
                )},
                {"role": "user", "content": f"Sõna: {word}\nTähendus: {meaning}"}
            ],
            temperature=0.2,
            max_tokens=1024
        )
        response_text = response.choices[0].message.content
        print(f"API vastus sõnale '{word}' (tähendus: {meaning}): {response_text}")
        return response_text
    except Exception as e:
        print(f"API viga sõna '{word}' töötlemisel: {e}")
        return ""

def process_response(client, word, meaning):
    print(f"Analüüsin sõna: '{word}' – tähendus: '{meaning}'")
    try:
        text = get_response_for_input(client, word, meaning)
        if not text:
            return pd.Series(["Viga", "API vastus puudub või ebaõnnestus"])
        category = explanation = ""
        for line in text.splitlines():
            if line.lower().startswith("kasutus:"):
                category = line.split(":", 1)[1].strip()
            elif line.lower().startswith("põhjendus:"):
                explanation = line.split(":", 1)[1].strip()
        if not category:
            category = "Määramata"
        if not explanation:
            explanation = "Pole infot"
        return pd.Series([category, explanation])
    except Exception as e:
        print(f"Viga sõna '{word}' töötlemisel: {e}")
        return pd.Series(["Viga", f"Töötlemise viga: {str(e)}"])

def main():
    key = os.getenv("XAI_API_KEY")
    if not key:
        raise ValueError("Määra XAI_API_KEY keskkonnamuutujas!")

    client = OpenAI(
        api_key=key,
        base_url="https://api.x.ai/v1"
    )

    input_path = 'ekkd_i_k6nek_6s.csv'
    output_path = 'ekkd_i_k6nek_6s_väljund_grok3.csv'

    try:
        df = read_inputs_from_file(input_path)
        if not {'Katsesõna', 'Tähendus'}.issubset(df.columns):
            raise ValueError("Failis peab olema veerud 'Katsesõna' ja 'Tähendus'.")

        # Tagame, et kaust eksisteerib
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Algne DataFrame väljundi jaoks
        result_df = pd.DataFrame(columns=['Katsesõna', 'Tähendus', 'vastus', 'põhjendus'])
        
        # Kirjutame päised esimesel korral
        result_df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC, sep=',', encoding='utf-8-sig')

        # Töötleme read ükshaaval ja kirjutame jooksvalt faili
        for index, row in df.iterrows():
            word = row['Katsesõna']
            meaning = row['Tähendus']
            category, explanation = process_response(client, word, meaning)
            # Loome ühe rea DataFrame'i
            new_row = pd.DataFrame({
                'Katsesõna': [word],
                'Tähendus': [meaning],
                'vastus': [category],
                'põhjendus': [explanation]
            })
            # Lisame rea result_df-i ja kirjutame faili
            result_df = pd.concat([result_df, new_row], ignore_index=True)
            new_row.to_csv(output_path, mode='a', header=False, index=False, quoting=csv.QUOTE_NONNUMERIC, sep=',', encoding='utf-8-sig')
            print(f"Rida salvestatud sõnale '{word}'")

        print("\nEsimesed read väljundist:")
        print(result_df.head())
        print(f"Salvestatud: {output_path}")

    except Exception as e:
        print(f"Viga programmi käivitamisel: {e}")
        raise

if __name__ == "__main__":
    main()
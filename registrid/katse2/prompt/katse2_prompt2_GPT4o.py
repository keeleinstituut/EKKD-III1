#Kood EKKD-III1 registrite töörühma teise katse päringute tegemiseks OpenAI mudelilt GPT-4o.
#Autor: Eleri Aedmaa

import openai
import csv
import pandas as pd

# Failist sisendi lugemine
def read_inputs_from_file(file_path):
    try:
        return pd.read_csv(file_path, delimiter='\t', usecols=['Katsesõna', 'Tähendus'])
    except Exception as e:
        raise ValueError(f"Sisendfaili lugemisel tekkis viga: {e}")

# OpenAI mudelilt vastuse pärimine
def get_response_for_input(api_key, word, meaning):
    client = openai.OpenAI(api_key=api_key)  # OpenAI API kliendi loomine
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",  # Kasuta õiget mudeli nime
        messages=[
            {
                "role": "system",
                "content": (
                    "Oled eesti keele sõnaraamatu koostaja, kelle ülesandeks on määrata, kas sõnale või väljendile tuleks lisada registrimärgend."
                    "Kas eesti(keelset) sõna '{word}' tähenduses '{meaning}' kasutatakse pigem [informaalsetes, neutraalsetes/formaalsetes] tekstides? "
                    "Kui sa ei oska eristust teha või see ei tule selgelt esile, siis ütle, et 'ei kohaldu'. "
                    "Informaalsed tekstid on teiste seas näiteks blogid, foorumid, kommentaariumid, chativestlused, sotsiaalmeedia tekstid, trükivigasid täis tekstid, vahel ka raamatutegelaste otsekõne. "
                    "Palun põhjenda oma valikut. Lähtu vastates ainult oma treeningandmetest, mitte välisotsingutest ja andmebaasidest. "
                    "Kui sõna '{word}' kasutatakse pigem informaalsetes tekstides, siis mis on sõna '{word}' neutraalsed/formaalsed sünonüümid eesti keeles? "
                    "Kui sõna kasutatakse pigem neutraalsetes/formaalsetes tekstides, siis vasta 'ei kohaldu'. "
                    "Vastus peab olema järgmisel kujul:\n"
                    "Kasutus: [informaalsetes / neutraalsetes/formaalsetes / ei kohaldu]\n"
                    "Põhjendus: [Selgitus kasutuse kohta]\n"
                    "Sünonüümid: [Sünonüümid või 'ei kohaldu']"
                )
            },
            {
                "role": "user",
                "content": f"Sõna: {word}\nTähendus: {meaning}"
            }
        ],
        temperature=0.2,
        max_tokens=4096,
        top_p=1
    )
    print(f"GPT vastus sõnale '{word}': {response.choices[0].message.content}")  # Print vastus
    return response.choices[0].message.content

# Parandatud funktsioon vastuste töötlemiseks
def process_response(api_key, word, meaning):
    try:
        response = get_response_for_input(api_key, word, meaning)
        lines = response.split('\n')

        # Default values
        category = ""
        explanation = ""
        synonyms = ""

        # Parse the structured response
        for line in lines:
            if line.startswith("Kasutus:"):
                category = line.replace("Kasutus:", "").strip()
            elif line.startswith("Põhjendus:"):
                explanation = line.replace("Põhjendus:", "").strip()
            elif line.startswith("Sünonüümid:"):
                synonyms = line.replace("Sünonüümid:", "").strip()

        return pd.Series([category, explanation, synonyms])
    except Exception as e:
        return pd.Series(["Viga", f"Töötlemise viga: {e}", ""])

# Peafunktsioon
def main():
    # OpenAI API võtme sisestamine
    api_key = ""  # Asenda oma API võtmega

    # Sisendfaili tee
    input_file_path = 'katse2_sisend2.csv'  # Fail peab sisaldama "Katsesõna" ja "Tähendus" veerge
    output_file_path = 'katse2_prompt2_väljund_gpt4o.csv'

    try:
        user_inputs = read_inputs_from_file(input_file_path)

        # Kontrolli, et vajalikud veerud on olemas
        if 'Katsesõna' not in user_inputs.columns or 'Tähendus' not in user_inputs.columns:
            raise ValueError("Sisendfail peab sisaldama veerge 'Katsesõna' ja 'Tähendus'.")

        user_inputs[['vastus', 'põhjendus', 'sünonüümid']] = user_inputs.apply(
            lambda row: process_response(api_key, row['Katsesõna'], row['Tähendus']),
            axis=1
        )

        # Salvesta tulemused faili
        user_inputs.to_csv(output_file_path, index=False, quoting=csv.QUOTE_NONNUMERIC, sep='\t')
        print(f"Tulemused salvestatud faili: {output_file_path}")
    except Exception as e:
        print(f"Tekkis viga: {e}")

if __name__ == "__main__":
    main()

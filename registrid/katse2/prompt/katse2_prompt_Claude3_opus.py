#Kood EKKD-III1 registrite töörühma teise katse päringute tegemiseks Anthropicu mudelilt Claude 3 Opus.
#Autor: Eleri Aedmaa

import anthropic
import csv
import pandas as pd

def read_inputs_from_file(file_path):
    try:
        return pd.read_csv(file_path, delimiter='\t', usecols=['Katsesõna', 'Tähendus'])
    except Exception as e:
        raise ValueError(f"Sisendfaili lugemisel tekkis viga: {e}")

def get_response_for_input(api_key, word, meaning):
    client = anthropic.Client(api_key=api_key)
    response = client.messages.create(
        model="claude-3-opus-20240229",
        system=f"Oled eesti keele sõnaraamatu koostaja, kelle ülesandeks on määrata, kas sõnale või väljendile tuleks lisada registrimärgend. "
            f"Kas eesti(keelset) sõna '{word}' tähenduses '{meaning}' kasutatakse pigem [informaalsetes, neutraalsetes/formaalsetes] registrites? "
            "Kui sa ei oska eristust teha või see ei tule selgelt esile, siis ütle, et 'ei kohaldu'. "
            "Informaalsed registrid on teiste seas näiteks blogid, foorumid, kommentaariumid, chativestlused, sotsiaalmeedia tekstid, trükivigasid täis tekstid, vahel ka raamatutegelaste otsekõne. "
            "Palun põhjenda oma valikut. Lähtu vastates ainult oma treeningandmetest, mitte välisotsingutest ja andmebaasidest. "
            f"Kui sõna '{word}' kasutatakse pigem informaalsetes registrites, siis mis on sõna '{word}' neutraalsed/formaalsed sünonüümid eesti keeles? "
            "Kui sõna kasutatakse pigem neutraalsetes/formaalsetes registrites, siis vasta 'ei kohaldu'. "
            "Vastus peab olema järgmisel kujul:\n"
            "Kasutus: [informaalsetes / neutraalsetes/formaalsetes / ei kohaldu]\n"
            "Põhjendus: [Selgitus kasutuse kohta]\n"
            "Sünonüümid: [Sünonüümid või 'ei kohaldu']",
        messages=[{
            "role": "user",
            "content": f"Sõna: {word}\nTähendus: {meaning}"
        }],
        temperature=0.2,
        max_tokens=4096
    )
    print(f"Claude vastus sõnale '{word}': {response.content}")
    return response.content

def process_response(api_key, word, meaning):
    try:
        response = get_response_for_input(api_key, word, meaning)
        text = response[0].text if isinstance(response, list) else response
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
    except Exception as e:
        return pd.Series(["Viga", f"Töötlemise viga: {e}", ""])

def main():
    api_key = ""
    input_file_path = 'katse2_sisend2.csv'
    output_file_path = 'katse2_prompt3_väljund_claude3opus.csv'

    try:
        user_inputs = read_inputs_from_file(input_file_path)
        
        if 'Katsesõna' not in user_inputs.columns or 'Tähendus' not in user_inputs.columns:
            raise ValueError("Sisendfail peab sisaldama veerge 'Katsesõna' ja 'Tähendus'.")
        user_inputs[['vastus', 'põhjendus', 'sünonüümid']] = user_inputs.apply(
            lambda row: process_response(api_key, row['Katsesõna'], row['Tähendus']),
            axis=1
        )

        user_inputs.to_csv(output_file_path, index=False, quoting=csv.QUOTE_MINIMAL, sep='\t')
        print(f"Tulemused salvestatud faili: {output_file_path}")
    except Exception as e:
        print(f"Tekkis viga: {e}")
        
if __name__ == "__main__":
    main()
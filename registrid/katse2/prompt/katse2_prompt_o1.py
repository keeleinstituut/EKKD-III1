#Kood EKKD-III1 registrite töörühma teise katse päringute tegemiseks OpenAI mudelilt o1.
#Autor: Eleri Aedmaa

from openai import OpenAI
import csv
import pandas as pd

# Function to read input from a file
def read_inputs_from_file(file_path):
    try:
        return pd.read_csv(file_path, delimiter='\t', usecols=['Katsesõna', 'Tähendus'])
    except Exception as e:
        raise ValueError(f"Sisendfaili lugemisel tekkis viga: {e}")

# Function to get a response from the OpenAI model
def get_response_for_input(client, word, meaning):
    try:
        user_prompt = (
            f"Oled eesti keele sõnaraamatu koostaja, kelle ülesandeks on määrata, kas sõnale või väljendile tuleks lisada registrimärgend. "
            f"Kas eesti(keelset) sõna '{word}' tähenduses '{meaning}' kasutatakse pigem [informaalsetes, neutraalsetes/formaalsetes] registrites? "
            "Kui sa ei oska eristust teha või see ei tule selgelt esile, siis ütle, et 'ei kohaldu'. "
            "Informaalsed registrid on teiste seas näiteks blogid, foorumid, kommentaariumid, chativestlused, sotsiaalmeedia tekstid, trükivigasid täis tekstid, vahel ka raamatutegelaste otsekõne. "
            "Palun põhjenda oma valikut. Lähtu vastates ainult oma treeningandmetest, mitte välisotsingutest ja andmebaasidest. "
            f"Kui sõna '{word}' kasutatakse pigem informaalsetes registrites, siis mis on sõna '{word}' neutraalsed/formaalsed sünonüümid eesti keeles? "
            "Kui sõna kasutatakse pigem neutraalsetes/formaalsetes registrites, siis vasta 'ei kohaldu'. "
            "Vastus peab olema järgmisel kujul:\n"
            "Kasutus: [informaalsetes / neutraalsetes/formaalsetes / ei kohaldu]\n"
            "Põhjendus: [Selgitus kasutuse kohta]\n"
            "Sünonüümid: [Sünonüümid või 'ei kohaldu']"
        )

        response = client.chat.completions.create(
            model="o1-mini-2024-09-12",  # kohandage vastavalt vajadusele
            # Võite lisada ka system-role, kui tahate:
            # messages=[
            #     {"role": "system", "content": "System-level juhised..."},
            #     {"role": "user", "content": user_prompt}
            # ],
            messages=[{"role": "user", "content": user_prompt}],
            temperature=1,
            max_completion_tokens=4096,
            top_p=1
        )

        content = response.choices[0].message.content.strip()
        print(f"Full GPT Response for '{word}': {content}")
        return content
    except Exception as e:
        return f"Viga: {e}"

# Function to process the response
def process_response(client, word, meaning):
    try:
        response = get_response_for_input(client, word, meaning)
        lines = response.split('\n')

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

        if not category and not explanation and not synonyms:
            print(f"Unexpected response format for '{word}': {response}")

        return pd.Series([category, explanation, synonyms])
    except Exception as e:
        return pd.Series(["Viga", f"Töötlemise viga: {e}", ""])

def main():
    api_key = ""  # Replace with your actual API key
    client = OpenAI(api_key=api_key)  # Initialize client here

    input_file_path = 'katse2_sisend2.csv'
    output_file_path = 'katse2_prompt3_väljund_o1mini.csv'

    try:
        user_inputs = read_inputs_from_file(input_file_path)

        if 'Katsesõna' not in user_inputs.columns or 'Tähendus' not in user_inputs.columns:
            raise ValueError("Sisendfail peab sisaldama veerge 'Katsesõna' ja 'Tähendus'.")

        user_inputs[['vastus', 'põhjendus', 'sünonüümid']] = user_inputs.apply(
            lambda row: process_response(client, row['Katsesõna'], row['Tähendus']),
            axis=1
        )

        user_inputs.to_csv(output_file_path, index=False, quoting=csv.QUOTE_NONNUMERIC, sep='\t')
        print(f"Tulemused salvestatud faili: {output_file_path}")
    except Exception as e:
        print(f"Tekkis viga: {e}")

if __name__ == "__main__":
    main()

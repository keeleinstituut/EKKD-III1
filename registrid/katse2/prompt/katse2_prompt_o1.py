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
            f"Sa oled eesti keele sõnaraamatu koostaja. Eesti(keelset) sõna '{word}' tähenduses '{meaning}' kasutatakse pigem [informaalsetes, neutraalsetes/formaalsetes, võrdselt] tekstides. "
            "Informaalsed tekstid on näiteks blogid, foorumid, kommentaariumid, chativestlused, sotsiaalmeedia tekstid, trükivigasid täis tekstid, otsekõnes. "
            "Kui sa ei tea, siis ütle, et sa ei oska öelda. Palun põhjenda oma valikut selgelt ning esita põhjenduse järel sõna võimalikud neutraalsed sünonüümid, "
            "kui sõna kasutatakse pigem informaalsetes tekstides. Kui sõna kasutatakse pigem neutraalsetes/formaalsetes tekstides, siis vasta 'ei kohaldu'. "
            "Vastus peab olema järgmisel kujul:\n"
            "Kasutus: [informaalsetes / neutraalsetes/formaalsetes / võrdselt]\n"
            "Põhjendus: [Selgitus kasutuse kohta]\n"
            "Sünonüümid: [Sünonüümid või 'ei kohaldu']"
        )

        response = client.chat.completions.create(
            model="o1",
            messages=[
                {"role": "user", "content": user_prompt}
            ],
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
    api_key = ""  # Replace with your API key
    client = OpenAI(api_key=api_key)  # Initialize client here

    input_file_path = 'katse2_sisend.csv'
    output_file_path = 'katse2_väljund_o1mini.csv'

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

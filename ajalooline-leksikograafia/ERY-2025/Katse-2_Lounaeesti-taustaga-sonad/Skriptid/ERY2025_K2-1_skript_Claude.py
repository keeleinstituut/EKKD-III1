#Skript, mis otsib Claude'i abil sisendfailis olevatele sõnadele vasteid Gutslaffi sõnastikust, kasutades ka DWDSi abi. 
#Autor: Eleri Aedmaa

import os
import pandas as pd
import requests
import time
import logging
from anthropic import Anthropic

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fetch your Claude API key from environment variables for security
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
if not ANTHROPIC_API_KEY:
    logging.error("Claude API key not found. Please set the 'ANTHROPIC_API_KEY' environment variable.")
    exit(1)

# Initialize Anthropic client
client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Function to fetch data from DWDS using the German equivalent
def fetch_from_dwds(word):
    url = f"https://www.dwds.de/wb/{word}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            logging.warning(f"Failed to fetch data for {word}. HTTP status code: {response.status_code}")
            return f"Failed to fetch data for {word}. HTTP status code: {response.status_code}"
    except Exception as e:
        logging.error(f"An error occurred while fetching data from DWDS: {e}")
        return f"An error occurred while fetching data from DWDS: {e}"

# Function to generate content using Claude API with retry logic
def generate_content_with_claude(prompt, max_retries=5, backoff_factor=2):
    for attempt in range(1, max_retries + 1):
        try:
            message = client.messages.create(
                model="claude-3-opus-20240229",  # This model name may need to be updated
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception as e:
            wait_time = backoff_factor ** attempt
            logging.warning(f"Error with Claude API: {e}. Attempt {attempt}/{max_retries}. Retrying in {wait_time} seconds.")
            time.sleep(wait_time)
    return "Failed to generate content with Claude after multiple retries."

# Main processing logic
try:
    # Read the Excel file
    excel_file = 'Fail_murre_1.xlsx'
    df = pd.read_excel(excel_file)
    file_content = df.to_string()
    logging.info(f"Successfully read the Excel file '{excel_file}'.")

    # Read input words
    input_file = 'sisend_murdes6nad.txt'
    with open(input_file, 'r', encoding='utf-8') as file:
        input_words = [line.strip() for line in file if line.strip()]
    logging.info(f"Successfully read {len(input_words)} words from '{input_file}'.")

    # Process each word
    for idx, input_word in enumerate(input_words, start=1):
        logging.info(f"\nProcessing word {idx}/{len(input_words)}: {input_word}")

        matched_rows = df[df['Gutslaff'] == input_word]
        if not matched_rows.empty:
            gutslaff_version = matched_rows['Gutslaff'].values[0]
            german_equivalent = matched_rows['sks'].values[0]
            logging.info(f"Found match: Gutslaff='{gutslaff_version}', German Equivalent='{german_equivalent}'")
        else:
            logging.warning(f"No match found for the modern word '{input_word}' in the Excel file.")
            gutslaff_version = "Ei leitud"
            german_equivalent = input_word

        dwds_data = fetch_from_dwds(german_equivalent)
        logging.info(f"DWDS data for '{german_equivalent}': {dwds_data[:100]}...")

        prompt = (
            f"Sinu ülesanne on leida sisestatud sõna '{input_word}' esinemiskujud Gutslaffi sõnastikufailist. "
            f"Gutslaffi sõnastikus on selle sõna vaste '{gutslaff_version}'. "
            f"Allpool on sõnastikufaili sisu:\n\n{file_content}\n\n"
            f"Kui sa failist õiget sõnakuju ei leia, siis võid otsida sõna tähendust DWDSi kaudu. "
            f"{dwds_data}. "
            f"Vastuses esita sõnavorm täpselt sel kujul, nagu ta sõnastiku failis esineb. "
            f"Tulemus väljasta kujul *Sõna {input_word} esineb Gutslaffi sõnastikus kujul '{gutslaff_version}'."
        )

        generated_text = generate_content_with_claude(prompt)
        print(f"Result for '{input_word}':\n{generated_text}\n")

except FileNotFoundError as e:
    logging.error(f"File not found: {e}")
except Exception as e:
    logging.error(f"An error occurred: {e}")
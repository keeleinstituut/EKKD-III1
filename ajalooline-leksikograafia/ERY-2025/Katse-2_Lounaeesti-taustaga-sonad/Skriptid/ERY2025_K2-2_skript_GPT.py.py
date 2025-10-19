#Skript, mis otsib GPT abil sisendfailis olevatele Gutslaffi sõnadele tänapäevased vasted, kasutades ka VMSi abi. 
#Autor: Eleri Aedmaa

import os
import pandas as pd
import requests
import openai
import time
import logging
from openai.error import RateLimitError, OpenAIError
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up OpenAI API key (Replace with your actual OpenAI API key)
openai.api_key = ''

# Function to fetch and parse data from VMS using the given word and potential variants
def fetch_vms_data(word):
    """Fetch data from VMS for different variations of the word and return unstructured text."""
    base_url = "https://arhiiv.eki.ee/cgi-bin/vms.cgi?mrks="
    variants = [word, word.capitalize(), word.lower()]  # Different possible variations of the word

    vms_text_results = {}  # Store plain text results for each variant
    for variant in variants:
        url = f"{base_url}{variant}"
        logging.info(f"Fetching data for variant: {variant} from URL: {url}")
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Parse VMS data using BeautifulSoup and extract all text
                soup = BeautifulSoup(response.text, 'html.parser')
                extracted_text = soup.get_text(separator=' ', strip=True)  # Extract full text
                if extracted_text:
                    vms_text_results[variant] = extracted_text
                    print(f"Fetched text for '{variant}':\n{extracted_text[:500]}...\n")  # Print the first 500 characters
                else:
                    print(f"No readable text extracted for variant '{variant}'.")
            else:
                logging.warning(f"Failed to fetch data for '{variant}'. HTTP status code: {response.status_code}")
        except Exception as e:
            logging.error(f"An error occurred while fetching data from VMS for '{variant}': {e}")

    return vms_text_results

# Function to generate content using OpenAI's GPT API
def generate_content_with_gpt(prompt, max_retries=5, backoff_factor=2):
    for attempt in range(1, max_retries + 1):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in analyzing old Estonian vocabulary."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7,
                n=1,
                stop=None
            )
            return response['choices'][0]['message']['content'].strip()
        
        except RateLimitError:
            wait_time = backoff_factor ** attempt
            logging.warning(f"Rate limit reached. Attempt {attempt}/{max_retries}. Retrying in {wait_time} seconds.")
            time.sleep(wait_time)
        except OpenAIError as e:
            logging.error(f"An OpenAI error occurred: {e}")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            break
    return "Failed to generate content after multiple retries."

# Load and process the Excel file
try:
    excel_file = 'Fail_murre_1.xlsx'
    df = pd.read_excel(excel_file)
    words_column = df.iloc[:, 0]  # Assuming Gutslaff words are in the first column
    german_latin_column = df.iloc[:, 1]  # Assuming German/Latin equivalents are in the second column

except FileNotFoundError:
    logging.error(f"Error: The file '{excel_file}' was not found.")
    exit(1)
except Exception as e:
    logging.error(f"An error occurred while reading the file: {e}")
    exit(1)

# Process each word from the Excel file
for idx, input_word in enumerate(words_column, start=1):
    input_word = str(input_word).strip()  # Clean the word
    if not input_word:
        continue

    # Check if German/Latin equivalent exists
    german_latin_equivalent = str(german_latin_column[idx-1]).strip() if not pd.isna(german_latin_column[idx-1]) else None

    # Fetch and analyze VMS data
    vms_text_data = fetch_vms_data(input_word)

    # Construct the prompt based on available data
    if vms_text_data:
        # Include raw text from VMS in the prompt
        parsed_data = "\n".join([f"Variant '{var}': {text[:500]}..." for var, text in vms_text_data.items()])  # Limit to 500 chars
    else:
        parsed_data = "No VMS data found."

    prompt = (
        f"Sinu ülesanne on leida sisestatud Gutslaffi sõnale '{input_word}' tänapäevased vasted. "
        f"Abivahendina võid sõnastikufailist leida Gutslaffi sõnale ka saksa ja/või ladina vaste. "
        f"{'Saksa või ladina vaste on ' + german_latin_equivalent if german_latin_equivalent else 'Vaste puudub'}. "
        f"Kui sa failist õiget sõnakuju ei leia, siis vaata VMSi tulemusi, need ei pruugi vastust anda aga võivad aidata. "
        f"Tulemused VMSist on:\n{parsed_data}. "
        f"Pea meeles, et sõnade kirjakuju võib pisut erineda ja seega tuleb tähendust tõlgendada. "
        f"Vasteid võib olla mitu, samuti lisa sünonüümid kui need on olemas. Tulemus väljasta kujul: "
        f"Gutslaffi sõna '{input_word}' vaste(d) tänapäeva kirjakeeles on '{{nowadays_word}}'."
    )

    # Generate content with OpenAI's GPT-4
    generated_text = generate_content_with_gpt(prompt)
    
    # Print the result for the current word
    print(f"Result for '{input_word}':\n{generated_text}\n")

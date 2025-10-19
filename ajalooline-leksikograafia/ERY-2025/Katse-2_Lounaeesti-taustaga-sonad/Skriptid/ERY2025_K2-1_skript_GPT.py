#Skript, mis otsib GPT-4o abil sisendfailis olevatele sõnadele vasteid Gutslaffi sõnastikust, kasutades ka DWDSi abi. 


import os
import pandas as pd
import requests  # For handling DWDS fetching
import openai  # For using OpenAI's GPT API
import time
import logging
from openai.error import RateLimitError, OpenAIError

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up OpenAI API key
openai.api_key = 'your-key'  # Replace with your actual OpenAI API key


if not openai.api_key:
    logging.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    exit(1)

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

# Function to generate content using OpenAI's (GPT-4o) API with retry logic
def generate_content_with_gpt(prompt, max_retries=5, backoff_factor=2):
    for attempt in range(1, max_retries + 1):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-2024-08-06",  #
                messages=[
                    {"role": "system", "content": "You are an expert in analyzing old Estonian vocabulary."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,  # Adjust the number of tokens according to your needs
                temperature=0.7,  # Adjust temperature for creativity
                n=1,  # Number of responses to generate
                stop=None  # Define stop sequences if needed
            )
            return response['choices'][0]['message']['content'].strip()
        
        except RateLimitError as e:
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

# File handling part with error catching
try:
    # Read the Excel file for Gutslaff version and German equivalents
    excel_file = 'Fail_murre_1.xlsx'
    df = pd.read_excel(excel_file)

    # Convert the DataFrame to a string for inclusion in the prompt
    file_content = df.to_string()

    logging.info(f"Successfully read the Excel file '{excel_file}'.")

except FileNotFoundError:
    logging.error(f"Error: The file '{excel_file}' was not found.")
    file_content = "Faili ei leitud."
except Exception as e:
    logging.error(f"An error occurred while reading the file: {e}")
    file_content = "Faili lugemise viga."

# File handling for the input words from 'sisend_murdes6nad.txt'
try:
    # Open the input file and read all words
    input_file = 'sisend_murdes6nad.txt'
    with open(input_file, 'r', encoding='utf-8') as file:
        input_words = [line.strip() for line in file if line.strip()]

    logging.info(f"Successfully read {len(input_words)} words from '{input_file}'.")

    # Loop through each word in the input file
    for idx, input_word in enumerate(input_words, start=1):
        logging.info(f"\nProcessing word {idx}/{len(input_words)}: {input_word}")

        # Check if the modern word exists in the 'Gutslaff' column
        matched_rows = df[df['Gutslaff'] == input_word]

        if not matched_rows.empty:
            # If a match is found, extract the Gutslaff version and the German equivalent
            gutslaff_version = matched_rows['Gutslaff'].values[0]
            german_equivalent = matched_rows['sks'].values[0]
            logging.info(f"Found match: Gutslaff='{gutslaff_version}', German Equivalent='{german_equivalent}'")
        else:
            # If no match is found, we'll try to fetch from DWDS using the German translation
            logging.warning(f"No match found for the modern word '{input_word}' in the Excel file.")
            gutslaff_version = "Ei leitud"
            german_equivalent = None  # We will try to find this from DWDS

        # If the German equivalent was not found in the Excel file, use DWDS to search for it
        if german_equivalent is None:
            german_equivalent = input_word  # Assuming input word may help in finding the translation in DWDS
            dwds_data = fetch_from_dwds(german_equivalent)
            logging.info(f"DWDS data for '{german_equivalent}': {dwds_data[:100]}...")  # Log first 100 chars
        else:
            dwds_data = f"DWDS URL on https://www.dwds.de/wb/{german_equivalent}"
            logging.info(f"Using DWDS URL: {dwds_data}")

        # Create the prompt using the Gutslaff version (if found) and its German equivalent
        prompt = (
            f"Sinu ülesanne on leida sisestatud sõna '{input_word}' esinemiskujud Gutslaffi sõnastikufailist. "
            f"Gutslaffi sõnastikus on selle sõna vaste '{gutslaff_version}'. "
            f"Allpool on sõnastikufaili sisu:\n\n{file_content}\n\n"
            f"Kui sa failist õiget sõnakuju ei leia, siis võid otsida sõna tähendust DWDSi kaudu. "
            f"{dwds_data}. Vastuses esita sõnavorm täpselt sel kujul, "
            f"nagu ta sõnastiku failis esineb. Tulemus väljasta kujul *Sõna "
            f"{input_word} esineb Gutslaffi sõnastikus kujul '{gutslaff_version}'. "
            f"Lisa vastuse seletusse ka info, mis sa VMSist said ja mis aitas sul vastust kokku panna."
        )

        # Generate content with OpenAI's GPT-4-turbo (GPT-4o)
        generated_text = generate_content_with_gpt(prompt)

        # Print the result for the current word
        print(f"Result for '{input_word}':\n{generated_text}\n")

except FileNotFoundError:
    logging.error(f"Error: The file '{input_file}' was not found.")
except Exception as e:
    logging.error(f"An error occurred while reading the input file: {e}")


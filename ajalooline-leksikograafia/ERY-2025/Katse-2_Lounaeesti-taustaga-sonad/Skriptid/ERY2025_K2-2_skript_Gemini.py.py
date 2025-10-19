#Skript, mis otsib Gemini abil sisendfailis olevatele Gutslaffi sõnadele tänapäevased vasted, kasutades ka VMSi abi. 
#Autor: Eleri Aedmaa

import os
import pandas as pd
import requests  # For handling VMS fetching
import google.generativeai as genai
import time
import logging
from bs4 import BeautifulSoup  # For parsing HTML
from google.api_core.exceptions import ResourceExhausted  # Import added

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up Google AI API key (replace with your actual key)
os.environ["GOOGLE_API_KEY"] = "your_key"  

# Configure the Gemini API
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize the Gemini 1.5 Pro model
model = genai.GenerativeModel("gemini-1.5-pro")

# Function to fetch and parse data from VMS
def fetch_vms_data(word):
    """Fetch data from VMS and handle word variations."""
    base_url = "https://arhiiv.eki.ee/cgi-bin/vms.cgi?mrks="
    variants = [word, word.capitalize(), word.lower()] 

    vms_results = {} 
    for variant in variants:
        url = f"{base_url}{variant}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                extracted_data, synonyms = extract_vms_content(soup)
                if extracted_data:
                    vms_results[variant] = {'data': extracted_data, 'synonyms': synonyms}
            else:
                logging.warning(f"Failed to fetch data for '{variant}'. HTTP status code: {response.status_code}")
        except Exception as e:
            logging.error(f"An error occurred while fetching data from VMS: {e}")

    return vms_results

# Function to extract content and synonyms from VMS HTML (ADJUST THIS)
def extract_vms_content(soup):
    """Extract relevant content and synonyms from the VMS HTML response."""

    # 1. Try to find a dedicated definition section
    definition_section = soup.find("div", {"class": "definition"}) # Replace with actual class/id if found

    if definition_section:
        extracted_text = definition_section.get_text()
    else:
        # 2. If no dedicated definition, extract all text content from the main entry 
        entry_content = soup.find("div", {"class": "entry"}) # Replace with actual class/id if needed
        if entry_content:
            extracted_text = entry_content.get_text()
        else:
            extracted_text = "No specific definition or usage information found."

    # 3. Extract synonyms (if available)
    synonyms_section = soup.find("ul", {"class": "synonym-list"})  # Replace with actual class/id if needed
    synonyms = [syn.get_text() for syn in synonyms_section.find_all("li")] if synonyms_section else []

    return extracted_text, synonyms

# Function to generate content with Gemini
def generate_content_with_gemini(prompt, max_retries=5, backoff_factor=2):
    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()

        except ResourceExhausted:
            wait_time = backoff_factor ** attempt
            logging.warning(f"Rate limit reached. Attempt {attempt}/{max_retries}. Retrying in {wait_time} seconds.")
            time.sleep(wait_time)
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            break
    return "Failed to generate content after multiple retries."


# Load and process the Excel file
try:
    excel_file = 'Fail_murre_1.xlsx'  # Replace with your Excel file name
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
    german_latin_equivalent = str(german_latin_column[idx - 1]).strip() if not pd.isna(
        german_latin_column[idx - 1]) else None

    # Fetch and analyze VMS data
    vms_data = fetch_vms_data(input_word)

    # Construct the prompt with VMS data AND Latin/German equivalent
    if vms_data:
        parsed_data = "\n".join([f"Variant '{var}': {info['data']}" for var, info in vms_data.items()])
        synonyms = "\n".join(
            [f"Variant '{var}': {', '.join(info['synonyms'])}" for var, info in vms_data.items() if info['synonyms']])
    else:
        parsed_data = "No VMS data found."
        synonyms = "No synonyms found."

    prompt = (
        f"You are an expert in analyzing old Estonian vocabulary. "
        f"Sinu ülesanne on leida sisestatud Gutslaffi sõnale '{input_word}' tänapäevased vasted. "
        f"Kasuta selleks kõiki sulle kättesaadavaid vahendeid, " 
        f"sh sinu enda teadmisi eesti keelest ja ajaloost, VMSi andmeid ja ladina/saksa vasteid. "
        f"{'Saksa või ladina vaste on ' + german_latin_equivalent if german_latin_equivalent else 'Vaste puudub'}. "
        f"Tulemused VMSist on:\n{parsed_data}\n"
        f"Leitud sünonüümid: {synonyms}\n"
        f"Pea meeles, et sõnade kirjakuju võib pisut erineda ja seega tuleb tähendust tõlgendada. "
        f"Vasteid võib olla mitu, samuti lisa sünonüümid kui need on olemas. "
        f"Tulemus väljasta ühel real kujul: "
        f"Gutslaffi sõna '{input_word}' vaste(d) tänapäeva kirjakeeles on: "
    )

    # Generate content with Gemini
    generated_text = generate_content_with_gemini(prompt)

    # Print the result for the current word
    print(f"Result for '{input_word}':\n{generated_text}\n")
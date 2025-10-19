#Skript, mis otsib Claude'i abil sisendfailis olevatele Gutslaffi sõnadele tänapäevased vasted, kasutades ka VMSi abi. 
#Autor: Eleri Aedmaa
import os
import pandas as pd
import requests
import time
import logging
from bs4 import BeautifulSoup
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

# Print Anthropic library version and available methods
import anthropic
print(f"Anthropic library version: {anthropic.__version__}")
try:
    print(f"Available methods in Anthropic client: {dir(client)}")
except Exception as e:
    print(f"Error when inspecting Anthropic client: {e}")

# Function to fetch and parse data from VMS using the given word and potential variants
def fetch_vms_data(word):
    """Fetch data from VMS and handle word variations."""
    base_url = "https://arhiiv.eki.ee/cgi-bin/vms.cgi?mrks="
    variants = [word, word.capitalize(), word.lower()]  # Different possible variations of the word

    vms_results = {}  # Store results for each variation
    for variant in variants:
        url = f"{base_url}{variant}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Parse VMS data using BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                extracted_data, synonyms = extract_vms_content(soup)
                if extracted_data:
                    vms_results[variant] = {'data': extracted_data, 'synonyms': synonyms}
            else:
                logging.warning(f"Failed to fetch data for '{variant}'. HTTP status code: {response.status_code}")
        except Exception as e:
            logging.error(f"An error occurred while fetching data from VMS: {e}")

    return vms_results

def extract_vms_content(soup):
    """Extract relevant content and synonyms from the VMS HTML response."""
    content = soup.find_all("div", {"class": "content"})
    synonyms_section = soup.find_all("div", {"class": "synonyms"})  # Hypothetical class name for synonyms

    if not content:
        return None, None

    extracted_text = "\n".join([entry.get_text() for entry in content])

    # Extract synonyms if available
    synonyms = [syn.get_text() for syn in synonyms_section] if synonyms_section else []

    return extracted_text, synonyms

# Function to generate content using Claude's API
def generate_content_with_claude(prompt, max_retries=5, backoff_factor=2):
    for attempt in range(1, max_retries + 1):
        try:
            message = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=500,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception as e:
            wait_time = backoff_factor ** attempt
            logging.warning(f"Error with Claude API: {e}. Attempt {attempt}/{max_retries}. Retrying in {wait_time} seconds.")
            time.sleep(wait_time)
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
    vms_data = fetch_vms_data(input_word)

    # Construct the prompt based on available data
    if vms_data:
        # Include extracted VMS content and synonyms in the prompt
        parsed_data = "\n".join([f"Variant '{var}': {info['data']}" for var, info in vms_data.items()])
        synonyms = "\n".join([f"Variant '{var}': {', '.join(info['synonyms'])}" for var, info in vms_data.items() if info['synonyms']])
    else:
        parsed_data = "No VMS data found."
        synonyms = "No synonyms found."

    prompt = (
        f"Sinu ülesanne on leida sisestatud Gutslaffi sõnale '{input_word}' tänapäevased vasted. "
        f"Abivahendina võid sõnastikufailist leida Gutslaffi sõnale ka saksa ja/või ladina vaste. "
        f"{'Saksa või ladina vaste on ' + german_latin_equivalent if german_latin_equivalent else 'Vaste puudub'}. "
        f"Kui sa failist õiget sõnakuju ei leia, siis vaata VMSi tulemusi. "
        f"Tulemused VMSist on:\n{parsed_data}. "
        f"Leitud sünonüümid: {synonyms}. "
        f"Pea meeles, et sõnade kirjakuju võib pisut erineda ja seega tuleb tähendust tõlgendada. "
        f"Vasteid võib olla mitu, samuti lisa sünonüümid kui need on olemas. Tulemus väljasta kujul: "
        f"Gutslaffi sõna '{input_word}' vaste(d) tänapäeva kirjakeeles on '{{nowadays_word}}'."
    )

    # Generate content with Claude's API
    generated_text = generate_content_with_claude(prompt)
    
    # Print the result for the current word
    print(f"Result for '{input_word}':\n{generated_text}\n")
#Skript, mis otsib Gemini abil sisendfailis olevatele sõnadele vasteid Gutslaffi sõnastikust, kasutades ka DWDSi abi. 
#Autor: Eleri Aedmaa

import os
import pandas as pd
import google.generativeai as genai
import requests  # For handling DWDS fetching

# Set up API key
os.environ["GOOGLE_API_KEY"] = "your_key"

# Configure the Gemini API
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize the Gemini 1.5 Pro model
model = genai.GenerativeModel("gemini-1.5-pro")

# Function to fetch data from DWDS using the German equivalent
def fetch_from_dwds(word):
    url = f"https://www.dwds.de/wb/{word}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return f"Failed to fetch data for {word}. HTTP status code: {response.status_code}"
    except Exception as e:
        return f"An error occurred while fetching data from DWDS: {e}"

# File handling part with error catching
try:
    # Read the Excel file for Gutslaff version and German equivalents
    excel_file = 'Fail_murre_1.xlsx'
    df = pd.read_excel(excel_file)

    # Convert the DataFrame to a string for inclusion in the prompt
    file_content = df.to_string()

except FileNotFoundError:
    print(f"Error: The file '{excel_file}' was not found.")
    file_content = "Faili ei leitud."

except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    file_content = "Faili lugemise viga."

# File handling for the input words from 'sisend_murdes6nad.txt'
try:
    # Open the input file and read all words
    with open('sisend_murdes6nad.txt', 'r', encoding='utf-8') as file:
        input_words = [line.strip() for line in file.readlines()]

    # Loop through each word in the input file
    for input_word in input_words:
#        print(f"\nProcessing word: {input_word}")

        # Check if the modern word exists in the 'Gutslaff' column
        matched_rows = df[df['Gutslaff'] == input_word]

        if not matched_rows.empty:
            # If a match is found, extract the Gutslaff version and the German equivalent
            gutslaff_version = matched_rows['Gutslaff'].values[0]
            german_equivalent = matched_rows['sks'].values[0]
        else:
            # If no match is found, we'll try to fetch from DWDS using the German translation
            print(f"No match found for the modern word '{input_word}' in the Excel file.")
            gutslaff_version = "Ei leitud"
            german_equivalent = None  # We will try to find this from DWDS

        # If the German equivalent was not found in the Excel file, use DWDS to search for it
        if german_equivalent is None:
            german_equivalent = input_word  # Assuming input word may help in finding the translation in DWDS
            dwds_data = fetch_from_dwds(german_equivalent)
#            print(f"DWDS data for '{german_equivalent}': {dwds_data}")
        else:
            dwds_data = f"DWDS URL on https://www.dwds.de/wb/{german_equivalent}"

        # Create the prompt using the Gutslaff version (if found) and its German equivalent
        response = model.generate_content(f"You are an expert in analyzing old Estonian vocabulary. "
                                          f"Sinu ülesanne on leida sisestatud sõna '{input_word}' esinemiskujud Gutslaffi sõnastikufailist. "
                                          f"Gutslaffi sõnastikus on selle sõna vaste '{gutslaff_version}'. "
                                          f"Allpool on sõnastikufaili sisu:\n\n{file_content}\n\n"
                                          f"Kui sa failist õiget sõnakuju ei leia, siis võid otsida sõna tähendust DWDSi kaudu. "
                                          f"{dwds_data}. Vastuses esita sõnavorm täpselt sel kujul, "
                                          "nagu ta sõnastiku failis esineb. Tulemus väljasta kujul *Sõna "
                                          f"{input_word} esineb Gutslaffi sõnastikus kujul '{gutslaff_version}'.")

        # Get the result and print it for the current word
        print(response.text)

except FileNotFoundError:
    print(f"Error: The file 'sisend_murdes6nad.txt' was not found.")
except Exception as e:
    print(f"An error occurred while reading the input file: {e}")

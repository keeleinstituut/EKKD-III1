import os
import pandas as pd
import google.generativeai as genai
import requests  # For handling DWDS fetching

# Set up API key
os.environ["GOOGLE_API_KEY"] = "my-api-key-here"

# Configure the Gemini API
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize the Gemini 1.5 Pro model
model = genai.GenerativeModel("gemini-1.5-pro")

# File handling part with error catching
try:
    # Read the Excel file
    excel_file = 'Fail_katse2.xlsx'
    df = pd.read_excel(excel_file)
    
    # Convert the DataFrame to a string for inclusion in the prompt
    file_content = df.to_string()

except FileNotFoundError:
    print(f"Error: The file '{excel_file}' was not found.")
    file_content = "Faili ei leitud."

except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    file_content = "Faili lugemise viga."

# Read the text file
try:
    text_file = 'loend_katse2.txt'
    with open(text_file, 'r', encoding='utf-8') as file:
        words_to_search_content = file.read()

except FileNotFoundError:
    print(f"Error: The file '{text_file}' was not found.")
    words_to_search_content = "Tekstifaili ei leitud."

except Exception as e:
    print(f"An error occurred while reading the text file: {e}")
    words_to_search_content = "Tekstifaili lugemise viga."

# Function to fetch data from DWDS
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

# Set temperature in the request
temperature = 0.2  # A value between 0 and 1, higher values mean more randomness

# Create the prompt
response = model.generate_content(f"You are an expert in analyzing old Estonian vocabulary. Sinu ülesanne on leida tekstifailis ('loend_katse2.txt') esitatud sõnade esinemiskujud sõnastikufailist ('Fail_katse2.xlsx').\n\nAllpool on tekstifaili sisu:\n\n{words_to_search_content}\n\nAllpool on sõnastikufaili sisu:\n\n{file_content}\n\n Vastuses loetle ridade kaupa, millised tabelis esitatud vanad sõnad vastavad tekstifailis nimetatud ametitele, näiteks nii: * [vastuse rea nr] kohtunik on vanades sõnastikes kujul /sundija/. Allikas: Stahl, lk 101, real nr [siia kirjuta tabeli rea number]. Kui sa tekstifaili sõnade tähendust ei tea, otsi Sõnaveebist: https://sonaveeb.ee/search/unif/dlall/dsall/[siia kirjuta otsitav sõna ilma kantsulgudeta]. Tulemus väljasta txt-formaadis failina.")

# Get the result and write it to a txt file
output_file = 'gemini_response.txt'

try:
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(response.text)
    print(f"Output successfully written to {output_file}")

except Exception as e:
    print(f"An error occurred while writing the output to the file: {e}")
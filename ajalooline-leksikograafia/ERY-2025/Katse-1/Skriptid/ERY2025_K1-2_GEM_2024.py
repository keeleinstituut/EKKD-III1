import os
import pandas as pd
import google.generativeai as genai
import requests  # Make sure you import the requests library for handling DWDS fetching

# Set up API key
os.environ["GOOGLE_API_KEY"] = "my-api-key-here"

# Configure the Gemini API
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize the Gemini 1.5 Pro model
model = genai.GenerativeModel("gemini-1.5-pro")

# File handling part with error catching
try:
    # Read the Excel file
    excel_file = 'Fail_katse1.xlsx'
    df = pd.read_excel(excel_file)
#    print(df.head())
    
    # Convert the DataFrame to a string for inclusion in the prompt
    file_content = df.to_string()

except FileNotFoundError:
    print(f"Error: The file '{excel_file}' was not found.")
    file_content = "Faili ei leitud."

except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    file_content = "Faili lugemise viga."

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
response = model.generate_content(f"You are an expert in analyzing old Estonian vocabulary. Sinu ülesanne on leida sisestatud sõna esinemiskujud sõnastikufailist. Sisestatud sõna on 'hirsnik', mille sünonüüm on 'õiguseleidja'. Allpool on sõnastikufaili sisu:\n\n{file_content}\n\n Vastuses esita sõnavorm täpselt sel kujul, nagu ta sõnastiku failis esineb. Nimeta autor ja lehekülg. Sisestatud sõna tähendust võid kontrollida Sõnaveebist: https://sonaveeb.ee/search/unif/dlall/dsall/hirsnik Tulemus väljasta kujul Sõna *[siia kirjuta sisestatud sõna]* esineb [autori nimi] real [siia kirjuta rea number] kujul [siia kirjuta otsitava sõna vana esinemiskuju täpselt samal moel, nagu ta esineb failis] (leheküljel [siia kirjuta leheküljenumber sama rea väljalt päisega lk]). Pärast tulemuse väljastamist selgita, kuidas sa selle tulemuseni jõudsid. Kui sa oled vastuses ebakindel, siis ütle, et vastus puudub.")

# Get the result and print it
print(response.text)

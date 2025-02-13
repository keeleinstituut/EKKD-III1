import os
import pandas as pd
import google.generativeai as genai

# Set up API key
os.environ["GOOGLE_API_KEY"] = "my-api-key-here"

# Configure the Gemini API
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize the Gemini 1.5 Pro model
model = genai.GenerativeModel("gemini-1.5-pro")

# Read the Excel file
excel_file = 'Fail_katse1.xlsx'
df = pd.read_excel(excel_file)
print
(df.head())

# Convert the DataFrame to a string for inclusion in the prompt
file_content = df.to_string()

# Function to fetch data from DWDS
def fetch_from_dwds(word):
    url = f"https://www.dwds.de/wb/{word}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return "Failed to fetch data"

# Create the prompt
response = model.generate_content("You are an expert in analyzing old Estonian vocabulary. Sinu ülesanne on leida sisestatud sõna esinemiskujud sõnastikufailist. Sisestatud sõna on 'kohtunik'. Allpool on sõnastikufaili sisu:\n\n{file_content}\n\n Kui sa failist õiget sõnakuju ei leia, siis võid otsida sõna tähendust DWDSi kaudu. DWDSi URL on https://www.dwds.de/wb/Richter. Vastuses esita sõnavorm täpselt sel kujul, nagu ta sõnastiku failis esineb. Nimeta autor ja lehekülg. Tulemus väljasta kujul *Sõna kohtunik esineb [autori nimi] real kujul [siia kirjuta otsitava sõna vana esinemiskuju täpselt samal moel, nagu ta esineb failis] (leheküljel [siia kirjuta leheküljenumber sama rea väljalt päisega lk]).")

# Get the result and print it
print(response.text)
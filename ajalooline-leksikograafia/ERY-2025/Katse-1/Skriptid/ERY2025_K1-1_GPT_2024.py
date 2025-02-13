import requests
import pandas as pd

# Read the Excel file
excel_file = 'Fail_katse1.xlsx'
df = pd.read_excel(excel_file)

# Convert the DataFrame to a string for inclusion in the prompt
file_content = df.to_string()

# Define your API key
api_key = 'my-api-key-here'

# Define the URL and headers for the OpenAI API request
url = 'https://api.openai.com/v1/chat/completions'

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}',
}

# Prepare the data for the API request
data = {
    "model": "gpt-4o",  # model name
    "messages": [
        {"role": "system", "content": "You are an expert in analyzing old Estonian vocabulary."},
        {
            "role": "user",
            "content": (
                f"Sinu ülesanne on leida sisestatud sõna esinemiskujud sõnastikufailist. Sisestatud sõna on 'kohtunik'. Allpool on sõnastikufaili sisu:\n\n{file_content}\n\n Kui sa failist õiget sõnakuju ei leia, siis võid otsida sõna tähendust DWDSi kaudu. DWDSi URL on https://www.dwds.de/wb/Richter. Vastuses esita sõnavorm täpselt sel kujul, nagu ta sõnastiku failis esineb. Nimeta autor ja lehekülg. Tulemus väljasta kujul *Sõna kohtunik esineb [autori nimi] real kujul [siia kirjuta otsitava sõna vana esinemiskuju täpselt samal moel, nagu ta esineb failis] (leheküljel [siia kirjuta leheküljenumber sama rea väljalt päisega lk])."
            )
        }
    ],
    "max_tokens": 10000,
    "temperature": 0.2
}

# Make the API request
response = requests.post(url, headers=headers, json=data)

# Get the result and print it
result = response.json()
print(result['choices'][0]['message']['content'])
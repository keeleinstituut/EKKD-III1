import requests
import pandas as pd
import json

# Read the Excel file
excel_file = 'Fail_katse1.xlsx'
df = pd.read_excel(excel_file)

# Convert the DataFrame to a string for inclusion in the prompt
file_content = df.to_string()

# Define your API key
api_key = "my-api-key-here"

# Define the URL and headers for the Anthropic Claude API request
url = 'https://api.anthropic.com/v1/messages'

headers = {
    'Content-Type': 'application/json',
    'x-api-key': api_key,
    'anthropic-version': '2023-06-01'
}

# Prepare the data for the API request
data = {
    "model": "claude-3-opus-20240229",
    "max_tokens": 4096,
    "temperature": 0.2,
    "messages": [
        {
            "role": "user",
            "content": f"You are an expert in analyzing old Estonian vocabulary. Sinu ülesanne on leida sisestatud sõna esinemiskujud sõnastikufailist. Sisestatud sõna on 'hirsnik', mille sünonüüm on 'õiguseleidja'. Allpool on sõnastikufaili sisu:\n\n{file_content}\n\n Vastuses esita sõnavorm täpselt sel kujul, nagu ta sõnastiku failis esineb. Nimeta autor ja lehekülg. Sisestatud sõna tähendust võid kontrollida Sõnaveebist: https://sonaveeb.ee/search/unif/dlall/dsall/hirsnik Tulemus väljasta kujul Sõna *[siia kirjuta sisestatud sõna]* esineb [autori nimi] real [siia kirjuta rea number] kujul [siia kirjuta otsitava sõna vana esinemiskuju täpselt samal moel, nagu ta esineb failis] (leheküljel [siia kirjuta leheküljenumber sama rea väljalt päisega lk]). Pärast tulemuse väljastamist selgita, kuidas sa selle tulemuseni jõudsid. Kui sa oled vastuses ebakindel, siis ütle, et vastus puudub."
        }
    ]
}

try:
    # Make the API request
    response = requests.post(url, headers=headers, json=data)
    
    # Check if the request was successful
    response.raise_for_status()
    
    # Get the result and print it
    result = response.json()
    print("Full API Response:")
    print(json.dumps(result, indent=2))
    
    # Try to access the content
    if 'content' in result:
        print("\nContent:")
        print(result['content'])
    else:
        print("\nContent key not found. Available keys:")
        print(list(result.keys()))
        
        # If there's a 'message' key, it might contain error information
        if 'message' in result:
            print("\nError message:")
            print(result['message'])

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
    if response.text:
        print("Response text:")
        print(response.text)

except json.JSONDecodeError:
    print("Failed to decode JSON. Raw response:")
    print(response.text)

except Exception as e:
    print(f"An unexpected error occurred: {e}")
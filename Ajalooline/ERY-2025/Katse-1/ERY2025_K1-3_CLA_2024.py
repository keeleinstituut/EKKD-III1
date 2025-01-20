import requests
import pandas as pd
import json

# Read the Excel file
excel_file = 'Fail_katse2.xlsx'
df = pd.read_excel(excel_file)

# Convert the DataFrame to a string for inclusion in the prompt
file_content = df.to_string()

# Read the txt file
txt_file = 'loend_katse2.txt'
with open(txt_file, 'r', encoding='utf-8') as f:
    words_to_search_content = f.read()

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
            "content": f"You are an expert in analyzing old Estonian vocabulary. Sinu ülesanne on leida tekstifailis ('loend_katse2.txt') esitatud sõnade esinemiskujud sõnastikufailist ('Fail_katse2.xlsx').\n\nAllpool on tekstifaili sisu:\n\n{words_to_search_content}\n\nAllpool on sõnastikufaili sisu:\n\n{file_content}\n\n Vastuses loetle ridade kaupa, millised tabelis esitatud vanad sõnad vastavad tekstifailis nimetatud ametitele, näiteks nii: * [vastuse rea nr] kohtunik on vanades sõnastikes kujul /sundija/. Allikas: Stahl, lk 101, real nr [siia kirjuta tabeli rea number]. Kui sa tekstifaili sõnade tähendust ei tea, otsi Sõnaveebist: https://sonaveeb.ee/search/unif/dlall/dsall/[siia kirjuta otsitav sõna ilma kantsulgudeta]. Tulemus väljasta txt-formaadis failina."
        }
    ]
}

try:
    # Make the API request
    response = requests.post(url, headers=headers, json=data)
    
    # Check if the request was successful
    response.raise_for_status()
    
    # Get the result
    result = response.json()
    
    # Access the content
    if 'content' in result:
        content = result['content'][0]['text']
        
        # Write the content to a txt file
        output_file = 'vastused.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Results have been written to {output_file}")
    else:
        print("Content key not found. Available keys:")
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
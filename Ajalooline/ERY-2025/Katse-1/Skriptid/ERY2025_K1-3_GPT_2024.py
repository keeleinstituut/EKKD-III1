import requests
import pandas as pd

# Read the Excel file
excel_file = 'Fail_katse1.xlsx'
df = pd.read_excel(excel_file)

# Convert the DataFrame to a string for inclusion in the prompt
file_content = df.to_string()

# Read the txt file containing words to search for
txt_file = 'loend_katse2.txt'
with open(txt_file, 'r', encoding='utf-8') as f:
    words_to_search = f.read().splitlines()  # Each word in a new line

# Convert the list of words into a formatted string for the prompt
words_to_search_content = "\n".join(words_to_search)

# Define your API key as a string
api_key = 'my-api-key-here'

# Define the URL and headers for the OpenAI API request
url = 'https://api.openai.com/v1/chat/completions'

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}',
}

# Prepare the data for the API request
data = {
    "model": "chatgpt-4o-latest",
    "messages": [
        {"role": "system", "content": "You are an expert in analyzing old Estonian vocabulary."},
        {
            "role": "user",
            "content": (
                f"Sinu ülesanne on leida tekstifailis ('loend_katse2.txt') esitatud sõnade esinemiskujud sõnastikufailist ('Fail_katse1.xlsx').\n\nAllpool on tekstifaili sisu:\n\n{words_to_search_content}\n\nAllpool on sõnastikufaili sisu:\n\n{file_content}\n\n Vastuses loetle ridade kaupa, millised tabelis esitatud vanad sõnad vastavad tekstifailis nimetatud ametitele, näiteks nii: * [vastuse rea nr] kohtunik on vanades sõnastikes kujul /sundija/. Allikas: Stahl, lk 101, real nr [siia kirjuta tabeli rea number]. Kui sa tekstifaili sõnade tähendust ei tea, otsi Sõnaveebist: https://sonaveeb.ee/search/unif/dlall/dsall/[siia kirjuta otsitav sõna ilma kantsulgudeta]. Tulemus väljasta txt-formaadis failina."
            )
        }
    ],
    "max_tokens": 10000,
    "temperature": 0.2
}

# Make the API request
response = requests.post(url, headers=headers, json=data)

# Check for a successful response
if response.status_code == 200:
    # Get the result from the response
    result = response.json()
    
    # Extract the message content from the result
    api_response_content = result['choices'][0]['message']['content']
    
    # Save the result to a txt file
    output_file = 'api_result.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(api_response_content)

    print(f"API response has been saved to {output_file}")
else:
    print(f"Failed to get a response. Status code: {response.status_code}")
    print(f"Error details: {response.text}")
import pandas as pd
import requests
import time
import os
import json

# Set up Claude API key
CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY')
if not CLAUDE_API_KEY:
    raise ValueError("Please set the CLAUDE_API_KEY environment variable")

# Function to generate Claude content with retry logic
def generate_content_with_claude(word, stahl_df, goseken_df, gutslaff_df, retries=3, backoff=2):
    # Convert the dictionary dataframes to lists of words and equivalents
    stahl_data = stahl_df.dropna().values.tolist()
    goseken_data = goseken_df.dropna().values.tolist()
    gutslaff_data = gutslaff_df.dropna().values.tolist()
    
    # Construct the prompt
    prompt = (
        f"Sulle on antud tänapäevane eesti sõna '{word}' ja kolm sõnaraamatut: Stahl, Göseken ja Gutslaff. "
        f"Iga sõnaraamat sisaldab vanemaid eesti keele sõnu, nii et sünonüümid on komadega eraldatud. Sinu ülesanne on leida kõige lähemad sõnad nendes sõnaraamatutes, "
        f"võttes arvesse nii tähendust kui ka kirjapilti. Kasuta oma keelelisi teadmisi ja tööriistu, et leida tähenduslikke ja lingvistilisi sarnasusi. Esita vastusesse kõik sünonüümid ehk ära kaota sõnastikes olevaid sõnu ära. Kui sa ei leia vastet, siis ütle, et vastet ei leidu aga lisa selle kohta selgitus. "
        f"Sõnaraamatu sõnade kuju ei tohi muuta.\n\n"
        f"Esita väljund ühel real järgmises formaadis, veergude vahele jäta semikoolon:\n"
        f"1. veerg: tänapäevane sõna\n"
        f"2. veerg: Stahli vaste\n"
        f"3. veerg: Gösekeni vaste\n"
        f"4. veerg: Gutslaffi vaste\n"
        f"5. veerg: Lühike selgitus, kuidas vastuseni jõudsid.\n\n"
        f"Siin on sõnaraamatu kirjed:\n\n"
        f"Stahli sõnaraamat: {stahl_data}\n"
        f"Gösekeni sõnaraamat: {goseken_data}\n"
        f"Gutslaffi sõnaraamat: {gutslaff_data}\n\n"
        f"Tänapäevane sõna: {word}\n"
        f"Palun anna iga sõnaraamatu jaoks kõik vasteid ja vajadusel lühike selgitus."
    )

    headers = {
        "Content-Type": "application/json",
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01"
    }

    data = {
        "model": "claude-3-opus-20240229",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.2
    }

    for attempt in range(retries):
        try:
            response = requests.post("https://api.anthropic.com/v1/messages", json=data, headers=headers)
            response.raise_for_status()
            return response.json()['content'][0]['text'].strip()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if response.text:
                print(f"Response content: {response.text}")
            if attempt < retries - 1:
                print(f"Retrying in {backoff} seconds...")
                time.sleep(backoff)
            else:
                return f"Error generating content after {retries} attempts: {str(e)}"

    return "Failed after retries."

# Main process
try:
    # Load data
    tnp_df = pd.read_excel('TNP.xlsx', header=None)
    stahl_df = pd.read_excel('Stahl.xlsx', header=None)
    goseken_df = pd.read_excel('Göseken.xlsx', header=None)
    gutslaff_df = pd.read_excel('Gutslaff.xlsx', header=None)

    # Loop through input words and ask Claude to find equivalents
    for word in tnp_df[0].dropna():
        # Generate content using Claude to reason about both meaning and spelling-based matches
        result = generate_content_with_claude(word, stahl_df, goseken_df, gutslaff_df)
        
        # Output result
        print(f"Modern word: {word}\nResult:\n{result}\n")

except Exception as e:
    print(f"Error: {e}")
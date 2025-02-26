# Kood EKKD-III1 registrite töörühma esimese katse päringute tegemiseks Gemini 1.0 Ultra mudelilt.
# Autor: Eleri Aedmaa
import google.generativeai as genai
import os

# Configure the API key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Select a model that's available in your account
model = genai.GenerativeModel('gemini-1.0-ultra')

# System instruction
system_instruction = "Sa oled eesti keele sõnaraamatu koostaja. Millistes tekstides kasutatakse sisestatud eesti(keelset) sõna? Kui sul ei ole selle kohta informatsiooni, siis ütle, et sa ei oska öelda."

# Read input from file
with open('katse1_sisend.txt', 'r') as file:
    lines = file.readlines()

# Process each word
for line in lines:
    word = line.strip()
    if not word:  # Skip empty lines
        continue
    
    # Combine the instruction and word
    prompt = f"{system_instruction}\n\nSõna: {word}"
    
    # Send request to API
    try:
        response = model.generate_content(prompt)
        
        # Print just the response text
        if response.text:
            print(response.text)
        else:
            print("No response text received.")
    except Exception as e:
        print(f"Error processing '{word}': {e}")
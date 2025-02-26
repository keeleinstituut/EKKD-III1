import anthropic
import os

# Ensure the API key is set in the environment
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# Read input from file
with open('katse1_sisend.txt', 'r') as file:
    lines = file.readlines()

# Process each word
for line in lines:
    word = line.strip()
    if not word:  # Skip empty lines
        continue
        
    # Send request to API
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=0.0,
            system="Sa oled eesti keele sõnaraamatu koostaja. Millistes tekstides kasutatakse sisestatud eesti(keelset) sõna? Kui sul ei ole selle kohta informatsiooni, siis ütle, et sa ei oska öelda.",
            messages=[{"role": "user", "content": word}]
        )
        
        # Print just the response text
        if isinstance(response.content, list) and response.content:
            print(response.content[0].text)
        else:
            print(response.content)
            
    except Exception as e:
        print(f"Error processing '{word}': {e}")
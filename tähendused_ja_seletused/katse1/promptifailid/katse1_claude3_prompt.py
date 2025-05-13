#Kood EKKD-III1 tähenduste töörühma esimese katse päringute tegemiseks Anthropicu mudelitelt Claude 3 Opus ja Sonnet.
#Autor: Eleri Aedmaa
import anthropic
import os

# API võti
# os.environ["ANTHROPIC_API_KEY"] = ""

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# sisendfail
with open('katse1_sisend.txt', 'r') as file:
    lines = file.readlines()

# sõnum
messages = []
for line in lines:
    line = line.strip()
    if line:  # tühjad read
        messages.append({"role": "user", "content": line})

responses = []
for i in range(0, len(messages), 2):
    user_message = messages[i]
    assistant_message = messages[i + 1] if i + 1 < len(messages) else {"role": "assistant", "content": "Sa oled eesti keele sõnaraamatu koostaja"}
    
    response = client.messages.create(
        model="claude-3-sonnet-20240229", #muuda vajadusel mudeli nime
        max_tokens=4000,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": "Sa oled eesti keele sõnaraamatu koostaja. Mis on eesti keeles järgmiste sõnade tähendused? Mitmetähenduslikele sõnadele võid anda mitu tähendust. Kui sa ei tea, siis ütle, et sa ei tea."
            },
            user_message
        ]
    )
    )
    
    responses.append(response.content)

# Print all responses
for response in responses:
    print(response)

#Kood EKKD-III1 registrite töörühma esimese katse päringute tegemiseks OpenAI mudelitelt.
#Autor: Eleri Aedmaa
from openai import OpenAI

def read_inputs_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines if line.strip()]  

def get_response_for_input(client, user_input):
    response = client.chat.completions.create(
        model="gpt-4o", #muuda vajadusel mudeli nime
        messages=[
            {
                "role": "system",
                "content": "Sa oled eesti keele sõnaraamatu koostaja. Mida tähendab eesti keeles see sõna? Mitmetähenduslikele sõnadele võid anda mitu tähendust. Kui sa ei tea, siis ütle, et sa ei tea."
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=0.2,
        max_tokens=4096,
        top_p=1
    )
    return response.choices[0].message.content

def main():
    client = OpenAI()
    file_path = 'katse1_sisend.txt'  # sisendsõnad eraldi real
    user_inputs = read_inputs_from_file(file_path)
    
    for user_input in user_inputs:
        response_message = get_response_for_input(client, user_input)
        print("Input:", user_input)
        print("Response:", response_message)
        print("-----------")  # eraldaja

if __name__ == "__main__":
    main()

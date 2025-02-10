#Kood EKKD-III1 registrite töörühma esimese katse päringute tegemiseks OpenAI mudelilt GPT-4o.
#Autor: Eleri Aedmaa
from openai import OpenAI

def read_inputs_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines if line.strip()]  # eemalda tühjad read

def get_response_for_input(client, user_input):
    response = client.chat.completions.create(
        model="gpt-4o", #vaheta vajadusel mudeli nime (ajas muutuvad)
        messages=[
            {
                "role": "Sa oled eesti keele sõnaraamatu koostaja",
                "content": "Millistes tekstides kasutatakse sisestatud eesti(keelset) sõna? Kui sul ei ole selle kohta informatsiooni, siis ütle, et sa ei oska öelda."
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
    file_path = 'katse1_sisend.txt'  # sisendfail sõnadega
    user_inputs = read_inputs_from_file(file_path)
    
    for user_input in user_inputs:
        response_message = get_response_for_input(client, user_input)
        print("Input:", user_input)
        print("Response:", response_message)
        print("-----------")  # vastuste eristamiseks

if __name__ == "__main__":
    main()

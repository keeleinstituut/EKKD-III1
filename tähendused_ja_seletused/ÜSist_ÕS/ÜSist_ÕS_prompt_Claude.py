import anthropic
import csv
import pandas as pd

def read_inputs_from_file(file_path):
    try:
        return pd.read_csv(file_path, delimiter='\t', usecols=['Katsesõna', 'ÜS_definitsioon'])
    except Exception as e:
        raise ValueError(f"Sisendfaili lugemisel tekkis viga: {e}")

def get_response_for_input(api_key, word, us_definition):
    client = anthropic.Client(api_key=api_key)
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        system=f"Sa oled eesti keele sõnaraamatu koostaja. Kirjuta sõnale '{word}' ÕSi definitsioon, lähtudes antud ÜS definitsioonist. "
               "Seletus kirjuta samas stiilis nagu definitsioon ÕS."
               "Järgi näidete stiili:\n\n"
               "**huumor**\n"
               "definitsioon ÜS - heatahtlik nali, koomiliste elunähtuste, sündmuste või inimeste puuduste ja nõrkuste heatahtlik naeruvääristamine\n"
               "definitsioon ÕS - heatahtlik nali\n"
               "**kaste**\n"
               "definitsioon ÜS - vedel või poolvedel lisand, mis muudab toidu mahlasemaks ja maitsvamaks\n"
               "definitsioon ÕS - soust\n"
               "**erinev**\n"
               "definitsioon ÜS - mingil viisil teistest, muust eristuv\n"
               "definitsioon ÕS - muust eristuv\n"
               "**eeldama**\n"
               "definitsioon ÜS - eeltingimusena mingit omadust, oskust vm asjaolu vajama või ootama\n"
               "definitsioon ÕS - eeldusena nõudma\n"
               "**juures**\n"
               "definitsioon ÜS - kellegi või millegi vahetus läheduses, hästi lähedal\n"
               "definitsioon ÕS - vahetus läheduses\n"
               "Vastus peab olema järgmisel kujul:\n"
               "definitsioon ÕS - [ÕS stiilis seletus]",
        messages=[{
            "role": "user",
            "content": f"Sõna: {word}\nÜS definitsioon: {us_definition}"
        }],
        temperature=0.2,
        max_tokens=4096
    )
    print(f"Claude vastus sõnale '{word}': {response.content}")
    return response.content

def process_response(api_key, word, us_definition):
    try:
        response = get_response_for_input(api_key, word, us_definition)
        text = response[0].text if isinstance(response, list) else response
        lines = text.split('\n')
        
        os_def = ""
        
        for line in lines:
            if line.startswith("definitsioon ÕS -"):
                os_def = line.replace("definitsioon ÕS -", "").strip()
                
        return pd.Series([os_def])
    except Exception as e:
        return pd.Series([f"Töötlemise viga: {e}"])

def main():
    api_key = ""
    input_file_path = 'ekkd_ii_t2hendusvihjeta_seletusega.csv'
    output_file_path = 't2hendusvihjeta_valjund.csv'
    
    try:
        user_inputs = read_inputs_from_file(input_file_path)
        
        if 'Katsesõna' not in user_inputs.columns or 'ÜS_definitsioon' not in user_inputs.columns:
            raise ValueError("Sisendfail peab sisaldama veerge 'Katsesõna' ja 'ÜS_definitsioon'.")
            
        user_inputs[['ÕS_definitsioon']] = user_inputs.apply(
            lambda row: process_response(api_key, row['Katsesõna'], row['ÜS_definitsioon']),
            axis=1
        )
        
        user_inputs.to_csv(output_file_path, index=False, quoting=csv.QUOTE_MINIMAL, sep='\t')
        print(f"Tulemused salvestatud faili: {output_file_path}")
    except Exception as e:
        print(f"Tekkis viga: {e}")

if __name__ == "__main__":
    main()
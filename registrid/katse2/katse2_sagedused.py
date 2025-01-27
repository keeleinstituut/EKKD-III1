# Kood EKKD-III1 registrite töörühma teise katse märksõnade sageduste hankimiseks
# eesti keele ühendkorpusest Sketch Engine'i API kaudu.
# Autorid: Esta Prangel, Eleri Aedmaa

import requests
import pandas as pd
import os
from dotenv import load_dotenv


load_dotenv(".env")

username = os.getenv("USER")
api_key = os.getenv("KEY")


def get_word_stats(word):
    url = 'https://api.sketchengine.eu/bonito/run.cgi/wordlist'
    params = {
        'corpname': 'preloaded/estonian_nc23',
        'wlattr': 'lemma',
        'wltype': 'simple',
        'wlpat': word,
        'format': 'json'
    }
    response = requests.get(url, params=params, auth=(username, api_key))
    return response.json()


def get_genre_distribution(word):
    q = 'q[lemma="' + word + '"]'
    url = 'https://api.sketchengine.eu/bonito/run.cgi/freqs'
    params = {
        'corpname': 'preloaded/estonian_nc23',
        'fcrit': 'doc.genre 0',
        'q': q,
        'format': 'json'
    }
    response = requests.get(url, params=params, auth=(username, api_key))
    return response.json()


df = pd.read_excel('katse2_sisend.xlsx', usecols=[0, 1], names=['märksõna', 'tähendus'])

data = []

for _, row in df.iterrows():
    w = row['märksõna']
    col_b = row['tähendus']

    try:
        word_row = {'märksõna': w, 'tähendus': col_b}
        word_data = get_word_stats(w)
        if 'Items' in word_data and word_data['Items']:
            item = word_data['Items'][0]
            word_row['kogusagedus'] = item['frq']
            word_row['suhteline sagedus'] = item['relfreq']
        else:
            word_row['kogusagedus'] = 0
            word_row['suhteline sagedus'] = 0

        genre_data = get_genre_distribution(w)

        if 'Blocks' in genre_data:
            for block in genre_data['Blocks']:
                if 'Items' in block:
                    for item in block['Items']:
                        genre_list = item.get('Word', 'tundmatu')
                        if isinstance(genre_list, list) and genre_list:
                            genre_name = genre_list[0].get("n", "tundmatu")
                        else:
                            genre_name = "unknown"
                        genre_frq = item.get('frq', 0)
                        genre_rel = item.get('rel', 0)
                        word_row[genre_name + '_abs'] = genre_frq
                        word_row[genre_name + '_rel'] = genre_rel
        print(word_row)
        data.append(word_row)

    except Exception as e:
        print(f"Viga sõnaga '{w}': {str(e)}")

output_df = pd.DataFrame(data)

output_df.columns = output_df.columns.str.replace('===NONE===', 'none', regex=False)

columns_to_keep = [col for col in output_df.columns if col.startswith(('märksõna',
                                                                       'tähendus',
                                                                       'kogusagedus',
                                                                       'suhteline sagedus',
                                                                       'blogs',
                                                                       'forums',
                                                                       'periodicals',
                                                                       'none'))]

output_df = output_df[columns_to_keep]

file_name = 'sagedused_se.xlsx'
output_df.to_excel(file_name, index=False)
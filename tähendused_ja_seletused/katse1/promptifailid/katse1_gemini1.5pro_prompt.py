#Kood EKKD-III1 tähenduste töörühma esimese katse päringute tegemiseks Google'i Gemini mudelitelt.
#Autor: Eleri Aedmaa

import os
import google.generativeai as genai

def loo_sonaraamat(sisendfaili_nimi='katse1_sisend.txt'):
    """Loeb sõnad sisendfailist, küsib Gemini API-lt nende tähendused ja prindib tulemused."""

    # API võtme seadistamine
    api_võti = os.getenv("GENAI_API_KEY")
    if not api_võti:
        raise ValueError("API võti peab olema seadistatud keskkonnamuutujas 'GENAI_API_KEY'.")

    # Gemini API kliendi seadistamine
    genai.configure(api_key=api_võti)

    # Mudeli valimine
    mudel = genai.GenerativeModel('gemini-1.5-pro')

    # Süsteemi juhised
    süsteemi_juhised = "Sa oled eesti keele sõnaraamatu koostaja. Mis on eesti keeles järgmiste sõnade tähendused? Mitmetähenduslikele sõnadele võid anda mitu tähendust. Kui sa ei tea, siis ütle, et sa ei tea."

    try:
        # Sisendfaili lugemine
        with open(sisendfaili_nimi, 'r', encoding='utf-8') as fail:
            read = fail.readlines()
            sonad = [sõna.strip() for sõna in read if sõna.strip()]

        if not sonad:
            print(f"Hoiatus: Sisendfail '{sisendfaili_nimi}' on tühi.")
            return

        # Vastuste kogumine ja printimine
        for sõna in sonad:
            viip = f"{süsteemi_juhised}\n\nSõna: {sõna}"
            try:
                vastus = mudel.generate_content(
                    viip,
                    generation_config={
                        "max_output_tokens": 4000,
                        "temperature": 0.0,
                    },
                )

                if vastus.text:
                   print(f"Sõna: {sõna}\nTähendus: {vastus.text}\n{'-' * 20}")
                else:
                    print(f"Sõna: {sõna}\nTähendust ei leitud.\n{'-' * 20}")

            except Exception as e:
                print(f"Viga sõna '{sõna}' töötlemisel: {e}")

    except FileNotFoundError:
        print(f"Viga: Sisendfaili '{sisendfaili_nimi}' ei leitud.")
    except Exception as e:
        print(f"Ootamatu viga: {e}")

if __name__ == "__main__":
    loo_sonaraamat()
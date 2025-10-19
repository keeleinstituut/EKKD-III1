import pandas as pd
import openai
import time

# Set up OpenAI API key
openai.api_key = 'oma_võti_siia'


# Function to generate GPT content 
def generate_content_with_gpt(word, stahl_df, goseken_df, gutslaff_df, retries=3, backoff=2):
    # Convert the dictionary dataframes to lists of words and equivalents
    stahl_data = stahl_df.dropna().values.tolist()
    goseken_data = goseken_df.dropna().values.tolist()
    gutslaff_data = gutslaff_df.dropna().values.tolist()

    # Construct the prompt to provide the word lists to the model
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
    )

    for _ in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "Oled leksikograaf, kes uurib vana eesti keelt."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,  # Adjust token limit if necessary
                temperature=0.2  # Keep the temperature low for more accurate results
            )
            return response['choices'][0]['message']['content'].strip()
        except openai.error.RateLimitError:
            time.sleep(backoff)
        except Exception:
            return "Error generating content."
    return "Failed after retries."

# Main process
try:
    # Load data
    tnp_df = pd.read_excel('TNP.xlsx', header=None)
    stahl_df = pd.read_excel('Stahl.xlsx', header=None)
    goseken_df = pd.read_excel('Göseken.xlsx', header=None)
    gutslaff_df = pd.read_excel('Gutslaff.xlsx', header=None)

    # Loop through input words and ask GPT to find equivalents
    for word in tnp_df[0].dropna():
        # Generate content using GPT to reason about both meaning and spelling-based matches
        result = generate_content_with_gpt(word, stahl_df, goseken_df, gutslaff_df)
        
        # Output result
        print(f"Modern word: {word}\nResult:\n{result}\n")

except Exception as e:
    print(f"Error: {e}")


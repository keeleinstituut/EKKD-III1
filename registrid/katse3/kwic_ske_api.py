#Kood kontekstide pärimiseks Sketch Engine'i APIt kasutades.
#Sisendiks fail, iga sõna eraldi real
#Autor: Eleri Aedmaa

import requests
import json
import os
from typing import Dict, List, Optional

class SketchEngineAPI:
    def __init__(self, username: str, api_key: str):
        """
        Initialize Sketch Engine API client
        
        Args:
            username: Your Sketch Engine username
            api_key: Your Sketch Engine API key
        """
        self.username = username
        self.api_key = api_key
        self.base_url = "https://api.sketchengine.eu/bonito/run.cgi"
        
    def get_concordances_ui_simple(self, 
                                  corpus: str = "preloaded/estonian_nc23", #korpuse nimi vajadusel muuta
                                  query: str = "huulekas",
                                  pagesize: int = 1000,
                                  fromp: int = 1,
                                  leftctx: str = "3",
                                  rightctx: str = "3") -> Dict:
        """
        Get concordances using the UI simple method
        
        Args:
            corpus: Corpus name
            query: Search query
            pagesize: Number of results per page
            fromp: Starting page
            leftctx: Left context in sentences
            rightctx: Right context in sentences
            
        Returns:
            Dictionary containing concordance results
        """
        # simple otsing
        ui_simple_query = f'q[word="{query}" | lemma="{query}" | lc="{query.lower()}" | lemma_lc="{query.lower()}"]'
        
        params = {
            'corpname': corpus,
            'q': ui_simple_query,
            'pagesize': pagesize,
            'fromp': fromp,
            'format': 'json',
            'kwicleftctx': f'-{leftctx}:s',
            'kwicrightctx': f'{rightctx}:s',
            'username': self.username,
            'api_key': self.api_key,
            'async': '0'
        }
        
        try:
            response = requests.get(self.base_url + "/view", params=params)
            response.raise_for_status()
            return response.json()
                
        except requests.exceptions.RequestException as e:
            print(f"Error making API request: {e}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            return {}
    

    
    def _get_all_pages_ui_simple(self, corpus, query, leftctx, rightctx):
        """Get all pages using the ui_simple method"""
        all_concordances = []
        page = 1
        pagesize = 1000
        
        print(f"Fetching all pages for '{query}'...")
        
        while True:
            print(f"Fetching page {page}... ({len(all_concordances)} collected so far)")
            
            result = self.get_concordances_ui_simple(
                corpus=corpus,
                query=query,
                pagesize=pagesize,
                fromp=page,
                leftctx=leftctx,
                rightctx=rightctx
            )
            
            if not result or 'Lines' not in result:
                break
                
            lines = result['Lines']
            if not lines:  # No more results
                break
                
            all_concordances.extend(lines)
            
            # Kontrolli, kas kõik vasted saadi kätte
            total_hits = result.get('fullsize', 0)
            if len(all_concordances) >= total_hits:
                print(f"✅ Collected all {len(all_concordances)} concordances!")
                break
                
            page += 1
            
            # Turvalisus
            if page > 300:  # Max ~100,000 results
                print(f"⚠️ Stopped at page {page} for safety")
                break
                
        print(f"Total concordances retrieved: {len(all_concordances)}")
        return all_concordances
    
    def save_concordances(self, 
                         concordances, 
                         filename: str):
        """
        Save concordances to a JSON file
        
        Args:
            concordances: Concordance data (Dict or List)
            filename: Output filename
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(concordances, f, ensure_ascii=False, indent=2)
            print(f"Concordances saved to {filename}")
        except Exception as e:
            print(f"Error saving file: {e}")
    
    def save_concordances_csv(self, 
                            concordances: List[Dict], 
                            filename: str):
        """
        Save concordances to a CSV file for easier analysis
        
        Args:
            concordances: List of concordance data
            filename: Output filename
        """
        try:
            import csv
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['ID', 'Left_Context', 'Keyword', 'Right_Context', 'Full_Context'])
                
                for i, line in enumerate(concordances, 1):
                    # eralda kontekstid
                    left = line.get('Left', [])
                    kwic = line.get('Kwic', [])
                    right = line.get('Right', [])
                    
                    # Convert to text
                    left_text = ' '.join([token.get('str', '') for token in left]).strip()
                    kwic_text = ' '.join([token.get('str', '') for token in kwic]).strip()
                    right_text = ' '.join([token.get('str', '') for token in right]).strip()
                    full_text = f"{left_text} {kwic_text} {right_text}".strip()
                    
                    writer.writerow([i, left_text, kwic_text, right_text, full_text])
            
            print(f"Concordances saved to CSV: {filename}")
        except Exception as e:
            print(f"Error saving CSV file: {e}")
    
    def save_full_context_only(self, 
                              concordances: List[Dict], 
                              filename: str):
        """
        Save only the full context (complete sentences) to a plain text file
        One concordance per line, no additional formatting or metadata
        
        Args:
            concordances: List of concordance data
            filename: Output filename
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for line in concordances:
                    # Extract all contexts
                    left = line.get('Left', [])
                    kwic = line.get('Kwic', [])
                    right = line.get('Right', [])
                    
                    # Convert to text and combine
                    left_text = ' '.join([token.get('str', '') for token in left]).strip()
                    kwic_text = ' '.join([token.get('str', '') for token in kwic]).strip()
                    right_text = ' '.join([token.get('str', '') for token in right]).strip()
                    
                    # Create full context
                    full_text = f"{left_text} {kwic_text} {right_text}".strip()
                    
                    # Write only the full context, nothing else
                    f.write(full_text + '\n')
            
            print(f"Full context saved to: {filename}")
        except Exception as e:
            print(f"Error saving full context file: {e}")
    
    def print_concordances(self, concordances, limit: int = 10):
        """
        Print concordances in a readable format
        
        Args:
            concordances: Concordance data (Dict with 'Lines' key or List of lines)
            limit: Maximum number of concordances to print
        """
        # Handle both dict and list input
        if isinstance(concordances, dict) and 'Lines' in concordances:
            lines = concordances['Lines'][:limit]
        elif isinstance(concordances, list):
            lines = concordances[:limit]
        else:
            print("No concordances found or unexpected format")
            return
            
        print(f"\nShowing first {len(lines)} concordances (3+3 sentence context):")
        print("=" * 100)
        
        for i, line in enumerate(lines, 1):
            # Parem kontekst + sõnaga lause + parem kontekst
            left = line.get('Left', [])
            kwic = line.get('Kwic', [])
            right = line.get('Right', [])
            
            # Listid stringideks
            left_text = ' '.join([token.get('str', '') for token in left])
            kwic_text = ' '.join([token.get('str', '') for token in kwic])
            right_text = ' '.join([token.get('str', '') for token in right])
            
            print(f"\n[{i:3d}]")
            print(f"Left:  {left_text}")
            print(f"KWIC:  **{kwic_text}**")
            print(f"Right: {right_text}")
            print("-" * 80)

    def process_single_word(self, word: str, corpus: str) -> Optional[List[Dict]]:
        """
        Process a single word and return all concordances using ui_simple method
        
        Args:
            word: The word to search for
            corpus: Corpus name
            
        Returns:
            List of all concordances for the word, or None if failed
        """
        print(f"\n{'='*70}")
        print(f"🔍 Processing word: '{word}'")
        print(f"{'='*70}")
        
        # Kasuta ui_simple meetodit
        print(f"Using UI simple method for '{word}'...")
        
        # Mitu vastet
        result = self.get_concordances_ui_simple(
            corpus=corpus,
            query=word,
            pagesize=1000,
            fromp=1,
            leftctx="3",
            rightctx="3"
        )
            
        if not result:
            print(f"❌ No response for '{word}'")
            return None
            
        total_hits = result.get('fullsize', 0)
        
        if total_hits == 0:
            print(f"❌ No results found for '{word}'")
            return None
        
        print(f"Found {total_hits:,} total hits for '{word}'")
        
        # Kõik leheküljed
        if total_hits > 1000:
            print(f"🔄 Retrieving ALL {total_hits:,} concordances...")
            all_concordances = self._get_all_pages_ui_simple(
                corpus, word, "3", "3"
            )
        else:
            # Single page is enough
            all_concordances = result.get('Lines', [])
            print(f"Single page sufficient: {len(all_concordances)} concordances")
        
        print(f"✅ Retrieved {len(all_concordances)} concordances for '{word}'")
        return all_concordances

def read_words_from_file(filename: str = "SkE_sisend.txt") -> List[str]:
    """
    Read words from input file, one word per line
    
    Args:
        filename: Input file name
        
    Returns:
        List of words to search for
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
        
        print(f"📖 Read {len(words)} words from {filename}:")
        for i, word in enumerate(words, 1):
            print(f"  {i:2d}. {word}")
        
        return words
    
    except FileNotFoundError:
        print(f"❌ Error: File '{filename}' not found!")
        print(f"Please create '{filename}' with one word per line.")
        return []
    except Exception as e:
        print(f"❌ Error reading file '{filename}': {e}")
        return []

def main():
    # Konfiguratsioon, kasutada tuleb enda kasutajanime ja API võtit
    USERNAME = ""
    API_KEY = ""
    CORPUS = "preloaded/estonian_nc23"
    
    # API klient
    api = SketchEngineAPI(USERNAME, API_KEY)
    
    # Loe sõnad sisendfailist
    words = read_words_from_file("SkE_sisend.txt")
    
    if not words:
        print("No words to process. Exiting.")
        return
    
    print(f"\n🚀 Starting to process {len(words)} words from SkE_sisend.txt")
    print(f"Corpus: {CORPUS}")
    print(f"Context: 3 sentences before + 3 sentences after")
    
    # Töötle iga sõna
    all_results = {}
    successful_words = []
    failed_words = []
    
    for i, word in enumerate(words, 1):
        print(f"\n🔄 [{i}/{len(words)}] Processing '{word}'...")
        
        try:
            concordances = api.process_single_word(word, CORPUS)
            
            if concordances and len(concordances) > 0:
                all_results[word] = concordances
                successful_words.append(word)
                
                # Eraldi failid igale sõnale
                safe_word = word.replace('/', '_').replace('\\', '_')  # Safe filename
                
                # Salvesta JSON
                api.save_concordances(
                    concordances, 
                    f"{safe_word}_concordances.json"
                )
                
                # Salvesta CSV
                api.save_concordances_csv(
                    concordances, 
                    f"{safe_word}_concordances.csv"
                )
                
                # Salvesta täiskontekstid
                api.save_full_context_only(
                    concordances, 
                    f"{safe_word}_full_context_only.txt"
                )
                
                # Prindi mõned näited
                print(f"\n📄 Sample concordances for '{word}':")
                api.print_concordances(concordances, limit=3)
                
            else:
                failed_words.append(word)
                print(f"❌ No concordances found for '{word}'")
                
        except Exception as e:
            failed_words.append(word)
            print(f"❌ Error processing '{word}': {e}")
    
    # Kokkuvõte
    print(f"\n{'='*70}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total words processed: {len(words)}")
    print(f"Successful: {len(successful_words)}")
    print(f"Failed: {len(failed_words)}")
    
    if successful_words:
        print(f"\n✅ Successfully processed words:")
        for word in successful_words:
            count = len(all_results[word])
            print(f"  - {word}: {count:,} concordances")
    
    if failed_words:
        print(f"\n❌ Failed words:")
        for word in failed_words:
            print(f"  - {word}")
    
    # Salvesta kokkuvõte
    summary = {
        'total_words': len(words),
        'successful_words': len(successful_words),
        'failed_words': len(failed_words),
        'successful_list': successful_words,
        'failed_list': failed_words,
        'results_summary': {word: len(concordances) for word, concordances in all_results.items()}
    }
    
    with open('processing_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 Files created:")
    print(f"  - processing_summary.json (overall summary)")
    for word in successful_words:
        safe_word = word.replace('/', '_').replace('\\', '_')
        print(f"  - {safe_word}_concordances.json")
        print(f"  - {safe_word}_concordances.csv") 
        print(f"  - {safe_word}_full_context_only.txt")

if __name__ == "__main__":
    main()
# Authors: Madis Jürviste, Eleri Aedmaa, Claude 3.5 Sonnet, GPT-4o

import os
import requests
import csv
import json
import base64
import logging
import re
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dictionary_parser.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DictionaryParser:
    def __init__(self, api_key: str, model: str = "claude-3-7-sonnet-20250219", enable_thinking: bool = False):
        self.api_key = api_key
        self.model = model
        self.default_max_tokens = 64000
        self.enable_thinking = enable_thinking
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "accept": "application/json"
        }
        
        # Add output directory for JSON files
        self.output_dir = "output_json"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")

        # Add debug logging for the system prompt
        logger.debug("Initializing DictionaryParser with system prompt:")
        
        # Updated system prompt with escaped braces for the example JSON
        self.system_prompt = """You are an expert in historical linguistics and lexicography analyzing a 1732 Estonian-German dictionary by Anton Thor Helle. Your task is to extract and structure ALL visible text entries from the provided page, ensuring no text is skipped. Be precise and follow these guidelines:

### **General Instructions:**
1. Process the page COLUMN BY COLUMN:
   - First extract all entries from the LEFT column (top to bottom)
   - Then extract all entries from the RIGHT column (top to bottom)
   - This is crucial for maintaining correct alphabetical order

2. Process ALL text in each column, including:
   - Partially visible or incomplete entries at page boundaries
   - Entries spanning multiple lines
   - Marginal notes, footnotes, or annotations

3. Preserve:
   - Original spellings, punctuation, and diacritical marks
   - Historical character usage (for example, "ö" instead of modern "õ")
   - All special characters, brackets, or nested parentheses
   - The exact order of entries as they appear in each column
   
4. Pay special attention to:
   - Cross-references: Handle entries using "siehe" (e.g., "anni, siehe hanni") carefully and retain their exact structure. These references must be placed in the `german_equivalent` field.

5. Grammatical information placement:
   - Place ALL declension numbers (e.g., "4,4", "7,1", "14,11") in the `estonian_declension` field
   - Place ALL grammatical markers like "Acc. o", "g. a", "ac. it" in the `estonian_declension` field
   - Do NOT scatter this information across other fields

### **Output Format:**
Structure each dictionary entry using these fields:
{{
    "estonian_headword": "main Estonian term",
    "estonian_synonyms": "variants if any",
    "german_equivalent": "primary German translation",
    "german_synonyms": "additional meanings",
    "latin_explanation": "Latin terms if any",
    "part_of_speech": "grammatical markers (e.g., 'adv.')",
    "estonian_declension": "declension/conjugation numbers and ALL grammatical info (e.g., '14,11', 'Acc. o')",
    "estonian_mwu": "usage examples in Estonian",
    "translated_mwu": "translations of examples in German"
}}

### **Additional Notes:**
1. Maintain EXACT order within each column: Process left column first (top to bottom), then right column (top to bottom)
2. Multi-line entries: Combine lines into a single structured entry, ensuring no part is omitted
3. Example handling: If multiple examples exist, include the first in the main entry
4. Do NOT remove any entries that appear to be duplicates - include ALL entries exactly as they appear in the source
5. Do NOT modernize historical characters (keep "ö" as is, do not convert to "õ")

Do not include any explanatory text, analysis, or comments - output ONLY the JSON object."""

        logger.debug(f"System prompt initialized: {self.system_prompt}")
    
    def create_user_prompt(self, page_number: int) -> str:
        """Create a user prompt with clear instructions for column-based processing."""
        return f"""You are processing page {page_number} of the dictionary. Extract ALL entries exactly as they appear, maintaining their precise order within each column.
"""

    def test_api_connection(self):
        """Test connection to the Claude API"""
        try:
            logger.info("Testing API connection...")
            test_data = {
                "model": self.model,
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": "Say 'API connection successful'"
                    }
                ]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=self.headers,
                json=test_data
            )
            
            logger.info(f"API test response status: {response.status_code}")
            if response.status_code == 200:
                logger.info("API connection test successful")
                response_data = response.json()
                response_text = response_data.get('content', [{}])[0].get('text', '')
                logger.info(f"API response: {response_text}")
                return True
            else:
                logger.error(f"API connection test failed: {response.status_code}")
                logger.error(f"API error details: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"API connection test exception: {str(e)}")
            import traceback
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            return False

    def parse_json_from_text_block(self, text_block):
        """
        Parse JSON from a text block, handling both code block and plain JSON formats.
        
        Args:
            text_block: The text content that might contain JSON, or a list of message blocks
            
        Returns:
            Parsed JSON object or None if parsing fails
        """
        if text_block is None:
            return None
            
        # Handle message blocks from API response
        if isinstance(text_block, list):
            # Find the text block containing the JSON response
            for block in text_block:
                if hasattr(block, 'type') and block.type == 'text':
                    text_content = block.text
                    break
            else:
                logging.error("No text block found in API response")
                return None
        else:
            # Handle direct text content
            text_content = text_block.text if hasattr(text_block, 'text') else text_block
            
        # Try to extract JSON from markdown code block first
        code_block_pattern = r'```(?:json)?\n(.*?)```'
        match = re.search(code_block_pattern, text_content, re.DOTALL)
        
        if match:
            # JSON was in a code block
            json_str = match.group(1)
        else:
            # Assume the text might be plain JSON
            json_str = text_content
        
        # Try to parse the JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON: {e}")
            return None

    def process_page(self, image_path: str, page_number: int = 1, max_tokens: int = None) -> List[Dict]:
        """Process a complete page at once with column-based approach."""
        try:
            logger.info(f"Opening image file: {image_path}")
            
            # Add debug logging for system prompt formatting
            logger.debug("Formatting system prompt with page number:")
            formatted_prompt = self.system_prompt.format(page_number=page_number)
            logger.debug(f"First 500 chars of formatted prompt: {formatted_prompt[:500]}")
            
            with open(image_path, 'rb') as file:
                image_content = file.read()
            logger.info(f"Successfully read image file, size: {len(image_content)} bytes")
            
            base64_content = base64.b64encode(image_content).decode('utf-8')
            logger.info(f"Successfully encoded image to base64, first 50 chars: {base64_content[:50]}...")

            # Calculate token budgets
            max_tokens_value = min(max_tokens or self.default_max_tokens, 20000)
            
            data = {
                "model": self.model,
                "max_tokens": max_tokens_value,
                "system": formatted_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_content
                                }
                            },
                            {
                                "type": "text",
                                "text": self.create_user_prompt(page_number)
                            }
                        ]
                    }
                ]
            }

            # Only add thinking if enabled and we have enough tokens
            if self.enable_thinking:
                thinking_budget = min(max_tokens_value // 2, 16000)  # Use at most half of max_tokens for thinking
                if thinking_budget > 0 and max_tokens_value > thinking_budget:
                    data["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": thinking_budget
                    }
                    logger.debug(f"Enabled thinking with budget: {thinking_budget} tokens")
                else:
                    logger.debug("Thinking disabled due to token constraints")
            else:
                logger.debug("Thinking mode is disabled")

            logger.info(f"Sending API request for page {page_number}")
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=self.headers,
                json=data
            )

            logger.info(f"API response status: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"API error: {response.text}")
                self._save_json_output([], page_number, error=response.text)
                return []

            response_data = response.json()
            logger.info(f"API JSON response: {response_data}")
            # Print raw response content
            # Enter interactive mode to inspect response
            #import code
            #code.interact(local=locals())

            raw_content = response_data.get('content', [{}])[0].get('text', '')
            logger.info(f"Raw content from page {page_number} (first 500 chars): {raw_content[:500]}...")
            print(f"Raw content from page {page_number} (first 500 chars):\n{raw_content[:500]}...\n")
            
            # Add detailed debugging for response structure
            logger.debug("Response data structure:")
            logger.debug(f"Keys in response_data: {list(response_data.keys())}")
            logger.debug(f"Content structure: {response_data.get('content')}")
            # Extract thinking content if available
            thinking_content = None
            if 'thinking' in response_data['content'][0]:
                thinking_content = response_data['content'][0].get('thinking', '')
                logger.info(f"Thinking content from page {page_number} (first 500 chars): {thinking_content[:500]}...")
                print(f"Thinking content used to improve extraction\n")
            

            logger.info("Attempting to parse entries from response...")
            entries = self.parse_response(response_data['content'][1]['text'])
            
            if not entries:
                logger.error("No entries were parsed from the response")
                logger.debug("Full raw content for debugging:")
                logger.debug(raw_content)
                self._save_json_output([], page_number, error="No entries parsed", raw_content=raw_content)
                return []
            
            # Post-process entries to fix common issues
            processed_entries = self.post_process_entries(entries)
            
            # Save the results to JSON
            self._save_json_output(processed_entries, page_number, thinking_content=thinking_content)
            
            return processed_entries

            # If we get here, 'content' wasn't in the response
            logger.error(f"No content in API response")
            self._save_json_output([], page_number, error="No content in API response")
            return []

        except Exception as e:
            logger.error(f"Error processing page: {str(e)}")
            logger.error(f"Full exception details:", exc_info=True)
            self._save_json_output([], page_number, error=str(e))
            return []

    def _save_json_output(self, entries: List[Dict], page_number: int, error: str = None, 
                         thinking_content: str = None, raw_content: str = None) -> None:
        """Save the results to JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare the output data
        output_data = {
            "page_number": page_number,
            "timestamp": timestamp,
            "entries": entries,
            "metadata": {
                "num_entries": len(entries),
                "model": self.model,
                "thinking_enabled": thinking_content is not None
            }
        }
        
        if error:
            output_data["error"] = error
        if thinking_content:
            output_data["thinking_content"] = thinking_content
        if raw_content:
            output_data["raw_content"] = raw_content

        # Save individual page result
        page_filename = f"page_{page_number:03d}_{timestamp}.json"
        page_path = os.path.join(self.output_dir, page_filename)
        
        try:
            with open(page_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved page {page_number} results to {page_path}")
            
            # Update the running collection file
            collection_path = os.path.join(self.output_dir, "all_pages.json")
            all_pages = []
            
            if os.path.exists(collection_path):
                try:
                    with open(collection_path, 'r', encoding='utf-8') as f:
                        all_pages = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Could not read existing all_pages.json, starting fresh")
                    all_pages = []
            
            # Update or append the current page's data
            page_index = next((i for i, p in enumerate(all_pages) 
                             if p["page_number"] == page_number), None)
            if page_index is not None:
                all_pages[page_index] = output_data
            else:
                all_pages.append(output_data)
            
            # Save the updated collection
            with open(collection_path, 'w', encoding='utf-8') as f:
                json.dump(all_pages, f, ensure_ascii=False, indent=2)
            logger.info(f"Updated all_pages.json with page {page_number} results")
            
        except Exception as e:
            logger.error(f"Error saving JSON output: {str(e)}")
            logger.error("Full exception details:", exc_info=True)

    def parse_response(self, response_text: str) -> List[Dict]:
        """Enhanced response parsing with better handling."""
        entries = []
        try:
            # Debug the input
            logger.debug("Starting parse_response with text:")
            logger.debug(f"First 1000 chars of response_text: {response_text[:1000]}")
            
            # Use regex to find all JSON objects in the response
            json_pattern = r'\{[^{}]*\}'
            potential_entries = list(re.finditer(json_pattern, response_text))
            
            logger.debug(f"Found {len(potential_entries)} potential JSON objects")
            
            for i, entry_match in enumerate(potential_entries):
                try:
                    entry_text = entry_match.group()
                    logger.debug(f"Processing potential entry {i+1}:")
                    logger.debug(f"Entry text: {entry_text}")
                    
                    entry = json.loads(entry_text)
                    logger.debug(f"Successfully parsed entry {i+1}: {entry}")
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error in entry {i+1}: {e}")
                    logger.warning(f"Problematic text: {entry_text}")
                    # Try to clean the text and parse again
                    try:
                        cleaned_text = entry_text.strip().replace('\n', '').replace('\r', '')
                        entry = json.loads(cleaned_text)
                        logger.info(f"Successfully parsed entry after cleaning: {entry}")
                        entries.append(entry)
                    except json.JSONDecodeError as e2:
                        logger.warning(f"Failed to parse even after cleaning: {e2}")
                    continue
                except Exception as e:
                    logger.warning(f"Error parsing entry {i+1}: {e}")
                    logger.warning(f"Entry text that caused error: {entry_text}")
                    continue

            logger.info(f"Successfully parsed {len(entries)} entries")
            return entries
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            logger.error("Full exception details:", exc_info=True)
            return []
            
    def post_process_entries(self, entries: List[Dict]) -> List[Dict]:
        """Post-process entries to fix common issues."""
        processed_entries = []
        
        for entry in entries:
            # Create a copy to avoid modifying the original
            processed_entry = entry.copy()
            
            # Fix grammatical information and declension numbers
            self._move_grammatical_info_to_declension(processed_entry)
            
            # Clean up empty fields
            for key in processed_entry:
                if processed_entry[key] is None:
                    processed_entry[key] = ''
                elif isinstance(processed_entry[key], str):
                    processed_entry[key] = processed_entry[key].strip()
            
            # Add the processed entry
            processed_entries.append(processed_entry)
            
        return processed_entries
    
    def _move_grammatical_info_to_declension(self, entry: Dict) -> None:
        """Move grammatical information to the estonian_declension field."""
        # Patterns for grammatical information
        grammar_patterns = [
            r'Acc\.\s*[a-z]',           # e.g., "Acc. o"
            r'acc\.\s*[a-z]',           # e.g., "acc. o"
            r'g\.\s*[a-z]',             # e.g., "g. a"
            r'ac\.\s*[a-z]',            # e.g., "ac. it"
            r'g\.\s*[a-z],\s*acc\.\s*[a-z]', # e.g., "g. a, acc. to"
            r'\b\d+,\d+\b',             # e.g., "14,11", "7,1"
        ]
        
        # Fields to check for grammatical information
        fields_to_check = [
            'german_equivalent', 
            'german_synonyms', 
            'translated_mwu',
            'estonian_synonyms',
            'latin_explanation',
            'estonian_mwu',
            'estonian_headword'
        ]
        
        for field in fields_to_check:
            if field in entry and entry[field]:
                value = entry[field]
                declension_info = []
                
                # Extract grammatical information
                for pattern in grammar_patterns:
                    matches = re.findall(pattern, value)
                    if matches:
                        for match in matches:
                            declension_info.append(match)
                            # Remove from the original field
                            value = re.sub(r'\b' + re.escape(match) + r'\b', '', value)
                
                # Clean up the text after removing grammatical info
                value = re.sub(r'\s+', ' ', value).strip()
                value = re.sub(r'\s*\.\s*$', '.', value)  # Fix trailing periods
                value = re.sub(r'\s*,\s*$', ',', value)   # Fix trailing commas
                
                # Update the field and declension information
                if declension_info:
                    entry[field] = value
                    if 'estonian_declension' in entry and entry['estonian_declension']:
                        entry['estonian_declension'] += ' ' + ' '.join(declension_info)
                    else:
                        entry['estonian_declension'] = ' '.join(declension_info)

def validate_entries(entries: List[Dict]) -> List[Dict]:
    """
    Validate that entries follow the rule that headwords should not contain spaces.
    Any entry with spaces in estonian_headword should be an MWU instead.
    """
    invalid_entries = []
    
    for i, entry in enumerate(entries):
        # Check if estonian_headword contains spaces (indicating it should be an MWU)
        if 'estonian_headword' in entry and entry['estonian_headword'] and ' ' in entry['estonian_headword'].strip():
            logger.warning(f"Entry {i} has spaces in estonian_headword, should be MWU: {entry['estonian_headword']}")
            invalid_entries.append((i, entry['estonian_headword']))
    
    if invalid_entries:
        logger.warning(f"Found {len(invalid_entries)} entries with spaces in estonian_headword that should be MWUs")
        # Log the first 5 examples
        for i, (idx, headword) in enumerate(invalid_entries[:5]):
            logger.warning(f"Example {i+1}: Index {idx}, Headword: '{headword}'")
    else:
        logger.info("Validation passed: No entries with spaces in estonian_headword found")
    
    return entries

def process_directory(input_dir: str, output_path: str, api_key: str, max_tokens: int = 64000):
    """Process directory with improved processing flow."""
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # Check if input directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        logger.info("Please create the directory and add PNG files to process")
        return

    parser = DictionaryParser(api_key, enable_thinking=True)
    
    # Test API connection first
    if not parser.test_api_connection():
        logger.error("API connection test failed. Please check your API key and internet connection.")
        return
        
    all_entries = []

    try:
        png_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])

        if not png_files:
            logger.warning(f"No PNG files found in {input_dir}")
            return
            
        # Log the files that will be processed
        logger.info(f"Found {len(png_files)} PNG files to process:")
        for i, png_file in enumerate(png_files, 1):
            logger.info(f"  {i}. {png_file}")

        for i, png_file in enumerate(png_files, 1):
            logger.info(f"Processing file {i}/{len(png_files)}: {png_file}")
            image_path = os.path.join(input_dir, png_file)
            
            # Check that the file exists and is readable
            if not os.path.exists(image_path):
                logger.error(f"File does not exist: {image_path}")
                continue
                
            # Check file size
            file_size = os.path.getsize(image_path)
            logger.info(f"File size: {file_size} bytes")
            if file_size == 0:
                logger.error(f"File is empty: {image_path}")
                continue
                
            entries = parser.process_page(image_path, i, max_tokens)
            if entries:
                all_entries.extend(entries)
                logger.info(f"Successfully processed {len(entries)} entries from {png_file}")
            else:
                logger.warning(f"No entries extracted from {png_file}")

        # Use entries directly as they come from the parser
        processed_entries = all_entries
        
        # Validate entries
        validated_entries = validate_entries(processed_entries)

        fieldnames = [
            'estonian_headword', 'estonian_synonyms', 'german_equivalent',
            'german_synonyms', 'latin_explanation', 'part_of_speech',
            'estonian_declension', 'estonian_mwu', 'translated_mwu'
        ]

        # Save to CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(validated_entries)

        logger.info(f"Results saved to {output_path}")
        logger.info(f"Total entries processed: {len(validated_entries)}")
        
        # Generate a Markdown table for visualization
        generate_markdown_table(validated_entries, output_path.replace('.csv', '.md'))

    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}")


def generate_markdown_table(entries: List[Dict], output_path: str):
    """Generate a Markdown table from the entries."""
    if not entries:
        logger.warning("No entries to generate markdown table")
        return
    
    fieldnames = [
        'estonian_headword', 'estonian_synonyms', 'german_equivalent',
        'german_synonyms', 'latin_explanation', 'part_of_speech',
        'estonian_declension', 'estonian_mwu', 'translated_mwu'
    ]
    
    try:
        with open(output_path, 'w', encoding='utf-8') as mdfile:
            # Write header
            mdfile.write('| ' + ' | '.join(fieldnames) + ' |\n')
            mdfile.write('| ' + ' | '.join(['---'] * len(fieldnames)) + ' |\n')
            
            # Write rows
            for entry in entries:
                row = []
                for field in fieldnames:
                    cell_content = entry.get(field, '')
                    # Escape pipe characters in Markdown tables
                    cell_content = str(cell_content).replace('|', '\\|')
                    row.append(cell_content)
                mdfile.write('| ' + ' | '.join(row) + ' |\n')
                
        logger.info(f"Markdown table saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating markdown table: {str(e)}")


if __name__ == "__main__":
    # Configuration
    api_key = ""
    
    # Get the current working directory
    current_dir = os.getcwd()
    logger.info(f"Current working directory: {current_dir}")
    
    # Define paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "Helle-12-PNG")
    output_path = os.path.join(script_dir, "Helle-12-mwu-c37S-1.csv")
    
    logger.info(f"Input directory path: {input_dir}")
    logger.info(f"Output file path: {output_path}")
    
    # Create input directory if it doesn't exist
    if not os.path.exists(input_dir):
        logger.info(f"Creating input directory: {input_dir}")
        os.makedirs(input_dir)
        logger.info(f"Please place your PNG files in: {input_dir}")
    
    max_tokens = 64000

    process_directory(input_dir, output_path, api_key, max_tokens)

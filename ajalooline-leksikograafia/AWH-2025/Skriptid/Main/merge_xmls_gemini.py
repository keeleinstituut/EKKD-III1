import os
import sys
import re
import asyncio
from collections import defaultdict
import pathlib
import litellm
import csv
import datetime

# --- Configuration ---
# Ensure your Gemini API key is set in your environment variables
# os.environ["GEMINI_API_KEY"] = "your_key_here"

MERGE_MODEL = "gemini/gemini-2.5-pro"
PROMPT_TEMPLATE_PATH = pathlib.Path(__file__).parent / "ai_merge_prompt.md"

# --- LiteLLM Setup ---
litellm.success_callback = []
litellm.failure_callback = []
litellm.callbacks = []
litellm.drop_params = True

def clean_response(text: str) -> str:
    """Cleans the XML response from the model by removing markdown fences."""
    text = text.strip()
    if text.startswith("```xml"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

async def merge_chunks_with_ai(group_key, chunk_files, output_path, prompt_template):
    """
    Merges a list of XML chunk files into a single XML file using an AI model.
    """
    # 1. Sort chunks into the correct reading order
    col1_chunks = sorted([f for f in chunk_files if '_column1_' in f], key=lambda f: int(re.search(r'_chunk(\d+)', f).group(1)))
    col2_chunks = sorted([f for f in chunk_files if '_column2_' in f], key=lambda f: int(re.search(r'_chunk(\d+)', f).group(1)))
    sorted_chunks = col1_chunks + col2_chunks

    # 2. Read and format chunk content for the prompt
    chunk_content_str = ""
    for i, chunk_file in enumerate(sorted_chunks):
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                content = f.read()
                chunk_content_str += f"--- CHUNK {i+1} ---\n{content}\n\n"
        except IOError as e:
            print(f"✗ Warning: Could not read chunk {os.path.basename(chunk_file)}: {e}", file=sys.stderr)

    if not chunk_content_str:
        print(f"- Skipping {os.path.basename(output_path)}: no chunk content found.")
        return None

    # 3. Construct the full prompt
    full_prompt = prompt_template.replace('{chunks}', chunk_content_str)

    # 4. Call the AI model
    print(f"▶️  Sending {len(sorted_chunks)} chunks to {MERGE_MODEL} for {os.path.basename(output_path)}...")
    try:
        response = await litellm.acompletion(
            model=MERGE_MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.0, # We want deterministic, high-quality output
        )
        
        merged_xml = response.choices[0].message.content
        if not merged_xml:
            print(f"✗ Error: Received an empty response from the model for {os.path.basename(output_path)}.", file=sys.stderr)
            return None

        cleaned_xml = clean_response(merged_xml)

        # 5. Save the result
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_xml)
        print(f"✓ AI-merged file saved to {os.path.basename(output_path)}")

        cost = litellm.completion_cost(completion_response=response)
        usage = response.usage
        metadata = {
            "merged_xml_file": os.path.basename(output_path),
            "group_key": group_key,
            "model": MERGE_MODEL,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "cost": cost,
        }
        return metadata

    except Exception as e:
        print(f"✗ Error calling LiteLLM for {os.path.basename(output_path)}: {e}", file=sys.stderr)
        return None


async def main():
    if len(sys.argv) != 3:
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
    except IOError as e:
        print(f"FATAL: Could not read prompt template at {PROMPT_TEMPLATE_PATH}: {e}", file=sys.stderr)
        sys.exit(1)

    # Group files based on page, prompt, variant, and run number
    merge_groups = defaultdict(list)
    filename_pattern = re.compile(r'(page\d+)_column\d+_chunk\d+_(.+)\.xml')

    for filename in os.listdir(input_dir):
        if filename.endswith(".xml"):
            match = filename_pattern.match(filename)
            if match:
                page_stem = match.group(1)
                variant_info = match.group(2)
                group_key = f"{page_stem}_{variant_info}"
                merge_groups[group_key].append(os.path.join(input_dir, filename))

    if not merge_groups:
        print(f"No XML chunk files found in '{input_dir}' matching the expected pattern.")
        return

    tasks = []
    for group_key, files in merge_groups.items():
        output_filename = f"{group_key}_merged_ai.xml"
        output_path = os.path.join(output_dir, output_filename)
        tasks.append(merge_chunks_with_ai(group_key, files, output_path, prompt_template))
    
    results = await asyncio.gather(*tasks)

    all_metadata = []
    for result in results:
        if isinstance(result, dict):
            all_metadata.append(result)

    if all_metadata:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        csv_filename = os.path.join(output_dir, f"ai_merge_log_{timestamp}.csv")
        
        fieldnames = [
            "merged_xml_file", "group_key", "model", "input_tokens",
            "output_tokens", "cost"
        ]
        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metadata)
        print(f"Saved merge metadata to {csv_filename}")

if __name__ == "__main__":
    if "GEMINI_API_KEY" not in os.environ:
        print("FATAL: Please set the GEMINI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)
    asyncio.run(main())

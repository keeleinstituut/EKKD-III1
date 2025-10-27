# pip install "litellm[vision]"            # vision extras pull in httpx & Pillow
import os, pathlib, json, base64, sys, asyncio, re, datetime
import xml.etree.ElementTree as ET
from collections import defaultdict
from itertools import product
from litellm import acompletion
from litellm.utils import get_model_info
from litellm.constants import DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET
import litellm

# Disable litellm's default logging callbacks
litellm.success_callback = []
litellm.failure_callback = []
litellm.callbacks = []

# Drop unsupported parameters instead of erroring
litellm.drop_params = True

def merge_column_chunks(chunk_files: list) -> str:
    """Merges content of several XML chunk files into a single string."""
    merged_content = ""
    for chunk_file in sorted(chunk_files): # sort to maintain order
        try:
            # Read the content of each chunk file
            with open(chunk_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # The model is supposed to return entry tags, but sometimes wraps them
                # in a root element like <entries> or <root>. We need to handle this.
                try:
                    # Try to parse it as a full XML document
                    root = ET.fromstring(content)
                    # If it's a wrapper, take the children's content
                    if root.tag in ['entries', 'root', 'div']:
                         merged_content += ''.join(ET.tostring(child, encoding='unicode') for child in root)
                    else:
                        # Otherwise, it's a single entry, so use it as is
                        merged_content += ET.tostring(root, encoding='unicode')
                except ET.ParseError:
                    # If parsing fails, it's likely already a fragment, so just append
                    merged_content += content + "\n"
        except FileNotFoundError:
            print(f"Warning: XML chunk file not found: {chunk_file}")
            continue
    return merged_content

def merge_xmls(col1_content: str, col2_content: str, merged_path: pathlib.Path, page_num: str):
    """Merges two XML fragments from strings into a single TEI page."""
    try:
        # Create the root structure for the merged file
        page_div = ET.Element('div', attrib={'type': 'page', 'n': page_num, 'corresp': f'#p{page_num}'})
        
        # Helper to append content to the page_div
        def append_content(content):
            if content:
                # Wrap content to make it a valid XML for parsing
                try:
                    root = ET.fromstring(f"<root>{content}</root>")
                    for child in root:
                        page_div.append(child)
                except ET.ParseError as e:
                    print(f"✗ Warning: Could not parse content for page {page_num}. Content might be malformed. Error: {e}")


        # Process and append content from both columns
        append_content(col1_content)
        append_content(col2_content)

        # Write the merged result
        tree = ET.ElementTree(page_div)
        ET.indent(tree, space="  ")
        tree.write(merged_path, encoding="utf-8", xml_declaration=False)
        return True
    except Exception as e:
        print(f"✗ Error merging XMLs for page {page_num}: {e}")
        return False
        
# ---- 1. config ----------------------------------------------------------------
DRY_RUN = False # If True, will print job params and skip API calls.
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."     # Claude-3 Opus Vision
os.environ["GEMINI_API_KEY"]    = "..."      # Gemini 2.5-Pro

# prompt variants from /prompts, storing filename stem for output
PROMPTS = [(p.stem, p.read_text(encoding="utf-8")) for p in pathlib.Path("prompts").glob("*.md")]

# ⇣⇣ Model variants + thinking budgets (labelled) ⇣⇣
MODELS = [
    # ---------- Gemini ----------
    # think_budget: 0=disable, -1=enable with default budget, >0=enable with budget
    #{"name": "gemini/gemini-2.5-pro",   "label": "g25pro_reasoning",   "provider": "gemini",    "think_budget": -1},
    #{"name": "gemini/gemini-2.5-flash", "label": "g25flash_no_thinking", "provider": "gemini",    "think_budget": 0},
    #{"name": "gemini/gemini-2.5-flash", "label": "g25flash_reasoning", "provider": "gemini",    "think_budget": -1},
    #{"name": "gemini/gemini-2.5-flash-lite-preview-06-17", "label": "g25flashlite", "provider": "gemini",    "think_budget": 0},

    # ---------- Anthropic ----------
    # think_budget: None=disable, >0=enable with budget
    #{"name": "anthropic/claude-opus-4-20250514",    "label": "opus4_no_thinking",    "provider": "anthropic", "think_budget": None},
    #{"name": "anthropic/claude-opus-4-20250514",    "label": "opus4_reasoning",    "provider": "anthropic", "think_budget": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET},
    #{"name": "anthropic/claude-sonnet-4-20250514",  "label": "sonnet4_no_thinking",  "provider": "anthropic", "think_budget": None},
    #{"name": "anthropic/claude-sonnet-4-20250514",  "label": "sonnet4_reasoning",  "provider": "anthropic", "think_budget": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET},
    {"name": "anthropic/claude-3-7-sonnet-20250219","label": "sonnet37_no_thinking", "provider": "anthropic", "think_budget": None},
    #{"name": "anthropic/claude-3-7-sonnet-20250219","label": "sonnet37_reasoning", "provider": "anthropic", "think_budget": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET},
    #{"name": "anthropic/claude-3-5-sonnet-20241022","label": "sonnet35_no_thinking", "provider": "anthropic", "think_budget": None},
]

for model_info in MODELS:
    try:
        # get_model_info() returns a dict. We access 'max_output_tokens' to get the max OUTPUT tokens for a given model
        model_info_obj = get_model_info(model_info["name"])
        max_tokens = model_info_obj.get("max_output_tokens")
        
        # Manual override for claude-3-7-sonnet, which seems to have an incorrect value in litellm
        if model_info["name"] == "anthropic/claude-3-7-sonnet-20250219":
            max_tokens = 64000
            
        if max_tokens:
            model_info["max_tokens"] = max_tokens
    except Exception as e:
        print(f"Warning: Error retrieving max_tokens for {model_info['name']}: {e}. Using default.")

REPEATS = 1
# SUCCESS_COUNT will be managed inside main
import argparse

# ---- 2. helper ----------------------------------------------------------------
def build_messages(prompt: str, img_path: pathlib.Path, model: str):
    """Return messages with image block formatted for the specific model."""
    # For Anthropic/Gemini/OpenAI models, encode image as base64 and use data URI
    if "anthropic" in model.lower() or "gemini" in model.lower() or "openai" in model.lower():
        with open(img_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        ext = img_path.suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            mime_type = 'image/jpeg'
        elif ext == '.png':
            mime_type = 'image/png'
        elif ext == '.gif':
            mime_type = 'image/gif'
        elif ext == '.webp':
            mime_type = 'image/webp'
        else:
            mime_type = 'image/jpeg'  # default fallback
        
        # Handle the newer OpenAI o3 models that expect a different image format
        if "openai" in model.lower() and any(x in model.lower() for x in ["o3", "o1"]):
            # For o3 and o1 models, use the direct URL format
            return [
                {"role": "user",
                 "content": [
                     {"type": "text", "text": prompt},
                     {
                         "type": "image_url",
                         "image_url": f"data:{mime_type};base64,{img_data}"
                     }
                 ]}
            ]
        else:
            # For other models, use the nested object format
            return [
                {"role": "user",
                 "content": [
                     {"type": "text", "text": prompt},
                     {
                         "type": "image_url",
                         "image_url": {"url": f"data:{mime_type};base64,{img_data}"}
                     }
                 ]}
            ]
    else:
        # Fallback for any other model, though most vision models prefer base64
        return [
            {"role": "user",
             "content": [
                 {"type": "text", "text": prompt},
                 {
                     "type": "image_url",
                     "image_url": {"url": f"file://{img_path.resolve()}"}
                 }
             ]}
        ]

def clean_response(text: str) -> str:
    """Cleans the XML response from the model."""
    text = text.strip()
    if text.startswith("```xml"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    
    if text.endswith("```"):
        text = text[:-3]
        
    return text.strip()

async def process_job(img_path: pathlib.Path, prompt_info: tuple, model_info: dict, rep: int, output_dir: pathlib.Path) -> dict | bool:
    """Processes a single job by making a concurrent API call."""
    prompt_name, prompt_content = prompt_info
    model_name = model_info['name']
    variant_label = model_info['label']
    
    output_filename = f"{img_path.stem}_{prompt_name}_{variant_label}_{rep + 1}.xml"
    output_path = output_dir / output_filename
    json_output_filename = f"{img_path.stem}_{prompt_name}_{variant_label}_{rep + 1}.json"
    json_output_path = output_dir / json_output_filename
    
    print(f"▶️  Starting job for {model_name} on {img_path.name} with prompt '{prompt_name}' (Repeat {rep+1})")
    
    try:
        messages = build_messages(prompt_content, img_path, model_name)

        kwargs = {
            "model": model_name,
            "messages": messages,
            "max_tokens": model_info.get("max_tokens", 32000)
        }
        tb = model_info.get("think_budget")
        if tb is not None:
            if model_info["provider"] == "anthropic":
                if tb > 0:
                    kwargs["thinking"] = {"type": "enabled", "budget_tokens": tb}
            elif model_info["provider"] == "gemini":
                # For Gemini, tb=-1 means enable with default budget
                if tb == -1:
                    kwargs["thinking"] = {"type": "enabled"}
                elif tb > 0: # This covers setting a budget (>0)
                    kwargs["thinking"] = {"type": "enabled", "budget_tokens": tb}

        if DRY_RUN:
            print(f"DRY RUN for {output_filename}:")
            # a simple json dump for messages will fail on the image data, so we handle it
            temp_kwargs = kwargs.copy()
            if "messages" in temp_kwargs:
                for msg in temp_kwargs["messages"]:
                    if "content" in msg and isinstance(msg["content"], list):
                        for item in msg["content"]:
                            if item.get("type") == "image_url" and "base64" in item.get("image_url", {}).get("url", ""):
                                item["image_url"]["url"] = "[base64 image data truncated]"
            print(json.dumps(temp_kwargs, indent=2))
            return True # Return a non-False value to be counted as a "success"

        max_retries = 5
        base_delay = 10  # seconds
        response = None
        for attempt in range(max_retries):
            try:
                response = await acompletion(**kwargs)
                break  # Success
            except litellm.RateLimitError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limit error for {output_filename}. Retrying in {delay}s... ({attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    print(f"✗ Rate limit error for {output_filename} after {max_retries} retries. Giving up.")
                    raise e  # Re-raise to be caught by the outer block
    
        if response is None:
            return False

        # Save raw JSON response
        with open(json_output_path, "w", encoding="utf-8") as f:
            # The response object is a Pydantic model so we can dump it to json
            f.write(response.model_dump_json(indent=4))
        
        content = response.choices[0].message.content
        if content:
            cleaned_content = clean_response(content)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_content)
            
            usage = response.usage
            cost = litellm.completion_cost(completion_response=response)

            metadata = {
                "output_filename": output_path.name,
                "json_output_filename": json_output_path.name,
                "image_name": img_path.name,
                "original_image": img_path.stem.split('_column')[0] + img_path.suffix,
                "prompt_name": prompt_name,
                "model_name": model_name,
                "variant_label": variant_label,
                "repeat": rep + 1,
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "cost": cost,
            }
            print(f"✓ Saved {output_filename} and {json_output_path.name}")
            return metadata
        else:
            print(f"✗ Empty response for {output_filename}")
            return False
            
    except Exception as e:
        print(f"✗ Error processing job for {output_filename}: {e}")
        return False

# ---- 3. main execution -------------------------------------------------------------
async def main():
    """Main function to run concurrent processing."""
    
    parser = argparse.ArgumentParser(description="Run XML generation prompts on chunked images.")
    parser.add_argument("chunk_dir", help="Directory with chunked image files.")
    parser.add_argument("--output-dir", help="Directory to save the generation results. If not provided, a timestamped directory will be created.")
    parser.add_argument("--concurrency-anthropic", type=int, default=20, help="Number of concurrent API calls for Anthropic.")
    parser.add_argument("--concurrency-gemini", type=int, default=10, help="Number of concurrent API calls for Gemini.")
    parser.add_argument("--concurrency-default", type=int, default=5, help="Number of concurrent API calls for other providers.")
    args = parser.parse_args()

    IMAGES_DIR = pathlib.Path(args.chunk_dir)
    
    if args.output_dir:
        OUTPUT_DIR = pathlib.Path(args.output_dir)
    else:
        OUTPUT_DIR = pathlib.Path(f"output_{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}")

    MERGED_XML_DIR = OUTPUT_DIR / "merged_xmls"

    # Create the output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    MERGED_XML_DIR.mkdir(exist_ok=True)

    print(f"Using output directory: {OUTPUT_DIR.resolve()}")

    IMAGES = list(IMAGES_DIR.glob("*.jpg"))
    if not IMAGES:
        print(f"No .jpg images found in {IMAGES_DIR}")
        return

    jobs_to_run = []
    total_possible_jobs = len(IMAGES) * len(PROMPTS) * len(MODELS) * REPEATS
    
    for img_path, prompt_info, model_info, rep_idx in product(IMAGES, PROMPTS, MODELS, range(REPEATS)):
        prompt_name, _ = prompt_info
        variant_label = model_info['label']
        
        output_filename = f"{img_path.stem}_{prompt_name}_{variant_label}_{rep_idx + 1}.xml"
        output_path = OUTPUT_DIR / output_filename
        
        if not output_path.exists():
            jobs_to_run.append((img_path, prompt_info, model_info, rep_idx))

    total_jobs = len(jobs_to_run)
    completed_jobs = total_possible_jobs - total_jobs
    print(f"Found {total_jobs} jobs to process ({completed_jobs} already completed).")

    semaphores = {
        "anthropic": asyncio.Semaphore(args.concurrency_anthropic),
        "gemini": asyncio.Semaphore(args.concurrency_gemini)
    }
    default_sem = asyncio.Semaphore(args.concurrency_default)

    async def process_job_with_semaphore(provider, *args):
        sem = semaphores.get(provider, default_sem)
        async with sem:
            await asyncio.sleep(0.1) # help spread out requests
            return await process_job(*args)

    tasks = []
    for img_path, prompt_info, model_info, rep in jobs_to_run:
        provider = model_info['provider']
        tasks.append(asyncio.create_task(process_job_with_semaphore(provider, img_path, prompt_info, model_info, rep, OUTPUT_DIR)))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and calculate success count
    success_count = 0
    all_metadata = []
    for result in results:
        if isinstance(result, dict):
            success_count += 1
            all_metadata.append(result)
        elif isinstance(result, Exception):
            # asyncio.gather with return_exceptions=True returns exceptions in the list
            print(f"A job failed with an exception: {result}")

    # ---- Merge XMLs ----
    print("\n--- Merging chunk XMLs into full page XMLs ---")
    
    # Group successful results by original image, prompt, and model variant
    merge_jobs = defaultdict(lambda: {"col1": [], "col2": []})
    for meta in all_metadata:
        original_image_stem = meta["original_image"].split('.')[0]
        key = (original_image_stem, meta["prompt_name"], meta["variant_label"], meta["repeat"])
        
        image_name = meta["image_name"]
        
        chunk_match = re.search(r'column(\d+)_chunk(\d+)', image_name)
        if chunk_match:
            col_num = int(chunk_match.group(1))
            chunk_num = int(chunk_match.group(2))

            chunk_info = {
                "path": OUTPUT_DIR / meta["output_filename"],
                "chunk_num": chunk_num
            }

            if col_num == 1:
                merge_jobs[key]["col1"].append(chunk_info)
            elif col_num == 2:
                merge_jobs[key]["col2"].append(chunk_info)

    for (original_stem, prompt, variant, rep), chunks in merge_jobs.items():
        # Sort chunks by chunk number
        sorted_col1_paths = [c["path"] for c in sorted(chunks["col1"], key=lambda x: x["chunk_num"])]
        sorted_col2_paths = [c["path"] for c in sorted(chunks["col2"], key=lambda x: x["chunk_num"])]

        col1_content = merge_column_chunks(sorted_col1_paths)
        col2_content = merge_column_chunks(sorted_col2_paths)

        if col1_content or col2_content:
            page_num = original_stem.replace('page', '')
            merged_filename = f"{original_stem}_{prompt}_{variant}_{rep}_merged.xml"
            merged_path = MERGED_XML_DIR / merged_filename
            
            if merge_xmls(col1_content, col2_content, merged_path, page_num):
                print(f"✓ Merged chunks for {original_stem} ({prompt}/{variant}/{rep}) into {merged_path.name}")
        else:
            print(f"✗ Skipping merge for {original_stem} ({prompt}/{variant}/{rep}) due to missing content.")

    # ---- 4. summary ---------------------------------------------------------------
    print(f"\nFinished. Saved {success_count} successful calls as XML out of {total_jobs} attempts.")

    if all_metadata:
        import csv
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        csv_filename = OUTPUT_DIR / f"generation_log_{timestamp}.csv"
        
        fieldnames = [
            "output_filename", "json_output_filename", "image_name", "original_image", "prompt_name", "model_name", "variant_label",
            "repeat", "input_tokens", "output_tokens", "cost"
        ]
        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metadata)
        print(f"Saved metadata to {csv_filename}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting gracefully.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)
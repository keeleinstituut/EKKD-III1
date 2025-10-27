import os, pathlib, json, base64, sys, asyncio
from itertools import product
from PIL import Image
from litellm import acompletion
import litellm

# Disable litellm's default logging callbacks
litellm.success_callback = []
litellm.failure_callback = []
litellm.callbacks = []

# Drop unsupported parameters instead of erroring
litellm.drop_params = True

# ---- 1. config ----------------------------------------------------------------
os.environ["OPENAI_API_KEY"]    = "sk-proj-..."         # GPT-4o Vision
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."     # Claude-3 Opus Vision
os.environ["GEMINI_API_KEY"]    = "..."      # Gemini 2.5-Pro

# local image files from /images
IMAGES = list(pathlib.Path("images").glob("*.jpg"))

# prompt variants from /prompts, storing filename stem for output
PROMPTS = [(p.stem, p.read_text(encoding="utf-8")) for p in pathlib.Path("prompts").glob("*.md")]

# five vision-capable models routed through LiteLLM
MODELS = [
    {"name": "anthropic/claude-3-7-sonnet-20250219", "max_tokens": 32000, "thinking": False},
    {"name": "anthropic/claude-sonnet-4-20250514", "max_tokens": 32000, "thinking": False},
    {"name": "anthropic/claude-opus-4-20250514", "max_tokens": 32000, "thinking": False},
    {"name": "gemini/gemini-2.5-pro", "max_tokens": 32000, "thinking": True},  # Using the main Pro model
    {"name": "gemini/gemini-2.5-flash-preview-04-17", "max_tokens": 32000, "thinking": True},  # Flash model
    {"name": "gemini/gemini-2.5-pro-exp-03-25", "max_tokens": 32000},  # Your original experimental model
    {"name": "gemini/gemini-2.5-flash", "max_tokens": 32000, "thinking": True},
]

REPEATS = 5
# SUCCESS_COUNT will be managed inside main
OUTPUT_DIR = pathlib.Path("output")

# Create the output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)


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

async def process_job(img_path: pathlib.Path, prompt_info: tuple, model_info: dict, rep: int) -> bool:
    """Processes a single job by making a concurrent API call."""
    prompt_name, prompt_content = prompt_info
    model_name = model_info['name']
    
    safe_model_name = model_name.replace('/', '_')
    output_filename = f"{img_path.stem}_{prompt_name}_{safe_model_name}_{rep + 1}.xml"
    output_path = OUTPUT_DIR / output_filename
    
    print(f"▶️  Starting job for {model_name} on {img_path.name} with prompt '{prompt_name}' (Repeat {rep+1})")
    
    try:
        messages = build_messages(prompt_content, img_path, model_name)
        
        response = await acompletion(
            model=model_name,
            messages=messages,
            max_tokens=model_info.get('max_tokens', 32000),
        )
        
        content = response.choices[0].message.content
        if content:
            cleaned_content = clean_response(content)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_content)
            print(f"✓ Saved {output_filename}")
            return True
        else:
            print(f"✗ Empty response for {output_filename}")
            return False
            
    except Exception as e:
        print(f"✗ Error processing job for {output_filename}: {e}")
        return False

# ---- 3. main execution -------------------------------------------------------------
async def main():
    """Main function to run concurrent processing."""
    
    jobs_to_run = []
    for img_path, prompt_info, model_info in product(IMAGES, PROMPTS, MODELS):
        prompt_name, _ = prompt_info
        model_name = model_info['name']
        safe_model_name = model_name.replace('/', '_')
        
        # Find the highest existing run number for this combination
        run_num = 1
        while True:
            output_filename = f"{img_path.stem}_{prompt_name}_{safe_model_name}_{run_num}.xml"
            output_path = OUTPUT_DIR / output_filename
            if not output_path.exists():
                break
            run_num += 1
            
        for i in range(REPEATS):
            jobs_to_run.append((img_path, prompt_info, model_info, run_num + i - 1))

    total_jobs = len(jobs_to_run)
    print(f"Found {total_jobs} total jobs to process.")

    tasks = []
    for img_path, prompt_info, model_info, rep in jobs_to_run:
        tasks.append(asyncio.create_task(process_job(img_path, prompt_info, model_info, rep)))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and calculate success count
    success_count = 0
    for result in results:
        if result is True:
            success_count += 1
        elif isinstance(result, Exception):
            # asyncio.gather with return_exceptions=True returns exceptions in the list
            print(f"A job failed with an exception: {result}")

    # ---- 4. summary ---------------------------------------------------------------
    print(f"\nFinished. Saved {success_count} successful calls as XML out of {total_jobs} attempts.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting gracefully.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)
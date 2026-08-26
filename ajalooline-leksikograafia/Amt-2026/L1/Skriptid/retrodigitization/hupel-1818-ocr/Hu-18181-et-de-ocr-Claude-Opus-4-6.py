#!/usr/bin/env python3
# Created: 2026-02-21 13-58-22
# Author: Madis Jürviste
"""
Batch OCR Processor for Hupel's Estonian-German Dictionary (1818 Edition)

This script processes JPG images of dictionary pages through the Claude Opus 4.6
Batch API with adaptive thinking (effort=high) to produce structured OCR output
with custom markup. One output .txt file is created per input .jpg page.

Large page sets are automatically split across multiple batches to stay within
the Batch API's 256 MB size limit.

Usage:
    python Hu-18181-et-de-ocr.py [--start-page PAGE] [--end-page PAGE]
    python Hu-18181-et-de-ocr.py --dry-run
    python Hu-18181-et-de-ocr.py --batch-ids msgbatch_XXX,msgbatch_YYY

Requirements:
    - ANTHROPIC_API_KEY environment variable must be set
    - Input JPGs in input/input-JPG/ folder
    - Output written to output-raw/ folder
"""

import os
import re
import sys
import time
import base64
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    import anthropic
except ImportError:
    print("Error: anthropic package not installed.")
    print("Install with: pip install anthropic")
    sys.exit(1)


# Configuration
CONFIG = {
    "model": "claude-opus-4-6",
    "max_tokens": 128000,
}

# Batch API size limit: 256 MB max, use 200 MB as safe threshold
MAX_BATCH_SIZE = 200 * 1024 * 1024

# Paths
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input" / "input-JPG"
OUTPUT_DIR = BASE_DIR / "output-raw"
PROMPT_FILE = BASE_DIR / "prompt" / "Hu-1818-ocr-prompt.txt"

# Pricing: Claude Opus 4.6 Batch API (50% discount on standard rates)
# Standard: $5.00 input / $25.00 output per MTok
# Batch:    $2.50 input / $12.50 output per MTok
BATCH_INPUT_PRICE_PER_MTOK = 2.50
BATCH_OUTPUT_PRICE_PER_MTOK = 12.50

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE_DIR / "ocr_batch.log"),
    ],
)
logger = logging.getLogger(__name__)


# ── Additional instructions appended to every request ────────────────────────

ADDITIONAL_INSTRUCTIONS = (
    "\n\n---\n\n"
    "**Additional instructions:**\n\n"
    "**1. Page-boundary continuations (CRITICAL — no content may be lost):**\n"
    "You are processing ONE page at a time and cannot see adjacent pages. "
    "Therefore, override Section 4 of the prompt above as follows:\n"
    "- If the page begins with a CONTINUATION of an entry that started on a "
    "previous page (i.e. text at the very top that is clearly the tail end of "
    "an entry, not a new headword), you MUST transcribe it. Output it at the "
    "very beginning of your transcription for this page, before the "
    "\"--- page [N] ---\" marker, wrapped as:\n"
    "--- continuation from previous page ---\n"
    "[transcribe the continuation text here using the normal tags]\n"
    "--- end continuation ---\n"
    "- If an entry is CUT OFF at the bottom of the page (i.e. it clearly "
    "continues on the next page), transcribe everything that is visible on "
    "this page and add a comment line at the end of that entry:\n"
    "# CO46: entry cut off — continues on next page\n"
    "- NEVER silently drop text. Every piece of visible text on the page must "
    "appear in the output, whether it is a complete entry, a continuation, "
    "or a partial entry that is cut off.\n\n"
    "**2. Comments:**\n"
    "At the end of any entry where you have a comment, observation, or note "
    "— about unclear readings, ambiguities, interesting features, possible "
    "alternative readings, or anything else worth noting — add a comment line "
    "in this format:\n"
    "# CO46: (your comment here)\n"
    "Be generous with comments. Add one whenever you consider it helpful or "
    "necessary, not only in extreme cases of illegibility. This helps the "
    "scholars working with this transcription."
)


# ── Helper functions ─────────────────────────────────────────────────────────

def load_prompt() -> str:
    """Load the OCR prompt from file."""
    if not PROMPT_FILE.exists():
        logger.error(f"Prompt file not found: {PROMPT_FILE}")
        sys.exit(1)
    return PROMPT_FILE.read_text(encoding="utf-8")


def get_input_files() -> list[tuple[int, Path]]:
    """
    Get sorted list of input JPG files with their page numbers.
    Returns list of (page_number, file_path) tuples.
    """
    files = []
    pattern = re.compile(r"Hu-1818-et-de_dict_main_Page_(\d+)\.jpg$", re.IGNORECASE)

    for file_path in INPUT_DIR.glob("*.jpg"):
        match = pattern.search(file_path.name)
        if match:
            page_num = int(match.group(1))
            files.append((page_num, file_path))

    files.sort(key=lambda x: x[0])
    return files


def encode_image_to_base64(image_path: Path) -> tuple[str, str]:
    """
    Encode an image file to base64.
    Returns tuple of (base64_data, media_type).
    """
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    return data, "image/jpeg"


def generate_entry_id() -> str:
    """Generate a unique entry ID: unix epoch timestamp in microseconds."""
    timestamp = int(time.time() * 1000000)
    time.sleep(0.000001)
    return str(timestamp)


def assign_entry_ids(text: str, page_number: int) -> str:
    """
    Assign unique timestamp IDs to all [ENTRY_ID] placeholders.
    Each entry receives a microsecond-precision unix epoch ID.
    """
    entry_count = 0

    def replacer(match):
        nonlocal entry_count
        entry_count += 1
        return f'<entry xml:id="{generate_entry_id()}">'

    result = re.sub(r'<entry xml:id="\[ENTRY_ID\]">', replacer, text)

    logger.info(f"Page {page_number}: Assigned {entry_count} entry IDs")
    return result


def get_output_filename(page_number: int) -> str:
    """
    Generate output filename: PPP_Hu-1818_YYYYMMDD_raw.txt
    Example: 015_Hu-1818_20260221_raw.txt
    """
    date_str = datetime.now().strftime("%Y%m%d")
    return f"{page_number:03d}_Hu-1818_{date_str}_raw.txt"


# ── Batch chunking ───────────────────────────────────────────────────────────

def estimate_message_text_len(prompt: str) -> int:
    """Estimate the byte length of the text portion of each request message."""
    # The message text is: prompt + additional instructions + page instruction.
    # Page number varies by a few digits — negligible.
    sample = prompt + ADDITIONAL_INSTRUCTIONS + "\n\n---\n\nPlease transcribe page 999 shown in the image below:"
    return len(sample.encode("utf-8"))


def estimate_request_size(image_path: Path, message_text_len: int) -> int:
    """
    Estimate the JSON byte size of a single batch request.
    Dominant cost is the base64-encoded image data.
    """
    file_size = image_path.stat().st_size
    b64_size = ((file_size + 2) // 3) * 4
    # JSON keys, quotes, nesting, model/params structure
    json_overhead = 1000
    return b64_size + message_text_len + json_overhead


def chunk_input_files(
    input_files: list[tuple[int, Path]], prompt: str
) -> list[list[tuple[int, Path]]]:
    """
    Split input files into chunks where each chunk's estimated JSON size
    stays under MAX_BATCH_SIZE (200 MB safe threshold for the 256 MB API limit).
    """
    msg_text_len = estimate_message_text_len(prompt)

    chunks: list[list[tuple[int, Path]]] = []
    current_chunk: list[tuple[int, Path]] = []
    current_size = 0

    for page_num, file_path in input_files:
        req_size = estimate_request_size(file_path, msg_text_len)

        if current_chunk and current_size + req_size > MAX_BATCH_SIZE:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0

        current_chunk.append((page_num, file_path))
        current_size += req_size

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ── Batch request building ───────────────────────────────────────────────────

def build_batch_requests(
    chunk: list[tuple[int, Path]], prompt: str
) -> list[dict]:
    """
    Build the list of batch request dicts for one chunk.
    Each request processes one page image.
    """
    requests = []
    for page_number, file_path in chunk:
        logger.info(f"Encoding image: {file_path.name}")
        image_data, media_type = encode_image_to_base64(file_path)

        custom_id = f"page-{page_number:03d}"

        message_text = (
            f"{prompt}"
            f"{ADDITIONAL_INSTRUCTIONS}"
            f"\n\n---\n\n"
            f"Please transcribe page {page_number} shown in the image below:"
        )

        request = {
            "custom_id": custom_id,
            "params": {
                "model": CONFIG["model"],
                "max_tokens": CONFIG["max_tokens"],
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": message_text,
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                        ],
                    }
                ],
            },
        }
        requests.append(request)

    return requests


# ── Main batch orchestration ─────────────────────────────────────────────────

def poll_batches(client: anthropic.Anthropic, batch_ids: list[str]):
    """Poll all batch IDs until every batch has status 'ended'."""
    pending = set(batch_ids)
    poll_count = 0

    while pending:
        still_pending = set()
        for bid in sorted(pending):
            message_batch = client.messages.batches.retrieve(bid)
            status = message_batch.processing_status
            counts = message_batch.request_counts

            logger.info(
                f"[Poll #{poll_count}] {bid}: {status} | "
                f"processing={counts.processing} "
                f"succeeded={counts.succeeded} "
                f"errored={counts.errored} "
                f"expired={counts.expired}"
            )

            if status != "ended":
                still_pending.add(bid)

        pending = still_pending
        if pending:
            poll_count += 1
            time.sleep(60)


def collect_results(
    client: anthropic.Anthropic, batch_ids: list[str]
) -> tuple[dict, list[int]]:
    """
    Retrieve results from all batches. Write output files and return
    (page_stats dict, failed_pages list).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    page_stats = {}
    failed_pages = []

    for bid in batch_ids:
        logger.info(f"Retrieving results for batch {bid}...")

        for result in client.messages.batches.results(bid):
            custom_id = result.custom_id
            page_num = int(custom_id.split("-")[1])

            if result.result.type == "succeeded":
                message = result.result.message

                # Extract text content (skip thinking blocks)
                response_text = ""
                for block in message.content:
                    if block.type == "text":
                        response_text += block.text

                if not response_text.strip():
                    logger.warning(f"Empty response for page {page_num}")
                    failed_pages.append(page_num)
                    continue

                # Assign timestamp IDs to entry placeholders
                processed_text = assign_entry_ids(response_text, page_num)

                # Token usage
                usage = message.usage
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens

                # Calculate costs (Batch API pricing)
                input_cost = (input_tokens / 1_000_000) * BATCH_INPUT_PRICE_PER_MTOK
                output_cost = (output_tokens / 1_000_000) * BATCH_OUTPUT_PRICE_PER_MTOK
                page_cost = input_cost + output_cost

                page_stats[page_num] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "total_cost": page_cost,
                }

                # Write output file (one .txt per .jpg)
                output_filename = get_output_filename(page_num)
                output_path = OUTPUT_DIR / output_filename
                output_path.write_text(processed_text, encoding="utf-8")

                logger.info(
                    f"Page {page_num:03d}: saved {output_filename} | "
                    f"input: {input_tokens:,} | output: {output_tokens:,} | "
                    f"cost: ${page_cost:.4f}"
                )

            elif result.result.type == "errored":
                logger.error(f"Error for page {page_num}: {result.result.error}")
                failed_pages.append(page_num)
            elif result.result.type == "expired":
                logger.error(f"Expired: page {page_num}")
                failed_pages.append(page_num)
            elif result.result.type == "canceled":
                logger.warning(f"Canceled: page {page_num}")
                failed_pages.append(page_num)

    return page_stats, failed_pages


def log_summary(batch_ids: list[str], page_stats: dict, failed_pages: list[int]):
    """Log the final token counts and pricing breakdown."""
    total_input_tokens = sum(s["input_tokens"] for s in page_stats.values())
    total_output_tokens = sum(s["output_tokens"] for s in page_stats.values())
    total_tokens = total_input_tokens + total_output_tokens
    total_input_cost = (total_input_tokens / 1_000_000) * BATCH_INPUT_PRICE_PER_MTOK
    total_output_cost = (total_output_tokens / 1_000_000) * BATCH_OUTPUT_PRICE_PER_MTOK
    total_cost = total_input_cost + total_output_cost

    logger.info("")
    logger.info("=" * 80)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Batch IDs: {', '.join(batch_ids)}")
    logger.info(f"  Model    : {CONFIG['model']}")
    logger.info(f"  Thinking : adaptive (effort=high)")
    logger.info(f"  Pages    : {len(page_stats)} succeeded, {len(failed_pages)} failed")
    logger.info("")
    logger.info("-" * 80)
    logger.info("PER-PAGE PRICING BREAKDOWN (Claude Opus 4.6 Batch API)")
    logger.info(
        f"  {'Page':<8} {'Input Tok':>12} {'In Cost':>10} "
        f"{'Output Tok':>12} {'Out Cost':>10} {'Page Cost':>10}"
    )
    logger.info("-" * 80)

    for page_num in sorted(page_stats.keys()):
        s = page_stats[page_num]
        logger.info(
            f"  {page_num:03d}      "
            f"{s['input_tokens']:>12,} "
            f"${s['input_cost']:>8.4f} "
            f"{s['output_tokens']:>12,} "
            f"${s['output_cost']:>8.4f} "
            f"${s['total_cost']:>8.4f}"
        )

    logger.info("-" * 80)
    logger.info("")
    logger.info("TOKEN TOTALS")
    logger.info(f"  Total input tokens  : {total_input_tokens:>12,}")
    logger.info(f"  Total output tokens : {total_output_tokens:>12,}")
    logger.info(f"  Total tokens        : {total_tokens:>12,}")
    logger.info("")
    logger.info(
        f"COST SUMMARY  (Batch API: input ${BATCH_INPUT_PRICE_PER_MTOK}/MTok, "
        f"output ${BATCH_OUTPUT_PRICE_PER_MTOK}/MTok)"
    )
    logger.info(f"  Input cost  : ${total_input_cost:>10.4f}")
    logger.info(f"  Output cost : ${total_output_cost:>10.4f}")
    logger.info(f"  TOTAL COST  : ${total_cost:>10.4f}")
    logger.info("=" * 80)

    if failed_pages:
        logger.warning(f"Failed pages: {sorted(failed_pages)}")


def process_batches(
    client: anthropic.Anthropic,
    input_files: list[tuple[int, Path]],
    prompt: str,
    batch_ids: Optional[list[str]] = None,
):
    """
    Submit batches (or monitor existing ones) and process all results.
    Automatically splits large page sets into multiple batches.
    """

    if batch_ids is None:
        # Chunk input files into size-safe batches
        chunks = chunk_input_files(input_files, prompt)
        logger.info(
            f"Split {len(input_files)} pages into {len(chunks)} batch(es) "
            f"(limit ~{MAX_BATCH_SIZE // (1024 * 1024)} MB per batch)"
        )

        batch_ids = []
        for i, chunk in enumerate(chunks):
            page_range = f"{chunk[0][0]:03d}-{chunk[-1][0]:03d}"
            logger.info(f"Batch {i + 1}/{len(chunks)}: pages {page_range} ({len(chunk)} pages)")

            logger.info(f"  Preparing requests...")
            batch_requests = build_batch_requests(chunk, prompt)

            logger.info(f"  Submitting to Anthropic API...")
            message_batch = client.messages.batches.create(requests=batch_requests)
            batch_ids.append(message_batch.id)
            logger.info(f"  Created: {message_batch.id}")

        logger.info("")
        logger.info(f"All {len(batch_ids)} batch(es) submitted:")
        for bid in batch_ids:
            logger.info(f"  {bid}")
    else:
        logger.info(f"Monitoring {len(batch_ids)} existing batch(es):")
        for bid in batch_ids:
            logger.info(f"  {bid}")

    # Poll all batches until all have ended
    logger.info("")
    logger.info("Polling for batch completion (checking every 60s)...")
    poll_batches(client, batch_ids)

    # Retrieve and process results from all batches
    logger.info("")
    page_stats, failed_pages = collect_results(client, batch_ids)

    # Log final summary
    log_summary(batch_ids, page_stats, failed_pages)


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Batch OCR Processor for Hupel's Estonian-German Dictionary (1818 Edition)"
    )
    parser.add_argument(
        "--start-page",
        type=int,
        help="Starting page number (inclusive)",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        help="Ending page number (inclusive)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without submitting batch",
    )
    parser.add_argument(
        "--batch-ids",
        type=str,
        help="Comma-separated batch IDs to monitor/retrieve (e.g. msgbatch_XXX,msgbatch_YYY)",
    )

    args = parser.parse_args()

    # Get input files
    input_files = get_input_files()

    # Filter by page range if specified
    if args.start_page is not None:
        input_files = [(p, f) for p, f in input_files if p >= args.start_page]
    if args.end_page is not None:
        input_files = [(p, f) for p, f in input_files if p <= args.end_page]

    if args.dry_run:
        prompt = load_prompt()
        chunks = chunk_input_files(input_files, prompt)
        msg_text_len = estimate_message_text_len(prompt)

        print(f"Would process {len(input_files)} pages in {len(chunks)} batch(es):")
        print(f"  (batch size limit: ~{MAX_BATCH_SIZE // (1024 * 1024)} MB)\n")

        for i, chunk in enumerate(chunks):
            est_size = sum(estimate_request_size(fp, msg_text_len) for _, fp in chunk)
            page_range = f"{chunk[0][0]:03d}–{chunk[-1][0]:03d}"
            print(f"  Batch {i + 1}: {len(chunk)} pages (pages {page_range}, ~{est_size / (1024 * 1024):.0f} MB)")

        if input_files:
            print(f"\nModel        : {CONFIG['model']}")
            print(f"Max tokens   : {CONFIG['max_tokens']:,}")
            print(f"Thinking     : adaptive (effort=high)")
            print(f"Output dir   : {OUTPUT_DIR}")
            print(f"Output files : one .txt per .jpg (e.g. 015_Hu-1818_YYYYMMDD_raw.txt)")
            print(f"Batch pricing: input ${BATCH_INPUT_PRICE_PER_MTOK}/MTok, "
                  f"output ${BATCH_OUTPUT_PRICE_PER_MTOK}/MTok")

        return

    # Parse existing batch IDs if provided
    existing_batch_ids = None
    if args.batch_ids:
        existing_batch_ids = [bid.strip() for bid in args.batch_ids.split(",") if bid.strip()]

    if not input_files and not existing_batch_ids:
        logger.error(f"No input files found in {INPUT_DIR}")
        sys.exit(1)

    # Verify API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Initialize client
    client = anthropic.Anthropic(api_key=api_key)

    # Load prompt
    prompt = load_prompt()
    logger.info("Loaded OCR prompt")

    if input_files:
        logger.info(f"Found {len(input_files)} input JPG files")
        for page_num, fp in input_files:
            logger.info(f"  Page {page_num}: {fp.name}")

    process_batches(client, input_files, prompt, batch_ids=existing_batch_ids)


if __name__ == "__main__":
    main()

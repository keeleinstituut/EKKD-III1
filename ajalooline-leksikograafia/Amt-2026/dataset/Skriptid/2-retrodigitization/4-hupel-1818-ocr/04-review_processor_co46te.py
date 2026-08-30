#!/usr/bin/env python3
# Created: 2026-02-26 18-00-48
# Author: Madis Jürviste
"""
Batch Review Processor for Hupel's Estonian-German Dictionary (1818 Edition)

This script reviews and corrects the initial OCR output by comparing it
against the original scanned images using Claude Opus 4.6 via the Batch API
with adaptive thinking (effort=medium).

Usage:
    python review_processor_co46te.py --pages 15-20,25,30-35
    python review_processor_co46te.py --pages 55 --dry-run
    python review_processor_co46te.py --batch-ids msgbatch_XXX,msgbatch_YYY

Requirements:
    - ANTHROPIC_API_KEY environment variable must be set
    - Raw OCR output in output-raw/ folder
    - Input JPGs in input-BSB/input-chx folder
    - Corrected output written to output-review_co46te/ folder
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
    "temperature": 1.0,
    "max_tokens": 128000,
    "thinking_effort": "medium",
    "source_prefix": "6",
    "edition_code": "818",
}

# Batch API size limit: 256 MB max, use 200 MB as safe threshold
MAX_BATCH_SIZE = 200 * 1024 * 1024

# Paths
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input-BSB" / "input-chx"
RAW_OUTPUT_DIR = BASE_DIR / "output-raw"
REVIEW_OUTPUT_DIR = BASE_DIR / "output-review_co46te"
MAIN_PROMPT_FILE = BASE_DIR / "prompt" / "Hu-1818-ocr-prompt.txt"
REVIEW_PROMPT_FILE = BASE_DIR / "Hu-1818_review-prompt.txt"
CAT_EXAMPLE_FILE = BASE_DIR / "Hu-1818-et-de_CAT.txt"

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
        logging.FileHandler(BASE_DIR / "review_processor_co46te.log"),
    ],
)
logger = logging.getLogger(__name__)


# ── Helper functions ─────────────────────────────────────────────────────────

def load_file(file_path: Path) -> str:
    """Load text content from a file."""
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    return file_path.read_text(encoding="utf-8")


def parse_pages_arg(pages_str: str) -> set[int]:
    """
    Parse the --pages argument into a set of page numbers.
    Supports ranges (15-20), individual pages (25), and comma-separated
    combinations (15-20,25,30-35).
    """
    pages = set()
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start, end = int(start.strip()), int(end.strip())
            if start > end:
                logger.error(f"Invalid range: {part} (start > end)")
                sys.exit(1)
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return pages


def get_jpg_files() -> dict[int, Path]:
    """
    Get all available JPG files mapped by page number.
    Pattern: Hu-1818-et-de_dict_main_Page_NNN.jpg
    """
    files = {}
    pattern = re.compile(r"Hu-1818-et-de_dict_main_Page_(\d+)\.jpg$", re.IGNORECASE)

    for file_path in INPUT_DIR.glob("*.jpg"):
        match = pattern.search(file_path.name)
        if match:
            page_num = int(match.group(1))
            files[page_num] = file_path

    return files


def get_raw_ocr_files() -> dict[int, Path]:
    """
    Get all raw OCR output files mapped by page number.
    Pattern: PPP_Hu-1818_*_raw.txt — match by leading 3-digit page number.
    If multiple files exist for the same page, use the most recently modified one.
    """
    files: dict[int, Path] = {}
    pattern = re.compile(r"^(\d{3})_Hu-1818_.*_raw\.txt$")

    for file_path in RAW_OUTPUT_DIR.glob("*_Hu-1818_*_raw.txt"):
        match = pattern.search(file_path.name)
        if match:
            page_num = int(match.group(1))
            if page_num not in files or file_path.stat().st_mtime > files[page_num].stat().st_mtime:
                files[page_num] = file_path

    return files


def resolve_input_files(
    requested_pages: set[int],
) -> list[tuple[int, Path, Path]]:
    """
    For each requested page, find both JPG and raw OCR files.
    Returns sorted list of (page_number, jpg_path, raw_path) tuples.
    Reports any mismatches.
    """
    jpg_files = get_jpg_files()
    raw_files = get_raw_ocr_files()

    matched = []
    missing_jpg = []
    missing_raw = []

    for page_num in sorted(requested_pages):
        has_jpg = page_num in jpg_files
        has_raw = page_num in raw_files

        if has_jpg and has_raw:
            matched.append((page_num, jpg_files[page_num], raw_files[page_num]))
        elif not has_jpg and not has_raw:
            missing_jpg.append(page_num)
            missing_raw.append(page_num)
        elif not has_jpg:
            missing_jpg.append(page_num)
        else:
            missing_raw.append(page_num)

    if missing_jpg:
        logger.warning(f"Missing JPG files for pages: {missing_jpg}")
    if missing_raw:
        logger.warning(f"Missing raw OCR files for pages: {missing_raw}")

    if not matched:
        logger.error("No pages have both JPG and raw OCR files")
        sys.exit(1)

    return matched


def encode_image_to_base64(image_path: Path) -> tuple[str, str]:
    """
    Encode an image file to base64.
    Returns tuple of (base64_data, media_type).
    """
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, "image/jpeg"


def get_output_filename(raw_filename: str) -> str:
    """
    Derive output filename from raw filename by replacing '_raw' with '_reviewed'.
    Example: 015_Hu-1818_20260221_raw.txt → 015_Hu-1818_20260221_reviewed.txt
    """
    return raw_filename.replace("_raw.txt", "_reviewed.txt")


def renumber_entry_ids(text: str, page_number: int) -> str:
    """
    Renumber all entry IDs on a page to ensure correct sequence.

    - Existing entries: preserve timestamp, update sequence number
    - New entries ([NEW]): assign new sequence number and timestamp

    ID format: 6818-PPP-NNN-TIMESTAMP
    """
    entry_pattern = re.compile(r'<entry xml:id="([^"]+)">')
    matches = list(entry_pattern.finditer(text))

    if not matches:
        return text

    result = text
    offset = 0

    for seq_num, match in enumerate(matches, start=1):
        old_id = match.group(1)
        old_tag = match.group(0)

        if old_id == "[NEW]":
            timestamp = int(time.time() * 1000000)
            time.sleep(0.000001)
        else:
            parts = old_id.split("-")
            if len(parts) >= 4:
                timestamp = parts[-1]
            else:
                timestamp = int(time.time() * 1000000)
                time.sleep(0.000001)

        new_id = f"{CONFIG['source_prefix']}{CONFIG['edition_code']}-{page_number:03d}-{seq_num:03d}-{timestamp}"
        new_tag = f'<entry xml:id="{new_id}">'

        start = match.start() + offset
        end = match.end() + offset

        result = result[:start] + new_tag + result[end:]
        offset += len(new_tag) - len(old_tag)

    logger.info(f"Page {page_number}: Renumbered {len(matches)} entries")
    return result


# ── Batch chunking ───────────────────────────────────────────────────────────

def estimate_message_text_len(review_prompt: str, main_prompt: str, cat_example: str) -> int:
    """Estimate the byte length of the text portion of each request message."""
    sample = (
        f"## REVIEW AND CORRECTION INSTRUCTIONS\n\n{review_prompt}\n\n---\n\n"
        f"## ORIGINAL TRANSCRIPTION GUIDELINES (for reference)\n\n{main_prompt}\n\n---\n\n"
        f"## CAT EXAMPLE (ground truth reference)\n\n{cat_example[:5000]}\n\n---\n\n"
        f"## INITIAL OCR TRANSCRIPTION TO REVIEW\n\n```\n{'x' * 50000}\n```\n\n---\n\n"
        f"## ORIGINAL PAGE IMAGE\n\nThe image below shows page 999 of the dictionary."
    )
    return len(sample.encode("utf-8"))


def estimate_request_size(image_path: Path, message_text_len: int) -> int:
    """
    Estimate the JSON byte size of a single batch request.
    Dominant cost is the base64-encoded image data.
    """
    file_size = image_path.stat().st_size
    b64_size = ((file_size + 2) // 3) * 4
    json_overhead = 1000
    return b64_size + message_text_len + json_overhead


def chunk_input_files(
    input_files: list[tuple[int, Path, Path]],
    review_prompt: str,
    main_prompt: str,
    cat_example: str,
) -> list[list[tuple[int, Path, Path]]]:
    """
    Split input files into chunks where each chunk's estimated JSON size
    stays under MAX_BATCH_SIZE (200 MB safe threshold for the 256 MB API limit).
    """
    msg_text_len = estimate_message_text_len(review_prompt, main_prompt, cat_example)

    chunks: list[list[tuple[int, Path, Path]]] = []
    current_chunk: list[tuple[int, Path, Path]] = []
    current_size = 0

    for page_num, jpg_path, raw_path in input_files:
        req_size = estimate_request_size(jpg_path, msg_text_len)

        if current_chunk and current_size + req_size > MAX_BATCH_SIZE:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0

        current_chunk.append((page_num, jpg_path, raw_path))
        current_size += req_size

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ── Batch request building ───────────────────────────────────────────────────

def build_batch_requests(
    chunk: list[tuple[int, Path, Path]],
    review_prompt: str,
    main_prompt: str,
    cat_example: str,
) -> list[dict]:
    """
    Build the list of batch request dicts for one chunk.
    Each request processes one page image with its raw OCR text.
    """
    requests = []
    for page_number, jpg_path, raw_path in chunk:
        logger.info(f"Encoding image: {jpg_path.name}")
        image_data, media_type = encode_image_to_base64(jpg_path)

        raw_ocr_content = raw_path.read_text(encoding="utf-8")

        custom_id = f"review-page-{page_number:03d}"

        message_text = (
            f"## REVIEW AND CORRECTION INSTRUCTIONS\n\n"
            f"{review_prompt}\n\n"
            f"---\n\n"
            f"## ORIGINAL TRANSCRIPTION GUIDELINES (for reference)\n\n"
            f"{main_prompt}\n\n"
            f"---\n\n"
            f"## CAT EXAMPLE (ground truth reference)\n\n"
            f"The following is a verified, correct example of how dictionary entries should "
            f"look. Use it as a reference for formatting, tag usage, and entry structure:\n\n"
            f"```\n{cat_example}\n```\n\n"
            f"---\n\n"
            f"## INITIAL OCR TRANSCRIPTION TO REVIEW\n\n"
            f"```\n{raw_ocr_content}\n```\n\n"
            f"---\n\n"
            f"## ORIGINAL PAGE IMAGE\n\n"
            f"The image below shows page {page_number} of the dictionary. Compare it "
            f"against the transcription above and produce a corrected version.\n\n"
            f"Please review and correct the transcription following all the rules "
            f"specified above. Output ONLY the corrected transcription."
        )

        request = {
            "custom_id": custom_id,
            "params": {
                "model": CONFIG["model"],
                "max_tokens": CONFIG["max_tokens"],
                "temperature": CONFIG["temperature"],
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": CONFIG["thinking_effort"]},
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
    client: anthropic.Anthropic,
    batch_ids: list[str],
    raw_files: dict[int, Path],
) -> tuple[dict, list[int]]:
    """
    Retrieve results from all batches. Write output files and return
    (page_stats dict, failed_pages list).
    """
    REVIEW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    page_stats = {}
    failed_pages = []

    for bid in batch_ids:
        logger.info(f"Retrieving results for batch {bid}...")

        for result in client.messages.batches.results(bid):
            custom_id = result.custom_id
            # custom_id format: "review-page-NNN"
            page_num = int(custom_id.split("-")[2])

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

                # Renumber entry IDs
                processed_text = renumber_entry_ids(response_text, page_num)

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

                # Derive output filename from raw filename
                if page_num in raw_files:
                    output_filename = get_output_filename(raw_files[page_num].name)
                else:
                    date_str = datetime.now().strftime("%Y%m%d")
                    output_filename = f"{page_num:03d}_Hu-1818_{date_str}_reviewed.txt"

                output_path = REVIEW_OUTPUT_DIR / output_filename
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
    logger.info("=" * 90)
    logger.info("BATCH REVIEW PROCESSING COMPLETE")
    logger.info("=" * 90)
    logger.info(f"  Batch IDs: {', '.join(batch_ids)}")
    logger.info(f"  Model    : {CONFIG['model']}")
    logger.info(f"  Thinking : adaptive (effort={CONFIG['thinking_effort']})")
    logger.info(f"  Max tok  : {CONFIG['max_tokens']:,}")
    logger.info(f"  Pages    : {len(page_stats)} succeeded, {len(failed_pages)} failed")
    logger.info("")
    logger.info("-" * 90)
    logger.info("PER-PAGE PRICING BREAKDOWN (Claude Opus 4.6 Batch API)")
    logger.info(
        f"  {'Page':<8} {'Input Tok':>12} {'In Cost':>10} "
        f"{'Output Tok':>12} {'Out Cost':>10} {'Page Cost':>10}"
    )
    logger.info("-" * 90)

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

    logger.info("-" * 90)
    logger.info("")
    logger.info("TOKEN TOTALS")
    logger.info(f"  Total input tokens  : {total_input_tokens:>12,}")
    logger.info(f"  Total output tokens : {total_output_tokens:>12,} (includes thinking tokens)")
    logger.info(f"  Total tokens        : {total_tokens:>12,}")
    logger.info("")
    logger.info(
        f"COST SUMMARY  (Batch API: input ${BATCH_INPUT_PRICE_PER_MTOK}/MTok, "
        f"output ${BATCH_OUTPUT_PRICE_PER_MTOK}/MTok)"
    )
    logger.info(f"  Input cost  : ${total_input_cost:>10.4f}")
    logger.info(f"  Output cost : ${total_output_cost:>10.4f}")
    logger.info(f"  TOTAL COST  : ${total_cost:>10.4f}")
    logger.info("=" * 90)

    if failed_pages:
        logger.warning(f"Failed pages: {sorted(failed_pages)}")


def process_batches(
    client: anthropic.Anthropic,
    input_files: list[tuple[int, Path, Path]],
    review_prompt: str,
    main_prompt: str,
    cat_example: str,
    batch_ids: Optional[list[str]] = None,
):
    """
    Submit batches (or monitor existing ones) and process all results.
    Automatically splits large page sets into multiple batches.
    """
    # Build raw_files lookup for output naming
    raw_files = {page_num: raw_path for page_num, _, raw_path in input_files}

    if batch_ids is None:
        chunks = chunk_input_files(input_files, review_prompt, main_prompt, cat_example)
        logger.info(
            f"Split {len(input_files)} pages into {len(chunks)} batch(es) "
            f"(limit ~{MAX_BATCH_SIZE // (1024 * 1024)} MB per batch)"
        )

        batch_ids = []
        for i, chunk in enumerate(chunks):
            page_range = f"{chunk[0][0]:03d}-{chunk[-1][0]:03d}"
            logger.info(f"Batch {i + 1}/{len(chunks)}: pages {page_range} ({len(chunk)} pages)")

            logger.info(f"  Preparing requests...")
            batch_requests = build_batch_requests(chunk, review_prompt, main_prompt, cat_example)

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
    page_stats, failed_pages = collect_results(client, batch_ids, raw_files)

    # Log final summary
    log_summary(batch_ids, page_stats, failed_pages)


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Batch Review Processor — Claude Opus 4.6 with Adaptive Thinking (effort=medium)"
    )
    parser.add_argument(
        "--pages",
        type=str,
        help="Page numbers to process (e.g., 15-20,25,30-35)",
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

    # Parse existing batch IDs if provided
    existing_batch_ids = None
    if args.batch_ids:
        existing_batch_ids = [bid.strip() for bid in args.batch_ids.split(",") if bid.strip()]

    # Require either --pages or --batch-ids
    if not args.pages and not existing_batch_ids:
        parser.error("Either --pages or --batch-ids is required")

    # Resolve input files if --pages provided
    input_files = []
    if args.pages:
        requested_pages = parse_pages_arg(args.pages)
        input_files = resolve_input_files(requested_pages)

    if args.dry_run:
        if not input_files:
            print("No pages to process (--pages required for --dry-run)")
            return

        review_prompt = load_file(REVIEW_PROMPT_FILE)
        main_prompt = load_file(MAIN_PROMPT_FILE)
        cat_example = load_file(CAT_EXAMPLE_FILE)
        chunks = chunk_input_files(input_files, review_prompt, main_prompt, cat_example)
        msg_text_len = estimate_message_text_len(review_prompt, main_prompt, cat_example)

        print(f"Would review {len(input_files)} pages in {len(chunks)} batch(es):")
        print(f"  (batch size limit: ~{MAX_BATCH_SIZE // (1024 * 1024)} MB)\n")

        for i, chunk in enumerate(chunks):
            est_size = sum(estimate_request_size(jp, msg_text_len) for _, jp, _ in chunk)
            page_range = f"{chunk[0][0]:03d}–{chunk[-1][0]:03d}"
            print(f"  Batch {i + 1}: {len(chunk)} pages (pages {page_range}, ~{est_size / (1024 * 1024):.0f} MB)")

        print(f"\nPages:")
        for page_num, jpg_path, raw_path in input_files:
            output_name = get_output_filename(raw_path.name)
            print(f"  Page {page_num:03d}: {jpg_path.name} + {raw_path.name} → {output_name}")

        print(f"\nModel        : {CONFIG['model']}")
        print(f"Temperature  : {CONFIG['temperature']}")
        print(f"Max tokens   : {CONFIG['max_tokens']:,}")
        print(f"Thinking     : adaptive (effort={CONFIG['thinking_effort']})")
        print(f"Output dir   : {REVIEW_OUTPUT_DIR}")
        print(f"Batch pricing: input ${BATCH_INPUT_PRICE_PER_MTOK}/MTok, "
              f"output ${BATCH_OUTPUT_PRICE_PER_MTOK}/MTok")
        return

    # Verify API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Initialize client
    client = anthropic.Anthropic(api_key=api_key)

    # Load prompts and CAT example
    review_prompt = load_file(REVIEW_PROMPT_FILE)
    main_prompt = load_file(MAIN_PROMPT_FILE)
    cat_example = load_file(CAT_EXAMPLE_FILE)
    logger.info("Loaded review prompt, OCR prompt, and CAT example")

    if input_files:
        logger.info(f"Found {len(input_files)} pages to review")
        for page_num, jpg_path, raw_path in input_files:
            logger.info(f"  Page {page_num:03d}: {jpg_path.name} + {raw_path.name}")

    process_batches(
        client, input_files, review_prompt, main_prompt, cat_example,
        batch_ids=existing_batch_ids,
    )


if __name__ == "__main__":
    main()

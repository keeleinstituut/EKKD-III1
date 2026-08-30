#!/usr/bin/env python3
# Created: 2026-01-22 22-26-34
# Author: Madis Jürviste
"""
Resize PNG images to 90% of original dimensions.

Reads from: input/input_PNG/
Writes to:  input/input_PNG-sm/
"""

from pathlib import Path
from PIL import Image

INPUT_DIR = Path(__file__).parent / "input" / "input_PNG"
OUTPUT_DIR = Path(__file__).parent / "input" / "input_PNG-sm"
SCALE = 0.90  # 90% of original size (10% reduction)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    png_files = sorted(INPUT_DIR.glob("*.png"))
    print(f"Found {len(png_files)} PNG files to resize")

    for png_path in png_files:
        with Image.open(png_path) as img:
            orig_width, orig_height = img.size
            new_width = int(orig_width * SCALE)
            new_height = int(orig_height * SCALE)

            # Use LANCZOS for high-quality downscaling
            resized = img.resize((new_width, new_height), Image.LANCZOS)

            output_path = OUTPUT_DIR / png_path.name
            resized.save(output_path, format='PNG', optimize=True)

            orig_size = png_path.stat().st_size
            new_size = output_path.stat().st_size
            reduction = (1 - new_size / orig_size) * 100

            print(f"{png_path.name}: {orig_width}x{orig_height} -> {new_width}x{new_height} "
                  f"({orig_size:,} -> {new_size:,} bytes, {reduction:.1f}% smaller)")

    print(f"\nDone! Resized images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

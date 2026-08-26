#!/usr/bin/env python3
# Created: 2025-12-17 21-32-36
# Author: Madis Jürviste
"""
Create zip files containing 10 txt files each from pages-by-10 folder.
Organizes files by page number ranges (135-144, 145-154, etc.)
"""

from pathlib import Path
import zipfile
import re

def main():
    # Setup paths
    base_dir = Path(__file__).parent
    source_dir = base_dir / "pages-by-10"

    # Get all txt files, sorted by page number
    txt_files = sorted(source_dir.glob("*.txt"),
                      key=lambda p: int(re.match(r'(\d+)_', p.name).group(1)))

    print(f"Found {len(txt_files)} txt files")
    print()

    # Group files into batches of 10
    batch_size = 10
    batches = []

    for i in range(0, len(txt_files), batch_size):
        batch = txt_files[i:i+batch_size]
        batches.append(batch)

    # Create zip files for each batch
    for batch in batches:
        # Get page range for this batch
        first_page = int(re.match(r'(\d+)_', batch[0].name).group(1))
        last_page = int(re.match(r'(\d+)_', batch[-1].name).group(1))

        # Create zip filename
        zip_filename = f"pages_{first_page:03d}-{last_page:03d}.zip"
        zip_path = source_dir / zip_filename

        # Create zip file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for txt_file in batch:
                # Add file to zip with just the filename (no path)
                zipf.write(txt_file, txt_file.name)

        print(f"Created: {zip_filename} ({len(batch)} files)")

    print()
    print(f"Successfully created {len(batches)} zip files in {source_dir}")

if __name__ == "__main__":
    main()

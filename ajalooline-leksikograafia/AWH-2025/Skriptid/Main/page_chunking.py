import os
import pathlib
from PIL import Image
import argparse

def split_image_into_chunks(image_path, output_dir, num_chunks_per_col=4, overlap_ratio=0.1):
    """
    Splits an image into two vertical columns, then splits each column into
    overlapping horizontal chunks.
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            midpoint = width // 2
            
            columns = {
                "column1": img.crop((0, 0, midpoint, height)),
                "column2": img.crop((midpoint, 0, width, height))
            }
            
            output_paths = []
            base_filename = os.path.basename(image_path)
            file_root, file_ext = os.path.splitext(base_filename)

            for col_name, col_img in columns.items():
                col_width, col_height = col_img.size
                
                if col_height == 0: continue

                # Calculate the height of each chunk without overlap
                chunk_height_base = col_height / num_chunks_per_col
                overlap_px = int(chunk_height_base * overlap_ratio)

                for i in range(num_chunks_per_col):
                    y_start = i * chunk_height_base
                    
                    # Add overlap
                    top_overlap = overlap_px // 2
                    bottom_overlap = overlap_px - top_overlap
                    
                    crop_top = int(max(0, y_start - top_overlap))
                    crop_bottom = int(min(col_height, y_start + chunk_height_base + bottom_overlap))

                    # For first and last chunk, ensure they align with the image borders
                    if i == 0:
                        crop_top = 0
                    if i == num_chunks_per_col - 1:
                        crop_bottom = col_height

                    if crop_top >= crop_bottom:
                        continue
                    
                    chunk_box = (0, crop_top, col_width, crop_bottom)
                    chunk_img = col_img.crop(chunk_box)
                    
                    output_chunk_filename = f"{file_root}_{col_name}_chunk{i+1}{file_ext}"
                    output_chunk_path = os.path.join(output_dir, output_chunk_filename)
                    
                    chunk_img.save(output_chunk_path)
                    output_paths.append(pathlib.Path(output_chunk_path))
            
            return output_paths
    except Exception as e:
        print(f"An unexpected error occurred while splitting {image_path}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Split images into smaller, overlapping chunks.")
    parser.add_argument("input_dir", help="Directory with images to chunk.")
    parser.add_argument("output_dir", help="Directory to save the chunked images.")
    parser.add_argument("--num-chunks", type=int, default=4, help="Number of chunks to create per column.")
    parser.add_argument("--overlap-ratio", type=float, default=0.1, help="Overlap ratio between chunks.")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)

    output_dir.mkdir(exist_ok=True)

    images = list(input_dir.glob("*.jpg"))
    if not images:
        print(f"No .jpg images found in {input_dir}")
        return

    print(f"--- Splitting {len(images)} images into chunks ---")
    total_chunks = 0
    for img_path in images:
        chunks = split_image_into_chunks(img_path, output_dir, args.num_chunks, args.overlap_ratio)
        total_chunks += len(chunks)
    print(f"Successfully split images into {total_chunks} chunks in {output_dir}")


if __name__ == "__main__":
    main()

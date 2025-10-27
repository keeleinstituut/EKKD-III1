import sys
import os
import pandas as pd
from lxml import etree
from difflib import SequenceMatcher
import re

def get_xml_root(filepath):
    """Parse an XML file and return its root element."""
    try:
        # remove_blank_text=True helps in ignoring formatting-only differences
        parser = etree.XMLParser(remove_blank_text=True, recover=True)
        tree = etree.parse(filepath, parser)
        return tree.getroot()
    except etree.XMLSyntaxError as e:
        print(f"Error parsing {filepath}: {e}", file=sys.stderr)
        return None
    except IOError as e:
        print(f"Error reading file {filepath}: {e}", file=sys.stderr)
        return None

def get_structural_elements(element, with_attributes=True):
    """
    Generate a list of structural elements (tags, and optionally attributes) for comparison.
    """
    structural_elements = []
    for el in element.iter():
        structural_elements.append(el.tag)
        if with_attributes:
            # Sorting attributes for a canonical representation
            for key, value in sorted(el.attrib.items()):
                structural_elements.append(f'@{key}={value}')
    return structural_elements

def get_normalized_content(element):
    """
    Extract and normalize all text content from an element.
    It concatenates all text nodes and normalizes whitespace.
    """
    text_content = ' '.join(el.strip() for el in element.itertext() if el.strip())
    return ' '.join(text_content.split()) # Normalize whitespace

def compare_sequences(seq1, seq2):
    """Compare two sequences and return a similarity ratio."""
    return SequenceMatcher(None, seq1, seq2).ratio()

def main():
    if len(sys.argv) != 3:
        sys.exit(1)

    output_dir = sys.argv[1]
    ground_truth_dir = sys.argv[2]

    results = []

    for filename in os.listdir(output_dir):
        if not filename.endswith(".xml"):
            continue

        try:
            filename_stem = os.path.splitext(filename)[0]
            parts = filename_stem.split('_')
            
            if len(parts) < 5:
                print(f"Warning: Filename {filename} does not have enough parts, skipping.", file=sys.stderr)
                continue

            page = parts[0]
            prompt = parts[1]
            provider = parts[2]
            run = parts[-1]
            model = '_'.join(parts[3:-1])

            if not page.startswith('page') or not run.isdigit():
                print(f"Warning: Filename {filename} does not match expected format (page/run), skipping.", file=sys.stderr)
                continue
        except IndexError:
            print(f"Warning: Filename {filename} does not match expected format, skipping.", file=sys.stderr)
            continue
            
        ground_truth_filename = f"{page}.xml"
        ground_truth_path = os.path.join(ground_truth_dir, ground_truth_filename)

        if not os.path.exists(ground_truth_path):
            print(f"Warning: Ground truth file not found for {filename}, skipping.", file=sys.stderr)
            continue

        output_path = os.path.join(output_dir, filename)
        
        root1 = get_xml_root(output_path)
        root2 = get_xml_root(ground_truth_path)

        if root1 is None or root2 is None:
            continue

        # Structural similarity (with attributes)
        structure1_with_attrs = get_structural_elements(root1, with_attributes=True)
        structure2_with_attrs = get_structural_elements(root2, with_attributes=True)
        structural_similarity_attrs = compare_sequences(structure1_with_attrs, structure2_with_attrs)

        # Structural similarity (tags only)
        structure1_tags_only = get_structural_elements(root1, with_attributes=False)
        structure2_tags_only = get_structural_elements(root2, with_attributes=False)
        structural_similarity_tags = compare_sequences(structure1_tags_only, structure2_tags_only)

        # Content similarity
        content1 = get_normalized_content(root1)
        content2 = get_normalized_content(root2)
        content_similarity = compare_sequences(content1, content2)
        
        results.append({
            "file": filename,
            "page": page,
            "prompt": prompt,
            "provider": provider,
            "model": model,
            "run": int(run),
            "structural_similarity_attrs": structural_similarity_attrs,
            "structural_similarity_tags": structural_similarity_tags,
            "content_similarity": content_similarity
        })

    if not results:
        print("No results to save. Exiting.")
        return

    df = pd.DataFrame(results)
    output_csv_path = os.path.join(os.path.dirname(output_dir), "comparison_results.csv")
    df.to_csv(output_csv_path, index=False)
    
    print(f"Comparison complete. Results saved to {output_csv_path}")


if __name__ == "__main__":
    main()

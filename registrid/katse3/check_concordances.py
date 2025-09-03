# Checking whether sentences exist in files which contain concordances from the Estonian National Corpus 2023
# Created by Esta Prangel in collaboration with Anthropic Claude Sonnet 4

import pandas as pd
import os
import glob
from pathlib import Path
import string


def find_matching_file(keyword, directory="."):
    """
    Find file matching the pattern: keyword_full_context_only.txt
    """
    pattern = f"{keyword}_full_context_only.txt"
    matching_files = glob.glob(os.path.join(directory, pattern))

    if matching_files:
        return matching_files[0]
    return None


def read_file_content(file_path):
    """
    Read file content and return as string, handling potential encoding issues
    """
    encodings = ['utf-8', 'windows-1252', 'iso-8859-1', 'cp1252']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                # Remove HTML tags and normalize spaces
                content = content.replace('<p>', '').replace('</p>', '').replace('<s>', '').replace('</s>', '').replace(
                    ' ,', ',').replace("' ", "'").replace(' .', '.')
                # Replace multiple spaces with single space
                content = ' '.join(content.split())
                # Convert to lowercase for comparison
                content = content.lower()
                return content
        except UnicodeDecodeError:
            continue

    print(f"Warning: Could not decode {file_path} with any standard encoding")
    return ""


def normalize_sentence(sentence):
    """
    Normalize a sentence for comparison:
    - Convert to lowercase
    - Strip trailing punctuation
    - Remove extra spaces
    """
    sentence = sentence.strip().lower()

    # Remove trailing punctuation (.,!?;:)
    while sentence and sentence[-1] in '.,!?;:':
        sentence = sentence[:-1]

    # Normalize spaces
    sentence = ' '.join(sentence.split())

    return sentence


def check_sentences_in_file(sentences, file_content):
    """
    Check which sentences exist in the file content
    Returns list of tuples: (original_sentence, found_status)
    """
    if not file_content:
        return [(sentence.strip(), "FILE_NOT_FOUND") for sentence in sentences]

    results = []
    for sentence in sentences:
        original_sentence = sentence.strip()
        if original_sentence:
            # Normalize the sentence for searching
            normalized = normalize_sentence(original_sentence)

            # Check if the normalized sentence appears in the file content
            if normalized and normalized in file_content:
                results.append((original_sentence, "FOUND"))
            else:
                results.append((original_sentence, "NOT_FOUND"))

    return results


def process_sheet(df, sheet_name, file_directory="."):
    """
    Process a single sheet of data
    """
    print(f"\nProcessing sheet: {sheet_name}")

    # Forward-fill empty keyword cells (column A)
    keyword_col = df.columns[0]
    df[keyword_col] = df[keyword_col].ffill()

    # Assuming columns are A, B (0, 1 indices)
    keyword_col = df.columns[0]  # Column A
    content_col = df.columns[1]  # Column B

    # Add new columns for results
    df['sentence_results'] = ""
    df['found_sentences'] = ""
    df['not_found_sentences'] = ""
    df['file_status'] = ""

    for index, row in df.iterrows():
        keyword = str(row[keyword_col]).strip()
        content = str(row[content_col])

        print(f"  Processing row {index + 1}: {keyword}")

        file_path = find_matching_file(keyword, file_directory)

        if file_path:
            print(f"    Found file: {file_path}")
            file_content = read_file_content(file_path)
            df.at[index, 'file_status'] = f"FOUND: {os.path.basename(file_path)}"
        else:
            print(f"    No matching file found for pattern: {keyword}_full_context_only.txt")
            file_content = ""
            df.at[index, 'file_status'] = "FILE_NOT_FOUND"

        sentences = content.split('|')

        sentence_results = check_sentences_in_file(sentences, file_content)

        found_sentences = []
        not_found_sentences = []
        detailed_results = []

        for sentence, status in sentence_results:
            if sentence:
                detailed_results.append(f"{sentence} [{status}]")
                if status == "FOUND":
                    found_sentences.append(sentence)
                else:
                    not_found_sentences.append(sentence)

        df.at[index, 'sentence_results'] = " | ".join(detailed_results)
        df.at[index, 'found_sentences'] = " | ".join(found_sentences)
        df.at[index, 'not_found_sentences'] = " | ".join(not_found_sentences)

        print(f"    Found: {len(found_sentences)} sentences")
        print(f"    Not found: {len(not_found_sentences)} sentences")

    return df


def process_excel_file(input_file, output_file=None, file_directory="."):
    """
    Main function to process the Excel file with multiple sheets
    """
    if output_file is None:
        path = Path(input_file)
        output_dir = Path("contexts")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{path.stem}_processed{path.suffix}"

    try:
        # Read all sheets from the Excel file
        all_sheets = pd.read_excel(input_file, sheet_name=None)
        print(f"Found {len(all_sheets)} sheets: {list(all_sheets.keys())}")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Dictionary to store processed sheets
    processed_sheets = {}

    # Dictionary to store filtered sheets (only rows with found sentences)
    filtered_sheets = {}

    # Process each sheet
    for sheet_name, df in all_sheets.items():
        print(f"\n{'=' * 50}")
        print(f"Processing sheet: {sheet_name}")
        print(f"{'=' * 50}")

        processed_df = process_sheet(df.copy(), sheet_name, file_directory)
        processed_sheets[sheet_name] = processed_df

        # Create filtered version (only rows with found sentences)
        found_rows = processed_df[processed_df['found_sentences'].str.len() > 0]
        if not found_rows.empty:
            filtered_sheets[sheet_name] = found_rows

    # Create statistics sheet
    stats_data = []

    for sheet_name, df in processed_sheets.items():
        # Calculate sentence-level statistics
        total_sentences = 0
        found_sentences = 0
        not_found_sentences = 0

        for _, row in df.iterrows():
            # Count sentences in this row
            content = str(row[df.columns[1]])  # Column B (content)
            sentences = [s.strip() for s in content.split('|') if s.strip()]
            row_total_sentences = len(sentences)

            # Count found and not found sentences
            found_list = str(row['found_sentences']).split('|') if str(row['found_sentences']) != 'nan' and str(
                row['found_sentences']) else []
            not_found_list = str(row['not_found_sentences']).split('|') if str(
                row['not_found_sentences']) != 'nan' and str(row['not_found_sentences']) else []

            row_found = len([s for s in found_list if s.strip()])
            row_not_found = len([s for s in not_found_list if s.strip()])

            total_sentences += row_total_sentences
            found_sentences += row_found
            not_found_sentences += row_not_found

        # Calculate row-level statistics
        total_rows = len(df)

        # Calculate percentages
        found_percentage = (found_sentences / total_sentences * 100) if total_sentences > 0 else 0
        not_found_percentage = (not_found_sentences / total_sentences * 100) if total_sentences > 0 else 0

        stats_data.append({
            'Sheet Name': sheet_name,
            'Total Rows': total_rows,
            'Total Sentences': total_sentences,
            'Found Sentences': found_sentences,
            'Not Found Sentences': not_found_sentences,
            'Found Sentences %': round(found_percentage, 2),
            'Not Found Sentences %': round(not_found_percentage, 2)
        })

    # Create overall totals
    total_rows_all = sum([stat['Total Rows'] for stat in stats_data])
    total_sentences_all = sum([stat['Total Sentences'] for stat in stats_data])
    total_found_all = sum([stat['Found Sentences'] for stat in stats_data])
    total_not_found_all = sum([stat['Not Found Sentences'] for stat in stats_data])

    # Add totals row
    stats_data.append({
        'Sheet Name': 'TOTAL (All Sheets)',
        'Total Rows': total_rows_all,
        'Total Sentences': total_sentences_all,
        'Found Sentences': total_found_all,
        'Not Found Sentences': total_not_found_all,
        'Found Sentences %': round(total_found_all / total_sentences_all * 100, 2) if total_sentences_all > 0 else 0,
        'Not Found Sentences %': round(total_not_found_all / total_sentences_all * 100,
                                       2) if total_sentences_all > 0 else 0,
        })

    # Create DataFrame for statistics
    stats_df = pd.DataFrame(stats_data)

    # Save all processed sheets to Excel file including statistics
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Write original processed sheets
            for sheet_name, df in processed_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Write statistics sheet
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)

            # Auto-adjust column widths for statistics sheet
            workbook = writer.book
            stats_worksheet = workbook['Statistics']
            for column in stats_worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 30)  # Max width of 30
                stats_worksheet.column_dimensions[column_letter].width = adjusted_width

        print(f"\nAll processed sheets (including Statistics sheet) saved to: {output_file}")

    except Exception as e:
        print(f"Error saving file: {e}")


if __name__ == "__main__":
    input_excel = "tabel.xlsx"
    text_files_directory = "contexts"

    process_excel_file(input_excel, file_directory=text_files_directory)

    print("\nProcessing complete!")

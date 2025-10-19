# Author: Joonatan Jakobson

import pandas as pd
import math
from tqdm import tqdm
import string
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import jiwer
from thefuzz import fuzz
import os

# --- Configuration ---
GROUND_TRUTH_FILE = 'corrected/WIP_ATH_Master-1_C37S-NT-API_20250306_output(ATH_Master-1_NT_20250306_output).csv' # Path to your LLM output CSV
LLM_OUTPUT_FILE = 'original/ORIGO_ATH_Master-1_C37S-NT-API_20250306_output(ATH_Master-1_NT_20250306_output)(1).csv' # Path to your corrected ground truth CSV
KEY_COLUMN = 'estonian_headword' # Column used to align entries
SECONDARY_KEY_COLUMN = 'german_equivalent' # Secondary column for fuzzy matching confirmation
DELIMITER = ';'
# Define the columns you want to compare (must exist in both files after loading)
COLUMNS_TO_COMPARE = [
    'estonian_headword',
    'german_equivalent',
]
HEADWORD_FUZZY_THRESHOLD = 30 # Similarity threshold (0-100) for primary key fuzzy matching
GERMAN_EQUIVALENT_FUZZY_THRESHOLD = 30 # Similarity threshold (0-100) for secondary key fuzzy matching
SEQUENCE_WINDOW = 10 # How many rows +/- to look around the current GT index in LLM for matches
OUTPUT_DIR = "publication_plots" # Directory to save the plot

# --- Helper Functions ---

def normalize_string(text):
    """Converts to lowercase, removes punctuation, and collapses whitespace."""
    if not isinstance(text, str):
        text = str(text)
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    # Convert to lowercase and collapse whitespace (strips lead/trail too)
    text = ' '.join(text.lower().split())
    return text

def calculate_cer(s1, s2):
    """Calculates Character Error Rate using jiwer AFTER NORMALIZATION."""
    # Normalize first
    s1_norm = normalize_string(s1)
    s2_norm = normalize_string(s2)

    if not s1_norm and not s2_norm:
        return 0.0
    if not s1_norm or not s2_norm:
        return 1.0 # Treat complete mismatch (one empty, one not) as 100% error
    try:
        # CER is calculated on the normalized strings
        return jiwer.cer(s1_norm, s2_norm)
    except ValueError: # Handle cases jiwer might struggle with
        return 1.0 if s1_norm != s2_norm else 0.0

def is_nan(value):
    """Check if a value is NaN or None."""
    return value is None or (isinstance(value, float) and math.isnan(value))

def safe_str(value):
    """Convert value to string, handling NaN/None as empty string."""
    return "" if is_nan(value) else str(value)

def align_entries(llm_df, gt_df, key_column, llm_secondary_key_column, gt_secondary_key_column, headword_fuzzy_threshold, german_fuzzy_threshold, sequence_window):
    """
    Aligns entries between LLM output and Ground Truth, handling primary (headword-focused)
    and fallback (secondary-key-driven + synonym check) matching.
    Allows one LLM row to potentially match multiple GT rows.

    Args:
        llm_df (pd.DataFrame): DataFrame of LLM output (for a single page).
        gt_df (pd.DataFrame): DataFrame of Ground Truth data (for a single page).
        key_column (str): The primary column name for alignment (e.g., 'estonian_headword').
        llm_secondary_key_column (str | None): The secondary column name in the LLM df.
        gt_secondary_key_column (str | None): The secondary column name in the GT df.
        headword_fuzzy_threshold (int): Min similarity score (0-100) for the primary key.
        german_fuzzy_threshold (int): Min similarity score (0-100) for the secondary key.
        sequence_window (int): Max distance +/- in indices to search for a match (within a page).
        SYNONYM_FALLBACK_THRESHOLD (int): Min similarity score (0-100) for matching GT headword within LLM synonyms during fallback.

    Returns:
        tuple: Contains:
            - list: aligned_pairs [(llm_row_dict, gt_row_dict), ...]
            - list: unaligned_llm [llm_row_dict, ...] (LLM rows not matched to any GT row on this page)
            - list: unaligned_gt [gt_row_dict, ...] (GT rows not matched for this page)
    """
    SYNONYM_FALLBACK_THRESHOLD = 85 # Threshold for matching GT key within LLM synonyms/key in fallback
    aligned_pairs = []
    unaligned_gt_rows = []
    llm_dict_list = llm_df.to_dict('records')
    gt_dict_list = gt_df.to_dict('records')
    llm_len = len(llm_dict_list)
    gt_len = len(gt_dict_list)

    matched_llm_indices = set()
    last_llm_match_idx_for_window = -1

    # Identify the name of the LLM synonym column IF AVAILABLE
    llm_synonym_col_name = 'estonian_synonyms'
    if llm_synonym_col_name not in llm_df.columns:
         llm_synonym_col_name = None


    for gt_index in range(gt_len):
        gt_row = gt_dict_list[gt_index]
        gt_key_orig = safe_str(gt_row.get(key_column, ''))
        gt_secondary_orig = safe_str(gt_row.get(gt_secondary_key_column, '')) if gt_secondary_key_column else ''
        # Normalize GT keys
        gt_key = normalize_string(gt_key_orig)
        gt_secondary = normalize_string(gt_secondary_orig)

        best_match_info = None # {llm_idx, score, dist, type: 'primary_exact' | 'primary_fuzzy' | 'fallback'}

        if gt_key:
            search_start = max(0, min(gt_index, last_llm_match_idx_for_window + 1) - sequence_window)
            search_end = min(llm_len, max(gt_index, last_llm_match_idx_for_window + 1) + sequence_window + 1)

            for llm_idx in range(search_start, search_end):
                llm_row_dict = llm_dict_list[llm_idx]
                llm_key_orig = safe_str(llm_row_dict.get(key_column, ''))
                llm_secondary_orig = safe_str(llm_row_dict.get(llm_secondary_key_column, '')) if llm_secondary_key_column else ''
                llm_synonyms_orig = safe_str(llm_row_dict.get(llm_synonym_col_name, '')) if llm_synonym_col_name else ''
                # Normalize LLM keys and synonyms
                llm_key = normalize_string(llm_key_orig)
                llm_secondary = normalize_string(llm_secondary_orig)
                llm_synonyms = normalize_string(llm_synonyms_orig)

                current_dist = abs(llm_idx - gt_index)
                current_match_type = None
                current_score = -1

                use_secondary_check = bool(gt_secondary_key_column and llm_secondary_key_column)

                # --- 1. Try Primary Match (Headword-focused) ---
                # Compare normalized keys
                if llm_key == gt_key:
                    passes_secondary_primary = True
                    german_score_primary = 0
                    # Use normalized secondary keys for comparison
                    if use_secondary_check:
                        if gt_secondary and llm_secondary:
                            german_score_primary = fuzz.token_set_ratio(gt_secondary, llm_secondary) # Fuzz works well with normalized
                            passes_secondary_primary = german_score_primary >= german_fuzzy_threshold
                        elif bool(gt_secondary) != bool(llm_secondary): # Normalized presence check
                            passes_secondary_primary = False
                        # else: both empty, passes_secondary_primary remains True

                    if passes_secondary_primary:
                        current_score = 300 - current_dist # High score for exact primary
                        current_match_type = 'primary_exact'

                # 1b. Fuzzy Headword Match + Secondary Check (if no exact match found yet)
                if current_match_type is None:
                    # Use normalized keys for fuzzy matching
                    headword_score = fuzz.ratio(gt_key, llm_key)
                    passes_headword = headword_score >= headword_fuzzy_threshold

                    if passes_headword:
                        passes_secondary_fuzzy = True
                        german_score_fuzzy = 0
                        # Use normalized secondary keys
                        if use_secondary_check:
                            if gt_secondary and llm_secondary:
                                german_score_fuzzy = fuzz.token_set_ratio(gt_secondary, llm_secondary)
                                passes_secondary_fuzzy = german_score_fuzzy >= german_fuzzy_threshold
                            elif bool(gt_secondary) != bool(llm_secondary): # Normalized presence check
                                passes_secondary_fuzzy = False
                            # else: both empty, passes_secondary_fuzzy remains True

                        if passes_secondary_fuzzy:
                            current_score = headword_score + (german_score_fuzzy if use_secondary_check and german_score_fuzzy > 0 else headword_score) - (current_dist * 0.1)
                            current_match_type = 'primary_fuzzy'

                # --- 2. Try Fallback Match (Secondary-driven, check headword in LLM key/synonyms) ---
                # Only try if no primary match found AND GT has a normalized secondary key
                if current_match_type is None and gt_secondary and use_secondary_check:
                    passes_secondary_fallback = False
                    german_score_fallback = 0
                    # Use normalized secondary keys
                    if llm_secondary: # LLM must also have normalized secondary for comparison
                        german_score_fallback = fuzz.token_set_ratio(gt_secondary, llm_secondary)
                        passes_secondary_fallback = german_score_fallback >= german_fuzzy_threshold

                    if passes_secondary_fallback:
                        # Secondary keys match (normalized)! Now check if normalized gt_key is in normalized llm_key or normalized llm_synonyms
                        found_score = 0
                        # Check within normalized LLM key
                        score_in_key = fuzz.partial_ratio(gt_key, llm_key)
                        if score_in_key > found_score: found_score = score_in_key

                        # Check within normalized LLM synonyms field
                        if llm_synonym_col_name and llm_synonyms:
                             # Check partial ratio against the whole normalized synonym string
                             score_in_syn_str = fuzz.partial_ratio(gt_key, llm_synonyms)
                             if score_in_syn_str > found_score: found_score = score_in_syn_str

                        if found_score >= SYNONYM_FALLBACK_THRESHOLD:
                            # Success! Assign a score based on german and found score
                            current_score = 100 + german_score_fallback + found_score - (current_dist * 0.2) # Lower base, heavier distance penalty
                            current_match_type = 'fallback'


                # --- Update Best Match For This GT Row Across All LLM Candidates --- 
                if current_match_type:
                    # Favor primary matches slightly if scores are identical
                    is_better = False
                    if best_match_info is None:
                        is_better = True
                    else:
                        # Higher score wins
                        if current_score > best_match_info['score']:
                             is_better = True
                        elif current_score == best_match_info['score']:
                             # Same score: closer distance wins
                             if current_dist < best_match_info['dist']:
                                 is_better = True
                             # Same score, same distance: prefer primary over fallback
                             elif current_dist == best_match_info['dist'] and \
                                  best_match_info['type'] == 'fallback' and \
                                  current_match_type != 'fallback':
                                      is_better = True

                    if is_better:
                        best_match_info = {'llm_idx': llm_idx, 'score': current_score, 'dist': current_dist, 'type': current_match_type}


        # --- Record the outcome for this GT row ---
        if best_match_info:
            best_llm_idx = best_match_info['llm_idx']
            aligned_pairs.append((llm_dict_list[best_llm_idx], gt_row))
            matched_llm_indices.add(best_llm_idx)
            last_llm_match_idx_for_window = max(last_llm_match_idx_for_window, best_llm_idx) # Update window guide smoothly
        else:
            unaligned_gt_rows.append(gt_row)


    # Identify unaligned LLM entries
    unaligned_llm_rows = [llm_dict_list[i] for i in range(llm_len) if i not in matched_llm_indices]

    return aligned_pairs, unaligned_llm_rows, unaligned_gt_rows


def create_cer_by_field_and_page_visualization(per_page_metrics, columns_to_plot, output_dir):
    """Generates and displays a plot for CER by Field and Page, optimized for publication."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pio.templates.default = "simple_white"

    pages = list(per_page_metrics.keys())
    
    fig_cer_fields = go.Figure()
    
    styles = {
        'estonian_headword': {'color': '#666666', 'dash': 'dash'},
        'german_equivalent': {'color': 'black', 'dash': 'solid'}
    }
    default_style = {'color': 'black', 'dash': 'dot'}
    
    fields = sorted(columns_to_plot)
    
    for i, field in enumerate(fields):
        field_cer_values = []
        for page in pages:
            try:
                cer = per_page_metrics[page]['fields'][field].get('cer', float('nan'))
                if not math.isnan(cer) and cer > 2.0:
                    cer = 2.0
                field_cer_values.append(cer)
            except (KeyError, AttributeError):
                field_cer_values.append(float('nan'))

        field_style = styles.get(field, default_style)

        fig_cer_fields.add_trace(go.Scatter(
            x=pages,
            y=field_cer_values,
            mode='lines',
            name=field.replace('_', ' ').title(),
            line=dict(
                color=field_style['color'], 
                width=1.5,
                dash=field_style['dash']
            ),
            text=[f"{v:.3f}" if not math.isnan(v) else "N/A" for v in field_cer_values],
            hovertemplate="Page: %{x}<br>" + field + " CER: %{text}<extra></extra>"
        ))
    
    if pages:
        fig_cer_fields.add_shape(
            type="line",
            xref="paper",
            x0=0,
            y0=1.0,
            x1=1,
            y1=1.0,
            line=dict(
                color="black",
                width=1,
                dash="solid",
            )
        )
    
    fig_cer_fields.update_layout(
        title=dict(
            text="Character Error Rate (CER) by Field and Page",
            font=dict(size=18)
        ),
        xaxis_title=dict(
            text="Page Number",
            font=dict(size=14)
        ),
        yaxis_title=dict(
            text="Character Error Rate (CER)",
            font=dict(size=14)
        ),
        yaxis=dict(
            range=[0, 2.0],
            tickfont=dict(size=12)
        ),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=11),
            tickmode='array',
            tickvals=[pages[i] for i in range(0, len(pages), 10)],
            ticktext=[pages[i] for i in range(0, len(pages), 10)]
        ),
        legend=dict(
            title_text='',
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        margin=dict(l=60, r=30, t=60, b=60)
    )
    
    output_filename_pdf = os.path.join(output_dir, "cer_by_field_and_page.pdf")
    output_filename_png = os.path.join(output_dir, "cer_by_field_and_page.png")

    try:
        fig_cer_fields.write_image(output_filename_pdf)
        fig_cer_fields.write_image(output_filename_png, scale=3)
    except Exception as e:
        print(f"\nCould not save the plot as an image file. Please make sure you have the 'kaleido' package installed (`pip install kaleido`).")
        print(f"Error details: {e}\n")

    fig_cer_fields.show()


def load_and_prepare_data(file_path, delimiter):
    """Loads a CSV file and removes rows with missing or blank page numbers."""
    df = pd.read_csv(file_path, delimiter=delimiter, keep_default_na=False, dtype=str)
    df['page_number'] = df['page_number'].str.strip()
    df.replace({'page_number': {'': pd.NA}}, inplace=True)
    df.dropna(subset=['page_number'], inplace=True)
    return df


# --- Main Execution ---
if __name__ == "__main__":
    try:
        llm_df = load_and_prepare_data(LLM_OUTPUT_FILE, DELIMITER)
        gt_df = load_and_prepare_data(GROUND_TRUTH_FILE, DELIMITER)

        if KEY_COLUMN not in llm_df.columns or KEY_COLUMN not in gt_df.columns:
             raise ValueError(f"Key column '{KEY_COLUMN}' not found in one or both files.")
        if SECONDARY_KEY_COLUMN not in llm_df.columns or SECONDARY_KEY_COLUMN not in gt_df.columns:
             print(f"Warning: Secondary key column '{SECONDARY_KEY_COLUMN}' not found in one or both files. Fuzzy matching will rely solely on '{KEY_COLUMN}'.")
             secondary_key_present = False
        else:
            secondary_key_present = True


        cols_to_load = list(set(COLUMNS_TO_COMPARE + [KEY_COLUMN] + ([SECONDARY_KEY_COLUMN] if secondary_key_present else []) + ['page_number']))

        missing_llm_cols = [c for c in cols_to_load if c not in llm_df.columns]
        missing_gt_cols = [c for c in cols_to_load if c not in gt_df.columns]
        if missing_llm_cols: print(f"Warning: Columns not found in LLM file: {missing_llm_cols}")
        if missing_gt_cols: print(f"Warning: Columns not found in GT file: {missing_gt_cols}")

        valid_llm_cols = [c for c in cols_to_load if c in llm_df.columns]
        valid_gt_cols = [c for c in cols_to_load if c in gt_df.columns]
        final_cols_to_compare = [c for c in COLUMNS_TO_COMPARE if c in valid_llm_cols and c in valid_gt_cols]

        potential_synonym_col = 'estonian_synonyms'
        llm_cols_to_check = set(valid_llm_cols) | {KEY_COLUMN} | ({SECONDARY_KEY_COLUMN} if secondary_key_present and SECONDARY_KEY_COLUMN in valid_llm_cols else set())
        if potential_synonym_col in llm_df.columns:
            llm_cols_to_check.add(potential_synonym_col)

        cols_for_llm_load = list(llm_cols_to_check)
        cols_for_gt_load = list(set(valid_gt_cols) | {KEY_COLUMN} | ({SECONDARY_KEY_COLUMN} if secondary_key_present and SECONDARY_KEY_COLUMN in valid_gt_cols else set()))

        llm_df_filtered = llm_df[cols_for_llm_load]
        gt_df_filtered = gt_df[cols_for_gt_load]

        gt_secondary_key_col = 'german_equivalent'
        llm_secondary_key_col = 'german_equivalent'

        gt_secondary_present = gt_secondary_key_col in gt_df_filtered.columns
        llm_secondary_present = llm_secondary_key_col in llm_df_filtered.columns

        if not gt_secondary_present:
            print(f"Warning: Ground Truth secondary key '{gt_secondary_key_col}' not found. Secondary check disabled.")
        if not llm_secondary_present:
            print(f"Warning: LLM secondary key '{llm_secondary_key_col}' not found. Secondary check disabled.")

        use_secondary_in_alignment = gt_secondary_present and llm_secondary_present

        gt_grouped = gt_df_filtered.groupby('page_number', sort=False)
        llm_grouped = llm_df_filtered.groupby('page_number')
        
        all_aligned_pairs = []
        all_unaligned_llm_rows = []
        all_unaligned_gt_rows = []
        
        per_page_metrics = {}

        total_gt_pages = len(gt_grouped)
        
        for page_num, gt_page_df in tqdm(gt_grouped, total=total_gt_pages, desc="Processing pages"):
            per_page_metrics[page_num] = {
                'gt_entries': len(gt_page_df),
                'llm_entries': 0,
                'matched_entries': 0,
                'cer_sum': 0.0,
                'wer_sum': 0.0,
                'normalized_chars': 0,
                'normalized_words': 0,
                'fields': {col: {'cer_sum': 0.0, 'wer_sum': 0.0, 
                                 'normalized_chars': 0, 'normalized_words': 0} 
                           for col in final_cols_to_compare}
            }
            
            if page_num in llm_grouped.groups:
                llm_page_df = llm_grouped.get_group(page_num)
                per_page_metrics[page_num]['llm_entries'] = len(llm_page_df)
            else:
                all_unaligned_gt_rows.extend(gt_page_df.to_dict('records'))
                continue

            page_aligned_pairs, page_unaligned_llm, page_unaligned_gt = align_entries(
                llm_page_df,
                gt_page_df,
                KEY_COLUMN,
                llm_secondary_key_col if use_secondary_in_alignment else None,
                gt_secondary_key_col if use_secondary_in_alignment else None,
                HEADWORD_FUZZY_THRESHOLD,
                GERMAN_EQUIVALENT_FUZZY_THRESHOLD,
                SEQUENCE_WINDOW
            )
            
            per_page_metrics[page_num]['matched_entries'] = len(page_aligned_pairs)
            
            for llm_row, gt_row in page_aligned_pairs:
                for col in final_cols_to_compare:
                    llm_val = safe_str(llm_row.get(col, ''))
                    gt_val = safe_str(gt_row.get(col, ''))
                    
                    cer = calculate_cer(llm_val, gt_val)
                    
                    gt_val_norm = normalize_string(gt_val)
                    num_chars_norm = len(gt_val_norm)
                    
                    per_page_metrics[page_num]['cer_sum'] += cer * num_chars_norm
                    per_page_metrics[page_num]['normalized_chars'] += num_chars_norm
                    
                    field_metrics = per_page_metrics[page_num]['fields'][col]
                    field_metrics['cer_sum'] += cer * num_chars_norm
                    field_metrics['normalized_chars'] += num_chars_norm
            
            all_aligned_pairs.extend(page_aligned_pairs)
            all_unaligned_llm_rows.extend(page_unaligned_llm)
            all_unaligned_gt_rows.extend(page_unaligned_gt)

        llm_pages = set(llm_grouped.groups.keys())
        llm_only_pages = llm_pages - gt_pages
        for page_num in llm_only_pages:
            llm_page_df = llm_grouped.get_group(page_num)
            per_page_metrics[page_num] = {
                'gt_entries': 0,
                'llm_entries': len(llm_page_df),
                'matched_entries': 0,
                'cer_sum': 0.0,
                'wer_sum': 0.0,
                'normalized_chars': 0,
                'normalized_words': 0,
                'fields': {col: {'cer_sum': 0.0, 'wer_sum': 0.0, 
                                 'normalized_chars': 0, 'normalized_words': 0} 
                           for col in final_cols_to_compare}
            }
            all_unaligned_llm_rows.extend(llm_page_df.to_dict('records'))

        # Calculate final per-page metrics
        for page_num in per_page_metrics:
            page_data = per_page_metrics[page_num]
            
            for field in page_data['fields']:
                field_data = page_data['fields'][field]
                
                if field_data['normalized_chars'] > 0:
                    field_data['cer'] = field_data['cer_sum'] / field_data['normalized_chars']
                else:
                    field_data['cer'] = float('nan')

        create_cer_by_field_and_page_visualization(per_page_metrics, final_cols_to_compare, OUTPUT_DIR)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}. Please check file paths.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

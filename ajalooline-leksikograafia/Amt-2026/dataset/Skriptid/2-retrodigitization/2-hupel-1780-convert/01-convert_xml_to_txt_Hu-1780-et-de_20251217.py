#!/usr/bin/env python3
# Created: 2025-12-17 21-24-55
# Author: Madis Jürviste
"""
Convert Hupel 1780 XML dictionary fragments to TXT format with perfect line tracking.

This version (v12-2) enhances v12 with improved marker extraction:
1. Extract :ex: (explanation) markers for "m.", "k." suffixes
2. Extract :us: (usage) markers for "bl.", "selt.", etc.
3. Keep :gr: (grammar) markers on separate lines per GroundTruth examples
4. Implement numbered translations (:tr-1:, :tr-2:) when semicolons separate senses
5. Copy GroundTruth files without conversion
6. All improvements based on GroundTruth folder examples
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
from datetime import datetime
import re
import shutil


class XMLLineTracker:
    """Tracks line numbers for XML elements"""

    def __init__(self, xml_path: Path):
        self.xml_path = xml_path
        self.entry_lines = {}
        self._build_line_map()

    def _build_line_map(self):
        """Build a mapping of entry xml:id to line numbers - tracks ALL entries including nested ones"""
        try:
            with open(self.xml_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Use a stack to track entries at all nesting levels
            entry_stack = []  # Each element is (entry_id, start_line)

            for line_num, line in enumerate(lines, 1):
                # Check for opening <entry> tag
                entry_match = re.search(r'<entry[^>]+xml:id="([^"]+)"', line)
                if entry_match:
                    entry_id = entry_match.group(1)
                    # Push this entry onto the stack
                    entry_stack.append((entry_id, line_num))

                # Check for closing </entry> tag
                if '</entry>' in line:
                    # Pop from stack and record the line range
                    if entry_stack:
                        entry_id, start_line = entry_stack.pop()
                        self.entry_lines[entry_id] = (start_line, line_num)
        except Exception as e:
            pass

    def get_lines(self, entry_id: str) -> Tuple[int, int]:
        """Get line range for an entry"""
        return self.entry_lines.get(entry_id, (0, 0))

    @staticmethod
    def find_exact_line(xml_path: Path, search_content: str, line_start: int, line_end: int) -> int:
        """
        Search for exact line containing specific content within a range.
        Returns the exact line number, or 0 if not found.

        Searches for the content (like 'wasahovi') within the specified line range.
        Prioritizes content within <orth>, <quote>, or other element text over attributes.
        """
        try:
            with open(xml_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            best_match = 0

            # Search within the specified range
            for line_num in range(line_start - 1, min(line_end, len(lines))):
                line = lines[line_num]

                # Check if content appears in element text (between > and <)
                # This pattern matches: >content< which indicates element text, not attribute
                if f'>{search_content}<' in line:
                    return line_num + 1  # This is the best match - element text

                # If not found as element text, check if it's anywhere in the line
                if search_content in line:
                    if best_match == 0:  # Keep first match as fallback
                        best_match = line_num + 1

            return best_match  # Return fallback or 0 if not found
        except Exception as e:
            return 0


class ConversionLogger:
    """
    New approach: Track content with metadata, then assign exact line numbers
    after final output is generated.
    """

    def __init__(self, log_path: Path, output_filename: str):
        self.log_path = log_path
        self.output_filename = output_filename
        self.tracked_lines = []  # List of tracked line metadata
        self.current_entry = None

    def start_entry(self, entry_id: str, xml_file: str, line_start: int, line_end: int):
        """Start tracking a new entry"""
        self.current_entry = {
            'entry_id': entry_id,
            'xml_file': xml_file,
            'line_start': line_start,
            'line_end': line_end
        }

    def track_output_line(self, content: str, marker: str, source_desc: str, xml_lines: str, search_key: str = None):
        """
        Track a line of output with its source metadata.

        search_key: A string to search for in the XML to find the exact line.
                   For example, for "* wasahovi", search_key would be "wasahovi".
                   If None, will try to extract from content.
        """
        # If no search_key provided, try to extract meaningful content from the line
        if search_key is None:
            # Remove marker prefix to get actual content
            search_key = content.replace(marker, '').strip() if marker else content

        self.tracked_lines.append({
            'content': content,
            'entry_id': self.current_entry['entry_id'] if self.current_entry else None,
            'xml_file': self.current_entry['xml_file'] if self.current_entry else None,
            'line_start': self.current_entry['line_start'] if self.current_entry else 0,
            'line_end': self.current_entry['line_end'] if self.current_entry else 0,
            'marker': marker,
            'source_desc': source_desc,
            'xml_lines': xml_lines,
            'search_key': search_key
        })

    def finalize_and_write_log(self, final_output: str, xml_file: str):
        """
        Map tracked content to exact line numbers in final output and write log.
        This is called AFTER the complete output is generated.
        """
        # Split final output into lines
        output_lines = final_output.split('\n')

        # Build a mapping: content -> list of line numbers where it appears
        content_to_lines = defaultdict(list)
        for line_num, line in enumerate(output_lines, 1):
            content_to_lines[line].append(line_num)

        # Assign exact line numbers to tracked metadata
        # We process in order and consume line numbers as we match
        content_usage_count = defaultdict(int)

        # Get base directory for resolving fragment paths
        base_dir = Path(__file__).parent / "20251029-raw_xml_fragments"

        for tracked in self.tracked_lines:
            content = tracked['content']
            if content in content_to_lines:
                # Get the next unused line number for this content
                usage_idx = content_usage_count[content]
                if usage_idx < len(content_to_lines[content]):
                    tracked['output_line'] = content_to_lines[content][usage_idx]
                    content_usage_count[content] += 1
                else:
                    tracked['output_line'] = -1  # Error: more tracked than actual
            else:
                tracked['output_line'] = -1  # Error: content not found

            # V12 ENHANCEMENT: Find exact XML line for this element
            # Use the search_key to find the exact line in the XML
            if tracked['xml_file'] and tracked['search_key'] and tracked['line_start'] > 0:
                xml_path = base_dir / tracked['xml_file']
                exact_line = XMLLineTracker.find_exact_line(
                    xml_path,
                    tracked['search_key'],
                    tracked['line_start'],
                    tracked['line_end']
                )
                if exact_line > 0:
                    # Update xml_lines to show exact line instead of range
                    tracked['xml_lines_exact'] = f"XML line {exact_line}"
                else:
                    # Fall back to range if exact line not found
                    tracked['xml_lines_exact'] = tracked['xml_lines']
            else:
                # No search possible, use original range
                tracked['xml_lines_exact'] = tracked['xml_lines']

        # Group by entry for log output
        entries_log = defaultdict(list)
        for tracked in self.tracked_lines:
            if tracked['entry_id']:
                entries_log[tracked['entry_id']].append(tracked)

        # Write log file
        with open(self.log_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write(f"CONVERSION LOG: {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\n")
            f.write(f"SOURCE: {xml_file} -> OUTPUT: {self.output_filename}\n")
            f.write("=" * 80 + "\n\n")

            # Entries
            for tracked in self.tracked_lines:
                entry_id = tracked['entry_id']
                if not entry_id:
                    continue

                # Check if this is the first line of this entry
                entry_lines = entries_log[entry_id]
                if entry_lines and tracked == entry_lines[0]:
                    # Write entry header
                    f.write("-" * 80 + "\n")
                    f.write(f"ENTRY: {entry_id} (xml:id=\"{entry_id}\")\n")
                    f.write(f"  SOURCE LINE {tracked['line_start']}-{tracked['line_end']} "
                           f"in {tracked['xml_file']}\n\n")

                    # Write all mappings for this entry
                    for mapping in entry_lines:
                        content = mapping['content']
                        output_line = mapping['output_line']

                        # Format the main line with alignment (50 chars wide)
                        # Content already contains the marker (e.g., ":tr: die Segel")
                        main_line = f"  {content}"
                        if len(main_line) < 50:
                            padding = 50 - len(main_line)
                            main_line += " " * padding

                        f.write(f"{main_line}→ OUTPUT line {output_line}\n")

                        # Format the FROM line with XML details (use exact line if available)
                        from_line = f"    FROM: {mapping['source_desc']}"
                        if mapping.get('xml_lines_exact'):
                            from_line += f" ({mapping['xml_lines_exact']})"
                        elif mapping['xml_lines']:
                            from_line += f" ({mapping['xml_lines']})"
                        f.write(f"{from_line}\n\n")

                    f.write("\n")


class FragmentConverter:
    """Converts XML fragments to TXT using the same logic as the main converter"""

    def __init__(self):
        pass

    def get_text_content(self, element, strip: bool = True) -> str:
        """Extract text content from element"""
        if element is None:
            return ""
        text = element.text or ""
        if strip:
            text = text.strip()
        return text

    def get_all_text(self, element) -> str:
        """Get all text content including nested elements"""
        if element is None:
            return ""
        return ''.join(element.itertext()).strip()

    def is_dialect_marker(self, corresp: str) -> bool:
        """
        Check if geographic marker is a dialect (vs regional name/place)
        V12-2 UPDATE: Check if corresp contains '_dialect' to distinguish from
        regional references which contain '_region', '_manor', etc.
        """
        if not corresp:
            return False
        corresp_clean = corresp.lstrip('#')
        # If corresp contains '_dialect', it's a dialect marker
        # Otherwise (e.g., '_region', '_manor', '_church'), it's a regional name
        return '_dialect' in corresp_clean

    def extract_inflected_forms(self, entry_elem) -> List[str]:
        """Extract all inflected form suffixes from entry, in order"""
        forms = []
        for form_elem in entry_elem.findall('.//form[@type="inflected"]'):
            orth_elem = form_elem.find('orth[@extent="suffix"]')
            if orth_elem is not None and orth_elem.text:
                forms.append(orth_elem.text.strip())
        return forms

    def extract_translations(self, sense_elem) -> List[str]:
        """Extract translations from a sense element"""
        translations = []
        for cit_elem in sense_elem.findall('cit[@type="translationEquivalent"]'):
            quote_elem = cit_elem.find('quote')
            if quote_elem is not None:
                text = self.get_all_text(quote_elem)
                if text:
                    translations.append(text)
        return translations

    def extract_geographic_markers(self, element) -> Tuple[List[str], List[str]]:
        """
        Extract geographic markers, separated into dialects and regional names
        Returns: (dialect_markers, regional_markers)

        V12-2 UPDATE: Search at all levels (including within sense elements)
        """
        dialects = []
        regionals = []

        # V12-2: Use .// to search at all levels, not just direct children
        for usg_elem in element.findall('.//usg[@type="geographic"]'):
            text = self.get_text_content(usg_elem)
            corresp = usg_elem.get('corresp', '')

            if not text:
                continue

            if self.is_dialect_marker(corresp):
                dialects.append(text)
            else:
                regionals.append(text)

        return dialects, regionals

    def process_entry(self, entry_elem, logger: ConversionLogger,
                     xml_file: str, line_start: int, line_end: int) -> List[str]:
        """Process a single entry element and return lines of text output"""
        output_lines = []
        entry_id = entry_elem.get('{http://www.w3.org/XML/1998/namespace}id', 'unknown')

        logger.start_entry(entry_id, xml_file, line_start, line_end)

        # Entry ID line
        entry_id_line = f"<entry xml:id=\"{entry_id}\">"
        output_lines.append(entry_id_line)
        logger.track_output_line(entry_id_line, '<entry xml:id="...">',
                                'entry element xml:id attribute', f'XML line {line_start}-{line_end}')

        # Extract headword (lemma)
        lemma_elem = entry_elem.find('form[@type="lemma"]/orth')
        headword = None
        if lemma_elem is not None:
            headword = self.get_text_content(lemma_elem)
            if headword:
                line = f"* {headword}"
                output_lines.append(line)
                logger.track_output_line(line, '*',
                                       '<form type="lemma"><orth>...</orth></form>',
                                       f'XML line {line_start}-{line_end}')

        # V12-2: Extract explanation markers (m., k., etc.) after lemma
        # Look for <metamark function="inflectionDelimiter">m.</metamark> or k.
        for child in entry_elem:
            if child.tag == 'metamark' or child.tag.endswith('}metamark'):
                if child.get('function') == 'inflectionDelimiter':
                    metamark_text = self.get_text_content(child)
                    if metamark_text in ['m.', 'k.']:
                        line = f":ex: {metamark_text}"
                        output_lines.append(line)
                        logger.track_output_line(line, ':ex:',
                                               '<metamark function="inflectionDelimiter">',
                                               f'XML line {line_start}-{line_end}', metamark_text)
                        break  # Only process first explanation marker after lemma

        # If no lemma, check for compound/phrase as main form
        if not headword:
            compound_elem = entry_elem.find('form[@type="compound"]/orth')
            if compound_elem is not None:
                headword = self.get_text_content(compound_elem)
                if headword:
                    line = f"* {headword}"
                    output_lines.append(line)
                    logger.track_output_line(line, '*',
                                           '<form type="compound"><orth>...</orth></form> (main form)',
                                           f'XML line {line_start}-{line_end}')

        # Process variants in document order
        for child in entry_elem:
            tag = child.tag
            if tag == 'form' or tag.endswith('}form'):
                form_type = child.get('type')
                if form_type == 'variant':
                    orth_elem = child.find('orth')
                    if orth_elem is not None:
                        variant = self.get_text_content(orth_elem)
                        line = f"~ {variant}"
                        output_lines.append(line)
                        logger.track_output_line(line, '~',
                                               '<form type="variant"><orth>...</orth></form>',
                                               f'XML line {line_start}-{line_end}')

                        # Check for inline dialect markers
                        child_dialects, child_regionals = self.extract_geographic_markers(child)
                        if child_dialects:
                            dialect_str = ' '.join(child_dialects)
                            line = f":di: {dialect_str}"
                            output_lines.append(line)
                            logger.track_output_line(line, ':di:',
                                                   'inline <usg type="geographic"> in variant',
                                                   f'XML line {line_start}-{line_end}')

        # Collect all inflected forms and combine on one line
        inflected_forms = self.extract_inflected_forms(entry_elem)
        if inflected_forms:
            gr_content = ', '.join(inflected_forms)
            line = f":gr: {gr_content}"
            output_lines.append(line)
            logger.track_output_line(line, ':gr:',
                                   f'{len(inflected_forms)} <form type="inflected"> elements',
                                   f'XML line {line_start}-{line_end}')

        # V12-2: Process senses with numbered translations when semicolons separate them
        # First, check if there are semicolon-delimited senses
        sense_groups = []
        current_group = []

        for child in entry_elem:
            if child.tag == 'sense' or child.tag.endswith('}sense'):
                current_group.append(child)
            elif child.tag == 'metamark' or child.tag.endswith('}metamark'):
                if child.get('function') == 'senseDelimiter' and child.text == ';':
                    # Semicolon found, start new group
                    if current_group:
                        sense_groups.append(current_group)
                        current_group = []

        # Add last group
        if current_group:
            sense_groups.append(current_group)

        # If we have multiple groups (semicolon-separated), use numbered translations
        use_numbered_translations = len(sense_groups) > 1

        # Process each sense group
        for group_idx, sense_group in enumerate(sense_groups, 1):
            for sense_elem in sense_group:
                # Check for cross-reference
                xr_elem = sense_elem.find('xr[@type="related"]/ref')
                if xr_elem is not None:
                    xr_text = self.get_all_text(xr_elem)
                    line = f":xr: {xr_text}"
                    output_lines.append(line)
                    logger.track_output_line(line, ':xr:',
                                           '<xr type="related"><ref>...</ref></xr>',
                                           f'XML line {line_start}-{line_end}')
                    continue

                # Extract translations
                translations = self.extract_translations(sense_elem)
                for tr in translations:
                    if use_numbered_translations:
                        line = f":tr-{group_idx}: {tr}"
                        marker = f':tr-{group_idx}:'
                    else:
                        line = f":tr: {tr}"
                        marker = ':tr:'
                    output_lines.append(line)
                    sense_id = sense_elem.get('{http://www.w3.org/XML/1998/namespace}id', '')
                    logger.track_output_line(line, marker,
                                           '<cit type="translationEquivalent"><quote>...</quote></cit>',
                                           f'sense {sense_id}, XML line {line_start}-{line_end}', tr)

                # V12-2: Extract geographic markers within this sense (right after translations)
                sense_dialects, sense_regionals = self.extract_geographic_markers(sense_elem)
                if sense_dialects:
                    dialect_str = ' '.join(sense_dialects)
                    line = f":di: {dialect_str}"
                    output_lines.append(line)
                    sense_id = sense_elem.get('{http://www.w3.org/XML/1998/namespace}id', '')
                    logger.track_output_line(line, ':di:',
                                           f'<usg type="geographic"> in sense {sense_id}',
                                           f'XML line {line_start}-{line_end}', dialect_str)

                for regional in sense_regionals:
                    line = f":rn: {regional}"
                    output_lines.append(line)
                    sense_id = sense_elem.get('{http://www.w3.org/XML/1998/namespace}id', '')
                    logger.track_output_line(line, ':rn:',
                                           f'<usg type="geographic"> in sense {sense_id}',
                                           f'XML line {line_start}-{line_end}', regional)

        # Geographic markers at entry level (not within senses)
        # Extract only markers that are direct children of entry, not within senses
        dialects, regionals = [], []
        for usg_elem in entry_elem.findall('usg[@type="geographic"]'):
            text = self.get_text_content(usg_elem)
            corresp = usg_elem.get('corresp', '')
            if not text:
                continue
            if self.is_dialect_marker(corresp):
                dialects.append(text)
            else:
                regionals.append(text)

        if dialects:
            dialect_str = ' '.join(dialects)
            line = f":di: {dialect_str}"
            output_lines.append(line)
            logger.track_output_line(line, ':di:',
                                   '<usg type="geographic"> at entry level',
                                   f'XML line {line_start}-{line_end}', dialect_str)

        for regional in regionals:
            line = f":rn: {regional}"
            output_lines.append(line)
            logger.track_output_line(line, ':rn:',
                                   '<usg type="geographic"> at entry level',
                                   f'XML line {line_start}-{line_end}', regional)

        # V12-2: Extract usage markers (bl., selt., etc.) and other usg types
        for usg_elem in entry_elem.findall('usg'):
            usg_type = usg_elem.get('type', '')
            usg_text = self.get_text_content(usg_elem)
            if not usg_text:
                continue

            # Look for frequency, register, hint, domain, textType, pos, number types
            if usg_type in ['frequency', 'register', 'hint', 'domain', 'textType', 'pos', 'number']:
                line = f":us: {usg_text}"
                output_lines.append(line)
                logger.track_output_line(line, ':us:',
                                       f'<usg type="{usg_type}">',
                                       f'XML line {line_start}-{line_end}', usg_text)

        # V12-2: Extract notes as explanations (search at all levels)
        for note_elem in entry_elem.findall('.//note'):
            note_text = self.get_text_content(note_elem)
            if note_text:
                line = f":ex: {note_text}"
                output_lines.append(line)
                logger.track_output_line(line, ':ex:',
                                       '<note>',
                                       f'XML line {line_start}-{line_end}', note_text)

        # Process nested entries - show as :mw: content under parent
        for nested_entry in entry_elem.findall('entry'):
            phrase_elem = nested_entry.find('form[@type="phrase"]/orth')
            compound_elem = nested_entry.find('form[@type="compound"]/orth')

            if phrase_elem is not None:
                phrase = self.get_text_content(phrase_elem)
                if ' ' in phrase:  # multiword
                    line = f":mw: {phrase}"
                    output_lines.append(line)
                    logger.track_output_line(line, ':mw:',
                                           'nested <entry> with <form type="phrase">',
                                           f'nested entry, XML line {line_start}-{line_end}')

                    # Get translation
                    nested_sense = nested_entry.find('sense')
                    if nested_sense is not None:
                        nested_tr = self.extract_translations(nested_sense)
                        for tr in nested_tr:
                            line = f":mw/tr: {tr}"
                            output_lines.append(line)
                            logger.track_output_line(line, ':mw/tr:',
                                                   'translation in nested phrase entry',
                                                   f'nested entry, XML line {line_start}-{line_end}')
                else:  # single word sub-entry
                    line = f":se: {phrase}"
                    output_lines.append(line)
                    logger.track_output_line(line, ':se:',
                                           'nested <entry> with single-word phrase',
                                           f'nested entry, XML line {line_start}-{line_end}')

            elif compound_elem is not None:
                compound = self.get_text_content(compound_elem)
                if ' ' in compound:  # multiword
                    line = f":mw: {compound}"
                    output_lines.append(line)
                    logger.track_output_line(line, ':mw:',
                                           'nested <entry> with <form type="compound">',
                                           f'nested entry, XML line {line_start}-{line_end}')

                    # Get translation
                    nested_sense = nested_entry.find('sense')
                    if nested_sense is not None:
                        nested_tr = self.extract_translations(nested_sense)
                        for tr in nested_tr:
                            line = f":mw/tr: {tr}"
                            output_lines.append(line)
                            logger.track_output_line(line, ':mw/tr:',
                                                   'translation in nested compound entry',
                                                   f'nested entry, XML line {line_start}-{line_end}')

        return output_lines

    def convert_fragment(self, xml_path: Path, logger: ConversionLogger) -> Tuple[str, int, int]:
        """
        Convert a single XML fragment to TXT
        Returns: (txt_content, page_number, column_number)
        """
        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Create line tracker for this fragment
        line_tracker = XMLLineTracker(xml_path)

        # Extract page number and column from filename (e.g., "135_column1_chunk1..." -> page 135, column 1)
        match = re.match(r'(\d+)_column(\d+)_', xml_path.name)
        if match:
            page_num = int(match.group(1))
            column_num = int(match.group(2))
        else:
            # Fallback: try to find in XML
            page_num = None
            column_num = 1  # Default to column 1
            for elem in root.iter():
                if elem.tag.endswith('div') and elem.get('type') == 'page':
                    page_num = int(elem.get('n', 0))
                    break
            if page_num is None:
                raise ValueError(f"Could not extract page number from {xml_path.name}")

        output_lines = []

        # Process all entries in the fragment
        for entry_elem in root.findall('.//entry'):
            entry_id = entry_elem.get('{http://www.w3.org/XML/1998/namespace}id', 'unknown')
            line_start, line_end = line_tracker.get_lines(entry_id)

            entry_lines = self.process_entry(entry_elem, logger, xml_path.name,
                                           line_start, line_end)
            output_lines.extend(entry_lines)
            output_lines.append("")  # Blank line between entries

        txt_content = '\n'.join(output_lines)
        return txt_content, page_num, column_num


def main():
    """Main entry point"""
    # Setup paths
    base_dir = Path(__file__).parent
    source_dir = base_dir / "20251029-raw_xml_fragments"
    output_dir = base_dir / "converted_txt_files_v12-2"
    log_dir = base_dir / "conversion_logs_v12-2"
    groundtruth_dir = base_dir / "GroundTruth"

    # Create output directories
    output_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    # V12-2: Get today's date for output files
    today = datetime.now().strftime('%Y%m%d')

    # V12-2: Copy GroundTruth files to output without conversion
    groundtruth_pages = set()
    if groundtruth_dir.exists():
        print("Copying GroundTruth files...")
        for gt_file in groundtruth_dir.glob("*.txt"):
            # Extract page number from filename (e.g., "135_Hu-1780_20251108_edt.txt")
            match = re.match(r'(\d+)_', gt_file.name)
            if match:
                page_num = int(match.group(1))
                groundtruth_pages.add(page_num)

                # Copy with updated date
                new_filename = f"{page_num}_Hu-1780_{today}.txt"
                dest_path = output_dir / new_filename
                shutil.copy2(gt_file, dest_path)
                print(f"  Copied: {gt_file.name} → {new_filename}")
        print()

    # Find all XML fragment files
    xml_files = sorted(source_dir.glob('*.xml'))

    # Filter out any non-fragment files (like the converter script itself if it were XML)
    xml_files = [f for f in xml_files if re.match(r'\d+_column\d+_chunk\d+', f.name)]

    if not xml_files:
        print(f"No XML fragment files found in {source_dir}")
        return

    print(f"Found {len(xml_files)} XML fragment files")
    print(f"Output directory: {output_dir}")
    print(f"Log directory: {log_dir}")
    print()

    # Initialize converter
    converter = FragmentConverter()

    # Group fragments by page number
    fragments_by_page: Dict[int, List[Path]] = defaultdict(list)
    for xml_path in xml_files:
        match = re.match(r'(\d+)_', xml_path.name)
        if match:
            page_num = int(match.group(1))
            fragments_by_page[page_num].append(xml_path)

    print(f"Found {len(fragments_by_page)} unique pages")
    print()

    # Sort function for fragments: by column, then by chunk
    def sort_fragments(path: Path) -> Tuple[int, int]:
        match = re.match(r'\d+_column(\d+)_chunk(\d+)_', path.name)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return (0, 0)

    # Process each page
    for page_num in sorted(fragments_by_page.keys()):
        # V12-2: Skip pages that are in GroundTruth
        if page_num in groundtruth_pages:
            print(f"Skipping page {page_num} (in GroundTruth)")
            continue

        fragments = sorted(fragments_by_page[page_num], key=sort_fragments)
        print(f"Processing page {page_num} ({len(fragments)} fragments)...", end=' ')

        # Generate output filename
        output_filename = f"{page_num}_Hu-1780_{today}.txt"
        output_path = output_dir / output_filename

        # Setup logging for this page
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
        log_filename = f"{page_num}_Hu-1780_{today}_{timestamp}_log.txt"
        log_path = log_dir / log_filename
        logger = ConversionLogger(log_path, output_filename)

        # Convert all fragments for this page, tracking column breaks
        page_txt_parts = []
        last_column = None

        for fragment_path in fragments:
            try:
                # Add column break comment if switching columns
                txt_content, _, column_num = converter.convert_fragment(fragment_path, logger)
                if txt_content.strip():  # Only add non-empty content
                    if last_column is not None and column_num != last_column:
                        page_txt_parts.append(f"--- column break ---")

                    page_txt_parts.append(txt_content)
                    last_column = column_num
            except Exception as e:
                print(f"\n  Warning: Error processing {fragment_path.name}: {e}")
                continue

        # Concatenate all fragments for this page
        if page_txt_parts:
            # Join all fragments with blank lines between them
            full_page_content = '\n\n'.join(page_txt_parts)

            # Add page header
            page_header = f"--- page {page_num} ---\n"
            full_page_content = page_header + full_page_content

            # Write output file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_page_content)

            # NOW assign exact line numbers and write log
            logger.finalize_and_write_log(full_page_content, f"page {page_num} (all fragments)")

            print(f"✓ → {output_filename}")
        else:
            print("✗ No content")

    print()
    print("Conversion complete!")
    print(f"Output files: {output_dir}")
    print(f"Log files: {log_dir}")


if __name__ == "__main__":
    main()

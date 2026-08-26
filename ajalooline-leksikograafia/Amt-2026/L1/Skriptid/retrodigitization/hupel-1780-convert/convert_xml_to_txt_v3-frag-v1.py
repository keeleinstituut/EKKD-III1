#!/usr/bin/env python3
# Created: 2025-11-08 14-07-32
# Author: Madis Jürviste
"""
XML Fragment to TXT Converter for Hupel 1780 Estonian-German Dictionary
Version: v3-frag-v1

Converts XML fragments to simplified text format
- Handles multiple fragments per page
- Avoids exact duplicates based on xml:id
- Groups output by page number
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
import re
from collections import defaultdict


class XMLLineTracker:
    """Tracks line numbers for XML elements in fragments"""

    def __init__(self, xml_path: Path):
        self.xml_path = xml_path
        self.entry_lines = {}
        self._build_line_map()

    def _build_line_map(self):
        """Build a mapping of entry xml:id to line numbers"""
        with open(self.xml_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_entry_id = None
        entry_start = None
        entry_depth = 0

        for line_num, line in enumerate(lines, 1):
            entry_match = re.search(r'<entry[^>]+xml:id="([^"]+)"', line)
            if entry_match:
                entry_id = entry_match.group(1)
                if entry_depth == 0:
                    current_entry_id = entry_id
                    entry_start = line_num
                entry_depth += 1

            if '</entry>' in line:
                entry_depth -= 1
                if entry_depth == 0 and current_entry_id and entry_start:
                    self.entry_lines[current_entry_id] = (entry_start, line_num)
                    current_entry_id = None
                    entry_start = None

    def get_lines(self, entry_id: str) -> Tuple[int, int]:
        """Get line range for an entry"""
        return self.entry_lines.get(entry_id, (0, 0))


class ConversionLogger:
    """Handles detailed logging of conversion mappings"""

    def __init__(self, log_path: Path, output_filename: str):
        self.log_path = log_path
        self.output_filename = output_filename
        self.entries = []
        self.current_entry = None
        self.current_line_number = 1

    def start_entry(self, entry_id: str, xml_file: str, line_start: int, line_end: int):
        """Start logging a new entry"""
        self.current_entry = {
            'entry_id': entry_id,
            'xml_file': xml_file,
            'line_start': line_start,
            'line_end': line_end,
            'mappings': []
        }

    def add_mapping(self, marker: str, content: str, source_desc: str, xml_lines: str):
        """Add a mapping for current entry"""
        if self.current_entry:
            output_line = self.current_line_number
            self.current_line_number += 1

            self.current_entry['mappings'].append({
                'marker': marker,
                'content': content,
                'source_desc': source_desc,
                'xml_lines': xml_lines,
                'output_line': output_line
            })

    def end_entry(self):
        """Finish logging current entry"""
        if self.current_entry:
            self.entries.append(self.current_entry)
            self.current_entry = None
            self.current_line_number += 1

    def write_log(self, page_num: str):
        """Write complete log file"""
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"CONVERSION LOG: {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\n")
            f.write(f"SOURCE: Page {page_num} fragments -> OUTPUT: {self.output_filename}\n")
            f.write("=" * 80 + "\n\n")

            for entry_data in self.entries:
                f.write("-" * 80 + "\n")
                f.write(f"ENTRY: {entry_data['entry_id']} (xml:id=\"{entry_data['entry_id']}\")\n")
                f.write(f"  SOURCE LINE {entry_data['line_start']}-{entry_data['line_end']} "
                       f"in {entry_data['xml_file']}\n\n")

                for mapping in entry_data['mappings']:
                    marker = mapping['marker']
                    content = mapping['content']
                    output_line = mapping['output_line']

                    main_line = f"  {marker} {content}"
                    if len(main_line) < 50:
                        padding = 50 - len(main_line)
                        main_line += " " * padding

                    f.write(f"{main_line}→ OUTPUT line {output_line}\n")

                    from_line = f"    FROM: {mapping['source_desc']}"
                    if mapping['xml_lines']:
                        from_line += f" ({mapping['xml_lines']})"
                    f.write(f"{from_line}\n\n")

                f.write("\n")


class XMLFragmentConverter:
    """Main converter class for XML fragments"""

    DIALECT_IDS = {
        'reval_dialect', 'dorpat_dialect', 'harjumaa_dialect',
        'pärnu_dialect', 'ösel_dialect', 'wiek_dialect', 'wirumaa_dialect'
    }

    def __init__(self, source_dir: Path, output_dir: Path, log_dir: Path):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.log_dir = log_dir

        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

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
        """Check if geographic marker is a dialect"""
        if not corresp:
            return False
        corresp_clean = corresp.lstrip('#')
        return corresp_clean in self.DIALECT_IDS

    def extract_inflected_forms(self, entry_elem) -> List[str]:
        """Extract all inflected form suffixes"""
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
        """Extract geographic markers (dialects, regional names)"""
        dialects = []
        regionals = []

        for usg_elem in element.findall('usg[@type="geographic"]'):
            text = self.get_text_content(usg_elem)
            corresp = usg_elem.get('corresp', '')

            if self.is_dialect_marker(corresp):
                dialects.append(text)
            else:
                regionals.append(text)

        return dialects, regionals

    def process_entry(self, entry_elem, logger: ConversionLogger,
                     xml_file: str, line_start: int, line_end: int) -> List[str]:
        """Process a single entry element"""
        output_lines = []
        entry_id = entry_elem.get('{http://www.w3.org/XML/1998/namespace}id', 'unknown')

        logger.start_entry(entry_id, xml_file, line_start, line_end)

        # Entry ID line
        output_lines.append(f"<entry xml:id=\"{entry_id}\">")
        logger.add_mapping('<entry xml:id="...">', entry_id,
                          'entry element xml:id attribute', f'XML line {line_start}')

        # Extract headword
        lemma_elem = entry_elem.find('form[@type="lemma"]/orth')
        headword = None
        if lemma_elem is not None:
            headword = self.get_text_content(lemma_elem)
            if headword:
                output_lines.append(f"* {headword}")
                logger.add_mapping('*', headword,
                                 '<form type="lemma"><orth>...</orth></form>',
                                 f'XML line {line_start}')

        # If no lemma, check for compound/phrase
        if not headword:
            compound_elem = entry_elem.find('form[@type="compound"]/orth')
            if compound_elem is not None:
                headword = self.get_text_content(compound_elem)
                if headword:
                    output_lines.append(f"* {headword}")
                    logger.add_mapping('*', headword,
                                     '<form type="compound"><orth>...</orth></form>',
                                     f'XML line {line_start}')

        # Process variants
        for child in entry_elem:
            tag = child.tag
            if tag == 'form' or tag.endswith('}form'):
                form_type = child.get('type')
                if form_type == 'variant':
                    orth_elem = child.find('orth')
                    if orth_elem is not None:
                        variant = self.get_text_content(orth_elem)
                        output_lines.append(f"~ {variant}")
                        logger.add_mapping('~', variant,
                                         '<form type="variant"><orth>...</orth></form>',
                                         f'XML line {line_start}-{line_end}')

                        child_dialects, child_regionals = self.extract_geographic_markers(child)
                        if child_dialects:
                            dialect_str = ' '.join(child_dialects)
                            output_lines.append(f":di: {dialect_str}")
                            logger.add_mapping(':di:', dialect_str,
                                             'inline <usg type="geographic"> in variant',
                                             f'XML line {line_start}-{line_end}')

        # Inflected forms
        inflected_forms = self.extract_inflected_forms(entry_elem)
        if inflected_forms:
            gr_content = ', '.join(inflected_forms)
            output_lines.append(f":gr: {gr_content}")
            logger.add_mapping(':gr:', gr_content,
                             f'{len(inflected_forms)} <form type="inflected"> elements',
                             f'XML line {line_start}-{line_end}')

        # Process senses
        for sense_elem in entry_elem.findall('sense'):
            # Cross-reference
            xr_elem = sense_elem.find('xr[@type="related"]/ref')
            if xr_elem is not None:
                xr_text = self.get_all_text(xr_elem)
                output_lines.append(f":xr: {xr_text}")
                logger.add_mapping(':xr:', xr_text,
                                 '<xr type="related"><ref>...</ref></xr>',
                                 f'XML line {line_start}-{line_end}')
                continue

            # Translations
            translations = self.extract_translations(sense_elem)
            for tr in translations:
                output_lines.append(f":tr: {tr}")
                sense_id = sense_elem.get('{http://www.w3.org/XML/1998/namespace}id', '')
                logger.add_mapping(':tr:', tr,
                                 '<cit type="translationEquivalent"><quote>...</quote></cit>',
                                 f'sense {sense_id}, XML line {line_start}-{line_end}')

        # Geographic markers
        dialects, regionals = self.extract_geographic_markers(entry_elem)
        if dialects:
            dialect_str = ' '.join(dialects)
            output_lines.append(f":di: {dialect_str}")
            logger.add_mapping(':di:', dialect_str,
                             '<usg type="geographic"> dialect markers',
                             f'XML line {line_start}-{line_end}')

        for regional in regionals:
            output_lines.append(f":rn: {regional}")
            logger.add_mapping(':rn:', regional,
                             '<usg type="geographic"> regional marker',
                             f'XML line {line_start}-{line_end}')

        # Nested entries
        for nested_entry in entry_elem.findall('entry'):
            phrase_elem = nested_entry.find('form[@type="phrase"]/orth')
            compound_elem = nested_entry.find('form[@type="compound"]/orth')

            if phrase_elem is not None:
                phrase = self.get_text_content(phrase_elem)
                if ' ' in phrase:
                    output_lines.append(f":mw: {phrase}")
                    logger.add_mapping(':mw:', phrase,
                                     'nested <entry> with <form type="phrase">',
                                     f'nested, XML line {line_start}-{line_end}')

                    nested_sense = nested_entry.find('sense')
                    if nested_sense is not None:
                        nested_tr = self.extract_translations(nested_sense)
                        for tr in nested_tr:
                            output_lines.append(f":mw/tr: {tr}")
                            logger.add_mapping(':mw/tr:', tr,
                                             'translation in nested phrase entry',
                                             f'nested, XML line {line_start}-{line_end}')
                else:
                    output_lines.append(f":se: {phrase}")
                    logger.add_mapping(':se:', phrase,
                                     'nested <entry> with single-word phrase',
                                     f'nested, XML line {line_start}-{line_end}')

            elif compound_elem is not None:
                compound = self.get_text_content(compound_elem)
                if ' ' in compound:
                    output_lines.append(f":mw: {compound}")
                    logger.add_mapping(':mw:', compound,
                                     'nested <entry> with <form type="compound">',
                                     f'nested, XML line {line_start}-{line_end}')

                    nested_sense = nested_entry.find('sense')
                    if nested_sense is not None:
                        nested_tr = self.extract_translations(nested_sense)
                        for tr in nested_tr:
                            output_lines.append(f":mw/tr: {tr}")
                            logger.add_mapping(':mw/tr:', tr,
                                             'translation in nested compound entry',
                                             f'nested, XML line {line_start}-{line_end}')

        logger.end_entry()
        return output_lines

    def process_page_fragments(self, page_num: str, fragment_files: List[Path]) -> Tuple[str, str, int]:
        """Process all fragments for a single page"""
        # Track seen entries to avoid duplicates
        seen_entries: Set[str] = set()
        all_entries = []

        # Collect all unique entries from all fragments
        for frag_file in fragment_files:
            line_tracker = XMLLineTracker(frag_file)

            try:
                tree = ET.parse(frag_file)
                root = tree.getroot()

                # Find page div or use root
                page_div = None
                for elem in root.iter():
                    if elem.tag.endswith('div'):
                        page_div = elem
                        break

                # Find only top-level entries (not nested ones)
                if page_div is not None:
                    entries_to_process = page_div.findall('entry')  # Direct children only
                else:
                    entries_to_process = root.findall('entry')  # Direct children only

                for entry_elem in entries_to_process:
                    entry_id = entry_elem.get('{http://www.w3.org/XML/1998/namespace}id', 'unknown')

                    # Skip if we've already seen this entry
                    if entry_id in seen_entries:
                        continue

                    seen_entries.add(entry_id)
                    line_start, line_end = line_tracker.get_lines(entry_id)

                    all_entries.append({
                        'element': entry_elem,
                        'entry_id': entry_id,
                        'fragment_file': frag_file.name,
                        'line_start': line_start,
                        'line_end': line_end
                    })

            except Exception as e:
                print(f"  Warning: Error parsing {frag_file.name}: {e}")
                continue

        # Generate output filename
        today = datetime.now().strftime('%Y%m%d')
        output_filename = f"{page_num}_Hu-1780_{today}_frag-v1.txt"

        # Setup logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        log_filename = f"{timestamp}_page{page_num}_log.txt"
        log_path = self.log_dir / log_filename
        logger = ConversionLogger(log_path, output_filename)

        output_lines = []

        # Page marker
        output_lines.append(f"--- page {page_num} ---")
        output_lines.append("")

        logger.current_line_number = 3

        # Process all collected entries
        for entry_data in all_entries:
            entry_lines = self.process_entry(
                entry_data['element'],
                logger,
                entry_data['fragment_file'],
                entry_data['line_start'],
                entry_data['line_end']
            )
            output_lines.extend(entry_lines)
            output_lines.append("")

        # Write log
        logger.write_log(page_num)

        output_content = '\n'.join(output_lines)
        entry_count = len(all_entries)

        return output_content, output_filename, entry_count

    def convert_all(self):
        """Convert all XML fragments grouped by page"""
        # Group fragments by page number
        fragments_by_page = defaultdict(list)

        for xml_file in sorted(self.source_dir.glob('*.xml')):
            match = re.match(r'(\d+)_', xml_file.name)
            if match:
                page_num = match.group(1)
                fragments_by_page[page_num].append(xml_file)

        print(f"Found {len(fragments_by_page)} pages with fragments")
        print(f"Output directory: {self.output_dir}")
        print(f"Log directory: {self.log_dir}")
        print()

        total_entries = 0

        for i, (page_num, fragment_files) in enumerate(sorted(fragments_by_page.items()), 1):
            print(f"[{i}/{len(fragments_by_page)}] Processing page {page_num} "
                  f"({len(fragment_files)} fragments)...", end=' ')

            try:
                output_content, output_filename, entry_count = self.process_page_fragments(
                    page_num, fragment_files
                )
                total_entries += entry_count

                output_path = self.output_dir / output_filename
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(output_content)

                print(f"✓ → {output_filename} ({entry_count} entries)")

            except Exception as e:
                print(f"✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

        print()
        print("=" * 80)
        print("Conversion complete!")
        print(f"Total entries converted: {total_entries}")
        print(f"Output files: {self.output_dir}")
        print(f"Log files: {self.log_dir}")
        print("=" * 80)


def main():
    """Main entry point"""
    base_dir = Path(__file__).parent
    source_dir = base_dir / "20251029-raw_xml_fragments"
    output_dir = base_dir / "converted-txt-frag-v1"
    log_dir = base_dir / "conversion-frag-v1"

    converter = XMLFragmentConverter(source_dir, output_dir, log_dir)
    converter.convert_all()


if __name__ == "__main__":
    main()

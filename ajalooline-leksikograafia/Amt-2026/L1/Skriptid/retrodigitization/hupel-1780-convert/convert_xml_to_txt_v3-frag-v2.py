#!/usr/bin/env python3
# Created: 2025-11-08 14-16-25
# Author: Madis Jürviste
"""
XML Fragment to TXT Converter for Hupel 1780 Estonian-German Dictionary
Version: v3-frag-v2

Converts XML fragments to simplified text format
- Handles multiple fragments per page
- Avoids exact duplicates based on xml:id
- Groups output by page number
- NEW: Text-based fallback parsing for malformed XML
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
import re
from collections import defaultdict


class TextBasedEntryExtractor:
    """Extracts entries from malformed XML using text-based parsing"""

    def __init__(self, xml_text: str):
        self.xml_text = xml_text

    def extract_entries(self) -> List[Dict]:
        """Extract entry data from text when XML parsing fails"""
        entries = []

        # Find all entry opening tags with xml:id
        # Pattern: <entry xml:id="something"> possibly with other attributes
        entry_pattern = r'<entry[^>]*?xml:id="([^"]+)"[^>]*?>'

        # Find all entry tags
        entry_matches = list(re.finditer(entry_pattern, self.xml_text))

        for i, match in enumerate(entry_matches):
            entry_id = match.group(1)
            start_pos = match.start()

            # Find the end of this entry
            # Look for the next entry start or end of text
            if i < len(entry_matches) - 1:
                end_pos = entry_matches[i + 1].start()
            else:
                end_pos = len(self.xml_text)

            # Extract content between this entry and next
            entry_text = self.xml_text[start_pos:end_pos]

            # Check if this looks like a top-level entry (not nested)
            # Nested entries typically appear after another entry has already started
            # We'll try to detect nesting by looking backward
            text_before = self.xml_text[max(0, start_pos-500):start_pos]

            # Count open entries before this one
            open_entries_before = text_before.count('<entry') - text_before.count('</entry>')

            # If there are unclosed entries before this, it's probably nested
            if open_entries_before > 0:
                continue

            # Try to extract basic fields
            entry_data = {
                'entry_id': entry_id,
                'text': entry_text,
                'is_text_based': True
            }

            entries.append(entry_data)

        return entries


class XMLLineTracker:
    """Tracks line numbers for XML elements in fragments"""

    def __init__(self, xml_path: Path):
        self.xml_path = xml_path
        self.entry_lines = {}
        self._build_line_map()

    def _build_line_map(self):
        """Build a mapping of entry xml:id to line numbers"""
        try:
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
        except:
            pass  # If reading fails, just have empty map

    def get_lines(self, entry_id: str) -> Tuple[int, int]:
        """Get line range for an entry"""
        return self.entry_lines.get(entry_id, (0, 0))


class ConversionLogger:
    """Handles detailed logging of conversion mappings"""

    def __init__(self, page_num: str):
        self.page_num = page_num
        self.mappings = []
        self.current_line_number = 1

    def add_mapping(self, marker: str, content: str, element_desc: str, xml_location: str):
        """Add a mapping entry"""
        self.mappings.append({
            'output_line': self.current_line_number,
            'marker': marker,
            'content': content,
            'source_desc': element_desc,
            'xml_lines': xml_location
        })
        self.current_line_number += 1

    def increment_line(self):
        """Increment output line counter"""
        self.current_line_number += 1

    def write_log(self, page_num: str):
        """Write log file"""
        pass  # Simplified for now


class XMLFragmentConverter:
    """Main converter class for XML fragments with malformed XML recovery"""

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

        self.recovery_stats = {
            'xml_parsed': 0,
            'text_recovered': 0,
            'entries_recovered': 0
        }

    def get_text_content(self, element, strip: bool = True) -> str:
        """Extract text content from an element"""
        if element is None:
            return ''
        text = ''.join(element.itertext())
        return text.strip() if strip else text

    def process_entry_from_xml(self, entry_elem, logger: ConversionLogger,
                               fragment_file: str, line_start: int, line_end: int) -> List[str]:
        """Process entry from successfully parsed XML"""
        output_lines = []

        # Entry marker
        entry_id = entry_elem.get('{http://www.w3.org/XML/1998/namespace}id', 'unknown')
        output_lines.append(f"<entry xml:id=\"{entry_id}\">")
        logger.increment_line()

        # Process headword
        orth_elem = entry_elem.find('.//form[@type="lemma"]/orth')
        if orth_elem is not None:
            headword = self.get_text_content(orth_elem)
            output_lines.append(f"* {headword}")
            logger.add_mapping('*', headword, '<form type="lemma"><orth>',
                             f'fragment {fragment_file}, XML line {line_start}-{line_end}')

        # Process variants
        for variant_elem in entry_elem.findall('.//form[@type="variant"]/orth'):
            variant = self.get_text_content(variant_elem)
            if variant:
                output_lines.append(f"~ {variant}")
                logger.add_mapping('~', variant, '<form type="variant"><orth>',
                                 f'fragment {fragment_file}, XML line {line_start}-{line_end}')

        # Process grammar
        gram_forms = []
        for gram_elem in entry_elem.findall('.//gramGrp/gram'):
            if gram_elem.get('type') in ['gen', 'part', 'inf', 'impf']:
                form = self.get_text_content(gram_elem)
                if form:
                    gram_forms.append(form)

        if gram_forms:
            output_lines.append(f":gr: {', '.join(gram_forms)}")
            logger.add_mapping(':gr:', ', '.join(gram_forms), '<gramGrp><gram>',
                             f'fragment {fragment_file}, XML line {line_start}-{line_end}')

        # Process type (gender/number)
        for typ_elem in entry_elem.findall('.//gramGrp/gram[@type="type"]'):
            typ = self.get_text_content(typ_elem)
            if typ:
                output_lines.append(f":ty: {typ}")
                logger.add_mapping(':ty:', typ, '<gram type="type">',
                                 f'fragment {fragment_file}, XML line {line_start}-{line_end}')

        # Process translations
        for sense_elem in entry_elem.findall('sense'):
            if sense_elem.get('xml:id') in self.DIALECT_IDS:
                continue

            for tr_elem in sense_elem.findall('cit[@type="translation"]/quote'):
                translation = self.get_text_content(tr_elem)
                if translation:
                    output_lines.append(f":tr: {translation}")
                    logger.add_mapping(':tr:', translation, '<cit type="translation"><quote>',
                                     f'fragment {fragment_file}, XML line {line_start}-{line_end}')

        # Process cross-references
        for xr_elem in entry_elem.findall('.//xr/ref'):
            ref_text = self.get_text_content(xr_elem)
            if ref_text:
                output_lines.append(f":xr: {ref_text}")
                logger.add_mapping(':xr:', ref_text, '<xr><ref>',
                                 f'fragment {fragment_file}, XML line {line_start}-{line_end}')

        # Process dialect markers
        dialects = []
        for usg_elem in entry_elem.findall('.//usg[@type="dialect"]'):
            dialect = self.get_text_content(usg_elem)
            if dialect and dialect not in dialects:
                dialects.append(dialect)

        if dialects:
            output_lines.append(f":di: {' '.join(dialects)}")
            logger.add_mapping(':di:', ' '.join(dialects), '<usg type="dialect">',
                             f'fragment {fragment_file}, XML line {line_start}-{line_end}')

        # Process regional markers
        for usg_elem in entry_elem.findall('.//usg[@type="geographic"]'):
            regional = self.get_text_content(usg_elem)
            if regional:
                output_lines.append(f":rn: {regional}")
                logger.add_mapping(':rn:', regional, '<usg type="geographic">',
                                 f'fragment {fragment_file}, XML line {line_start}-{line_end}')

        # Process nested entries - show as :mw: content under parent
        for nested_entry in entry_elem.findall('entry'):
            phrase_elem = nested_entry.find('form[@type="phrase"]/orth')
            compound_elem = nested_entry.find('form[@type="compound"]/orth')

            if phrase_elem is not None:
                phrase = self.get_text_content(phrase_elem)
                if ' ' in phrase:  # multiword
                    output_lines.append(f":mw: {phrase}")
                    logger.add_mapping(':mw:', phrase, 'nested <entry> with <form type="phrase">',
                                     f'nested entry, fragment {fragment_file}')

                    # Get translation for nested entry
                    for sense in nested_entry.findall('sense'):
                        for tr_elem in sense.findall('cit[@type="translation"]/quote'):
                            translation = self.get_text_content(tr_elem)
                            if translation:
                                output_lines.append(f":mw/tr: {translation}")
                                logger.add_mapping(':mw/tr:', translation, 'nested entry translation',
                                                 f'nested entry, fragment {fragment_file}')
                else:  # single word sub-entry
                    output_lines.append(f":se: {phrase}")
                    logger.add_mapping(':se:', phrase, 'nested <entry> with <form type="phrase">',
                                     f'nested entry, fragment {fragment_file}')

                    for sense in nested_entry.findall('sense'):
                        for tr_elem in sense.findall('cit[@type="translation"]/quote'):
                            translation = self.get_text_content(tr_elem)
                            if translation:
                                output_lines.append(f":se/tr: {translation}")
                                logger.add_mapping(':se/tr:', translation, 'nested entry translation',
                                                 f'nested entry, fragment {fragment_file}')

            elif compound_elem is not None:
                compound = self.get_text_content(compound_elem)
                output_lines.append(f":se: {compound}")
                logger.add_mapping(':se:', compound, 'nested <entry> with <form type="compound">',
                                 f'nested entry, fragment {fragment_file}')

                for sense in nested_entry.findall('sense'):
                    for tr_elem in sense.findall('cit[@type="translation"]/quote'):
                        translation = self.get_text_content(tr_elem)
                        if translation:
                            output_lines.append(f":se/tr: {translation}")
                            logger.add_mapping(':se/tr:', translation, 'nested entry translation',
                                             f'nested entry, fragment {fragment_file}')

        return output_lines

    def process_entry_from_text(self, entry_text: str, entry_id: str,
                                logger: ConversionLogger, fragment_file: str) -> List[str]:
        """Process entry from text-based extraction (malformed XML fallback)"""
        output_lines = []

        # Entry marker
        output_lines.append(f"<entry xml:id=\"{entry_id}\">")
        logger.increment_line()

        # Try to extract headword
        headword_match = re.search(r'<orth[^>]*>([^<]+)</orth>', entry_text)
        if headword_match:
            headword = headword_match.group(1).strip()
            output_lines.append(f"* {headword}")
            logger.add_mapping('*', headword, 'text-based: <orth>',
                             f'recovered from {fragment_file}')

        # Try to extract translations
        for tr_match in re.finditer(r'<quote[^>]*>([^<]+)</quote>', entry_text):
            translation = tr_match.group(1).strip()
            if translation:
                output_lines.append(f":tr: {translation}")
                logger.add_mapping(':tr:', translation, 'text-based: <quote>',
                                 f'recovered from {fragment_file}')

        # Try to extract dialect markers
        for usg_match in re.finditer(r'<usg[^>]*type="dialect"[^>]*>([^<]+)</usg>', entry_text):
            dialect = usg_match.group(1).strip()
            if dialect:
                output_lines.append(f":di: {dialect}")
                logger.add_mapping(':di:', dialect, 'text-based: <usg type="dialect">',
                                 f'recovered from {fragment_file}')

        # If we got nothing useful, at least mark it exists
        if len(output_lines) == 1:  # Only entry marker
            output_lines.append(f":tr: [entry recovered from malformed XML]")
            logger.add_mapping(':tr:', '[entry recovered from malformed XML]',
                             'text-based: minimal recovery',
                             f'recovered from {fragment_file}')

        return output_lines

    def process_page_fragments(self, page_num: str, fragment_files: List[Path]) -> Tuple[str, str, int, Dict]:
        """Process all fragments for a single page with malformed XML recovery"""
        # Track seen entries to avoid duplicates
        seen_entries: Set[str] = set()
        all_entries = []

        page_recovery_stats = {
            'xml_parsed': 0,
            'text_recovered': 0,
            'entries_recovered': 0
        }

        # Collect all unique entries from all fragments
        for frag_file in fragment_files:
            line_tracker = XMLLineTracker(frag_file)

            # First try XML parsing
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
                        'line_end': line_end,
                        'is_text_based': False
                    })

                page_recovery_stats['xml_parsed'] += 1

            except Exception as e:
                # XML parsing failed - try text-based extraction
                page_recovery_stats['text_recovered'] += 1

                try:
                    with open(frag_file, 'r', encoding='utf-8') as f:
                        xml_text = f.read()

                    extractor = TextBasedEntryExtractor(xml_text)
                    text_entries = extractor.extract_entries()

                    for entry_data in text_entries:
                        entry_id = entry_data['entry_id']

                        # Skip if we've already seen this entry
                        if entry_id in seen_entries:
                            continue

                        seen_entries.add(entry_id)
                        page_recovery_stats['entries_recovered'] += 1

                        all_entries.append({
                            'element': None,
                            'entry_id': entry_id,
                            'fragment_file': frag_file.name,
                            'line_start': 0,
                            'line_end': 0,
                            'is_text_based': True,
                            'text': entry_data['text']
                        })

                except Exception as text_error:
                    # Even text-based extraction failed
                    pass

        # Generate output filename
        today = datetime.now().strftime('%Y%m%d')
        output_filename = f"{page_num}_Hu-1780_{today}_frag-v2.txt"

        # Create logger
        logger = ConversionLogger(page_num)

        # Generate output
        output_lines = [f"--- page {page_num} ---", ""]
        logger.increment_line()
        logger.increment_line()

        for entry_data in all_entries:
            if entry_data['is_text_based']:
                # Process using text-based method
                entry_lines = self.process_entry_from_text(
                    entry_data['text'],
                    entry_data['entry_id'],
                    logger,
                    entry_data['fragment_file']
                )
            else:
                # Process using XML method
                entry_lines = self.process_entry_from_xml(
                    entry_data['element'],
                    logger,
                    entry_data['fragment_file'],
                    entry_data['line_start'],
                    entry_data['line_end']
                )

            output_lines.extend(entry_lines)
            output_lines.append("")

        output_content = '\n'.join(output_lines)
        entry_count = len(all_entries)

        return output_content, output_filename, entry_count, page_recovery_stats

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
                output_content, output_filename, entry_count, page_stats = self.process_page_fragments(
                    page_num, fragment_files
                )

                # Update global stats
                self.recovery_stats['xml_parsed'] += page_stats['xml_parsed']
                self.recovery_stats['text_recovered'] += page_stats['text_recovered']
                self.recovery_stats['entries_recovered'] += page_stats['entries_recovered']

                # Write output
                output_path = self.output_dir / output_filename
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(output_content)

                total_entries += entry_count

                recovery_info = ""
                if page_stats['entries_recovered'] > 0:
                    recovery_info = f" (+{page_stats['entries_recovered']} recovered)"

                print(f"✓ → {output_filename} ({entry_count} entries{recovery_info})")

            except Exception as e:
                print(f"✗ Error: {e}")
                continue

        print()
        print("=" * 80)
        print("Conversion complete!")
        print(f"Total entries converted: {total_entries}")
        print(f"Fragments parsed via XML: {self.recovery_stats['xml_parsed']}")
        print(f"Fragments recovered via text: {self.recovery_stats['text_recovered']}")
        print(f"Entries recovered from malformed XML: {self.recovery_stats['entries_recovered']}")
        print(f"Output files: {self.output_dir}")
        print(f"Log files: {self.log_dir}")
        print("=" * 80)


def main():
    # Set up paths
    script_dir = Path(__file__).parent
    source_dir = script_dir / "20251029-raw_xml_fragments"
    output_dir = script_dir / "converted-txt-frag-v2"
    log_dir = script_dir / "conversion-frag-v2"

    # Create converter and run
    converter = XMLFragmentConverter(source_dir, output_dir, log_dir)
    converter.convert_all()


if __name__ == '__main__':
    main()

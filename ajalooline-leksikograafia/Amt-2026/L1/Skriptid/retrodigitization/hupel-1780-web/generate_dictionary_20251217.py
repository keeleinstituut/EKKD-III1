#!/usr/bin/env python3
# Created: 2025-12-17 18-31-43
# Author: Madis Jürviste
"""Dictionary generator with improved formatting - Version 2025-12-17 v2"""
import re
from pathlib import Path

def parse_entry(lines, start_idx):
    """Parse entry preserving exact structure"""
    entry = {'id': '', 'elements': []}

    i = start_idx
    match = re.match(r'<entry xml:id="([^"]+)">', lines[i])
    if match:
        entry['id'] = match.group(1)

    i += 1

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('<entry') or line.startswith('---'):
            break

        if not line or line.startswith('#'):
            i += 1
            continue

        # Parse tags with type tracking for tr
        if line.startswith('* '):
            entry['elements'].append(('hw', line[2:].strip(), None))
        elif line.startswith('~ '):
            entry['elements'].append(('var', line[2:].strip(), None))
        elif line.startswith(':gr:'):
            entry['elements'].append(('gr', line[4:].strip(), None))
        elif line.startswith(':tr-'):
            # Extract type number (tr-1, tr-2, etc.)
            match = re.match(r':tr-(\d+):', line)
            tr_type = match.group(1) if match else '0'
            content = line.split(':', 2)[2].strip() if len(line.split(':', 2)) > 2 else ''
            entry['elements'].append(('tr', content, tr_type))
        elif line.startswith(':tr:'):
            entry['elements'].append(('tr', line[4:].strip(), '0'))
        elif line.startswith(':mw:'):
            entry['elements'].append(('mw', line[4:].strip(), None))
        elif line.startswith(':mw/tr:'):
            entry['elements'].append(('mwtr', line[7:].strip(), None))
        elif line.startswith(':di:'):
            entry['elements'].append(('di', line[4:].strip(), None))
        elif line.startswith(':xr:'):
            entry['elements'].append(('xr', line[4:].strip(), None))
        elif line.startswith(':ex:'):
            entry['elements'].append(('ex', line[4:].strip(), None))
        elif line.startswith(':rn:'):
            entry['elements'].append(('rn', line[4:].strip(), None))
        elif line.startswith(':us:'):
            entry['elements'].append(('us', line[4:].strip(), None))
        elif line.startswith(':rg:'):
            entry['elements'].append(('rg', line[4:].strip(), None))
        elif line.startswith(':se:'):
            entry['elements'].append(('se', line[4:].strip(), None))
        elif line.startswith(':ad:'):
            entry['elements'].append(('ad', line[4:].strip(), None))
        elif line.startswith(':ph:'):
            entry['elements'].append(('ph', line[4:].strip(), None))
        elif line.startswith(':fr:'):
            entry['elements'].append(('fr', line[4:].strip(), None))

        i += 1

    return entry, i

def parse_file(filepath):
    """Parse dictionary file maintaining order"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    entries = []
    i = 0

    while i < len(lines):
        if lines[i].strip().startswith('<entry'):
            entry, i = parse_entry(lines, i)
            if entry['elements']:
                entries.append(entry)
        else:
            i += 1

    return entries

def get_first_letter(entry):
    """Get first letter of headword"""
    for elem_type, content, _ in entry['elements']:
        if elem_type in ['hw', 'var']:
            if content:
                return content[0].upper()
    return 'A'

def process_xr_content(content, entry_id_map):
    """Process cross-reference to make links clickable, handling square bracket syntax"""
    # Check for pattern like "s. woöraspid. [woöraspiddo]"
    bracket_match = re.match(r'([sf]\.)\s+(.+?)\s*\[([^\]]+)\]', content)
    if bracket_match:
        prefix = bracket_match.group(1)
        display_word = bracket_match.group(2).strip()
        target_word = bracket_match.group(3).strip()

        target_id = entry_id_map.get(target_word.lower())
        if target_id:
            return f'{prefix} <a href="#{target_id}" class="xr-link">{display_word}</a>'
        else:
            return f'{prefix} {display_word}'

    # Original pattern without brackets
    match = re.match(r'([sf]\.)\s+(.+)', content)
    if match:
        prefix = match.group(1)
        ref_word = match.group(2).strip()

        target_id = entry_id_map.get(ref_word.lower())
        if target_id:
            return f'{prefix} <a href="#{target_id}" class="xr-link">{ref_word}</a>'

    return content

def should_omit_comma_after_gr(gr_content):
    """Check if comma should be omitted after grammar element"""
    gr_stripped = gr_content.strip().rstrip('.')
    return gr_stripped in ['G', 'Gen', 'Ac', 'Acc', 'Pl']

def generate_entry_html(entry, entry_id_map):
    """Generate HTML with sequential element processing"""
    parts = []
    parts.append(f'<div class="entry" id="{entry["id"]}">')

    # Collect all headwords/variants for header
    all_headwords = []
    for e_type, e_content, _ in entry['elements']:
        if e_type in ['hw', 'var']:
            all_headwords.append(e_content)

    # Output combined headword
    if all_headwords:
        headword_line = ' ~ '.join(all_headwords)
        parts.append(f'<div class="headword">{headword_line}</div>')

    # Content generation - process elements sequentially
    content_parts = []
    i = 0
    elements = entry['elements']

    while i < len(elements):
        e_type, e_content, e_extra = elements[i]

        # Handle hw/var: only output in content if immediately followed by gr/di
        if e_type in ['hw', 'var']:
            # Check if this hw/var is followed by gr or di
            has_following_gr_or_di = (i + 1 < len(elements) and
                                     elements[i + 1][0] in ['gr', 'di'])

            # Only output hw/var in content if immediately followed by gr or di
            if has_following_gr_or_di:
                content_parts.append(f'<span class="hw-repeat">{e_content}</span> ')

            i += 1

            # Check for immediately following gr tags
            gr_parts = []
            while i < len(elements) and elements[i][0] == 'gr':
                gr_parts.append(elements[i][1])
                i += 1

            if gr_parts:
                # Format grammar info
                for idx, gr_item in enumerate(gr_parts):
                    if idx == 0:
                        content_parts.append(f'<span class="grammatical-inline">&lt;{gr_item}')
                    else:
                        prev_gr = gr_parts[idx - 1]
                        if should_omit_comma_after_gr(prev_gr):
                            content_parts.append(f' {gr_item}')
                        else:
                            content_parts.append(f', {gr_item}')
                    if idx == len(gr_parts) - 1:
                        content_parts.append('&gt;</span> ')

            # Check for immediately following di tag
            if i < len(elements) and elements[i][0] == 'di':
                content_parts.append(f'<span class="dialect-info-inline">{elements[i][1]}</span> ')
                i += 1
            continue

        # Handle mw: output with immediately following context
        elif e_type == 'mw':
            # Always add bullet for mw
            content_parts.append('• ')
            content_parts.append(f'<span class="mw-expression-inline">{e_content}</span>')
            i += 1

            # Collect immediately following di, mwtr, tr, ex, us, rn, rg tags
            tr_items = []
            while i < len(elements) and elements[i][0] in ['di', 'mwtr', 'tr', 'ex', 'us', 'rn', 'rg']:
                next_type, next_content, next_extra = elements[i]

                if next_type == 'di':
                    content_parts.append(f' <span class="dialect-info-inline">{next_content}</span>')
                elif next_type == 'mwtr':
                    # Collect consecutive mwtr
                    mwtr_items = [next_content]
                    i += 1
                    while i < len(elements) and elements[i][0] == 'mwtr':
                        mwtr_items.append(elements[i][1])
                        i += 1
                    content_parts.append(f' <span class="mw-translation-inline">{", ".join(mwtr_items)}</span>')
                    continue
                elif next_type == 'tr':
                    # Collect all consecutive tr elements
                    tr_items.append(next_content)
                    i += 1
                    while i < len(elements) and elements[i][0] == 'tr':
                        tr_items.append(elements[i][1])
                        i += 1
                    # Output all collected tr as comma-separated
                    content_parts.append(f' <span class="translation-inline">{", ".join(tr_items)}</span>')
                    continue
                elif next_type == 'ex':
                    # Collect consecutive ex
                    ex_items = [next_content]
                    i += 1
                    while i < len(elements) and elements[i][0] == 'ex':
                        ex_items.append(elements[i][1])
                        i += 1
                    content_parts.append(f' <span class="ex-info-inline">({", ".join(ex_items)})</span>')
                    continue
                elif next_type in ['us', 'rn']:
                    content_parts.append(f' <span class="ex-info-inline">({next_content})</span>')
                elif next_type == 'rg':
                    content_parts.append(f' <span class="meta-info-inline">({next_content})</span>')

                i += 1

            content_parts.append(' ')
            continue

        # Handle standalone tr (not after mw)
        elif e_type == 'tr':
            # Collect consecutive tr elements with type grouping
            tr_groups = []
            current_tr_type = e_extra
            current_tr_group = [e_content]

            j = i + 1
            while j < len(elements) and elements[j][0] == 'tr':
                next_tr_type = elements[j][2]
                next_tr_content = elements[j][1]

                if next_tr_type == current_tr_type:
                    current_tr_group.append(next_tr_content)
                else:
                    tr_groups.append((current_tr_type, current_tr_group))
                    current_tr_type = next_tr_type
                    current_tr_group = [next_tr_content]
                j += 1

            tr_groups.append((current_tr_type, current_tr_group))

            # Format with proper punctuation
            tr_parts = []
            for tr_type, tr_list in tr_groups:
                tr_parts.append(', '.join(tr_list))

            tr_text = '; '.join(tr_parts)
            content_parts.append(f'<span class="translation-inline">{tr_text}</span> ')
            i = j
            continue

        # Handle standalone di
        elif e_type == 'di':
            content_parts.append(f'<span class="dialect-info-inline">{e_content}</span> ')

        # Handle xr
        elif e_type == 'xr':
            xr_html = process_xr_content(e_content, entry_id_map)
            content_parts.append(f'<span class="cross-ref-inline">{xr_html}</span> ')

        # Handle ex (standalone)
        elif e_type == 'ex':
            ex_items = [e_content]
            j = i + 1
            while j < len(elements) and elements[j][0] == 'ex':
                ex_items.append(elements[j][1])
                j += 1
            content_parts.append(f'<span class="ex-info-inline">({", ".join(ex_items)})</span> ')
            i = j
            continue

        # Handle us, rn (dark green)
        elif e_type in ['us', 'rn']:
            content_parts.append(f'<span class="ex-info-inline">({e_content})</span> ')

        # Handle rg, ad, ph, fr
        elif e_type in ['rg', 'ad', 'ph', 'fr']:
            content_parts.append(f'<span class="meta-info-inline">({e_content})</span> ')

        # Handle se
        elif e_type == 'se':
            content_parts.append('• ')
            content_parts.append(f'<span class="subentry-inline">{e_content}</span> ')

        # Handle gr (standalone, shouldn't normally happen)
        elif e_type == 'gr':
            gr_parts = [e_content]
            j = i + 1
            while j < len(elements) and elements[j][0] == 'gr':
                gr_parts.append(elements[j][1])
                j += 1

            for idx, gr_item in enumerate(gr_parts):
                if idx == 0:
                    content_parts.append(f'<span class="grammatical-inline">&lt;{gr_item}')
                else:
                    prev_gr = gr_parts[idx - 1]
                    if should_omit_comma_after_gr(prev_gr):
                        content_parts.append(f' {gr_item}')
                    else:
                        content_parts.append(f', {gr_item}')
                if idx == len(gr_parts) - 1:
                    content_parts.append('&gt;</span> ')
            i = j
            continue

        i += 1

    if content_parts:
        parts.append('<div class="content-block">')
        parts.append(''.join(content_parts))
        parts.append('</div>')

    parts.append('</div>')
    return '\n'.join(parts)

def main():
    # Parse all files in order
    dict_dir = Path('H-1780-cml')
    files = sorted(dict_dir.glob('*_Hu-1780_*.txt'),
                   key=lambda x: int(x.stem.split('_')[0]))

    all_entries = []
    for filepath in files:
        print(f"Parsing {filepath.name}...")
        entries = parse_file(filepath)
        all_entries.extend(entries)

    total_entries = len(all_entries)
    print(f"Total entries: {total_entries}")

    # Create entry ID map
    entry_id_map = {}
    for entry in all_entries:
        for e_type, e_content, _ in entry['elements']:
            if e_type in ['hw', 'var']:
                entry_id_map[e_content.lower()] = entry['id']
                break

    # Group by first letter maintaining exact file order (allowing letter repeats)
    letter_entries = []
    current_letter = None
    current_letter_entries = []

    for entry in all_entries:
        letter = get_first_letter(entry)
        if letter != current_letter:
            if current_letter_entries:
                letter_entries.append((current_letter, current_letter_entries))
            current_letter = letter
            current_letter_entries = [entry]
        else:
            current_letter_entries.append(entry)

    if current_letter_entries:
        letter_entries.append((current_letter, current_letter_entries))

    # Generate HTML
    content_parts = []
    letters = []

    for letter, entries in letter_entries:
        letters.append(letter)
        content_parts.append(f'<div id="letter-{letter}" class="letter-section">')
        content_parts.append(f'<h2 class="letter-heading">{letter}</h2>')

        for entry in entries:
            content_parts.append(generate_entry_html(entry, entry_id_map))

        content_parts.append('</div>')

    html_content = '\n'.join(content_parts)

    nav_items = '\n'.join([f'                <a href="#letter-{letter}">{letter}</a>'
                           for letter in letters])

    html_template = f'''<!DOCTYPE html>
<html lang="et">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AWH 1780 et-de</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Palatino Linotype', Palatino, 'Book Antiqua', Georgia, serif;
            background-color: #f5f3ed;
            color: #2c2416;
            line-height: 1.8;
            font-size: 16px;
            hyphens: none;
            -webkit-hyphens: none;
            -moz-hyphens: none;
            -ms-hyphens: none;
        }}

        .container {{
            display: flex;
            min-height: 100vh;
        }}

        .sidebar {{
            position: fixed;
            left: 0;
            top: 0;
            width: 180px;
            height: 100vh;
            background-color: #e8e4d9;
            border-right: 1px solid #ccc6b5;
            padding: 30px 20px;
            overflow-y: auto;
            box-shadow: 2px 0 8px rgba(0,0,0,0.05);
        }}

        .sidebar-title {{
            font-size: 1.3em;
            font-weight: 600;
            color: #1a150f;
            letter-spacing: 0.5px;
            line-height: 1.3;
        }}

        .sidebar-subtitle {{
            font-size: 1em;
            font-weight: 500;
            color: #4a3f2f;
            margin-top: 5px;
        }}

        .entry-count {{
            font-size: 0.85em;
            color: #6b5d44;
            margin-top: 8px;
            margin-bottom: 25px;
        }}

        .alphabet-nav {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .alphabet-nav a {{
            color: #4a3f2f;
            text-decoration: none;
            font-size: 0.95em;
            padding: 6px 12px;
            border-radius: 4px;
            transition: all 0.2s ease;
            font-weight: 500;
        }}

        .alphabet-nav a:hover {{
            background-color: #d4cfbc;
            color: #1a150f;
        }}

        .main-content {{
            margin-left: 180px;
            flex: 1;
            padding: 50px 80px;
            max-width: 1100px;
        }}

        .entry {{
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #e0dcc8;
            scroll-margin-top: 80px;
        }}

        .entry:last-child {{
            border-bottom: none;
        }}

        .headword {{
            font-size: 1.4em;
            font-weight: 700;
            color: #1a150f;
            margin-bottom: 10px;
            letter-spacing: 0.3px;
            line-height: 1.4;
        }}

        .content-block {{
            line-height: 1.9;
            max-width: 100%;
            font-size: 1.275em;
        }}

        .hw-repeat {{
            font-weight: 600;
            color: #2c2416;
        }}

        .grammatical-inline {{
            color: #5a4f3d;
        }}

        .translation-inline {{
            font-style: italic;
            color: #2c2416;
        }}

        .mw-expression-inline {{
            font-weight: 600;
            color: #2c2416;
        }}

        .mw-translation-inline {{
            font-style: italic;
            color: #3d3426;
        }}

        .cross-ref-inline {{
            color: #6b5d44;
        }}

        .xr-link {{
            color: #6b5d44;
            text-decoration: underline;
            cursor: pointer;
        }}

        .xr-link:hover {{
            color: #4a3f2f;
        }}

        .dialect-info-inline {{
            color: #8B0000;
            font-size: 0.9em;
        }}

        .ex-info-inline {{
            color: #006400;
            font-size: 0.95em;
            font-style: italic;
        }}

        .meta-info-inline {{
            color: #7a6f5d;
            font-size: 0.95em;
            font-style: italic;
        }}

        .subentry-inline {{
            font-weight: 600;
            color: #3d3426;
        }}

        .letter-section {{
            scroll-margin-top: 30px;
        }}

        .letter-heading {{
            font-size: 2.8em;
            font-weight: 300;
            color: #1a150f;
            margin: 55px 0 35px 0;
            padding-bottom: 14px;
            border-bottom: 2px solid #ccc6b5;
            letter-spacing: 2px;
        }}

        .letter-heading:first-child {{
            margin-top: 0;
        }}

        @media (max-width: 768px) {{
            .sidebar {{
                width: 120px;
                padding: 20px 10px;
            }}

            .main-content {{
                margin-left: 120px;
                padding: 30px 20px;
            }}

            .headword {{
                font-size: 1.25em;
            }}

            .content-block {{
                font-size: 1.1em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <aside class="sidebar">
            <div class="sidebar-title">AWH 1780</div>
            <div class="sidebar-subtitle">et-de</div>
            <div class="entry-count">{total_entries} entries</div>
            <nav class="alphabet-nav">
{nav_items}
            </nav>
        </aside>

        <main class="main-content">
{html_content}
        </main>
    </div>

    <script>
        document.querySelectorAll('.alphabet-nav a').forEach(link => {{
            link.addEventListener('click', (e) => {{
                e.preventDefault();
                const href = link.getAttribute('href');
                const targetId = href.substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {{
                    targetElement.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                    history.pushState(null, null, href);
                }}
            }});
        }});

        document.addEventListener('click', (e) => {{
            if (e.target.classList.contains('xr-link')) {{
                e.preventDefault();
                const href = e.target.getAttribute('href');
                const targetId = href.substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {{
                    targetElement.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                    targetElement.style.backgroundColor = '#f0ead6';
                    setTimeout(() => {{ targetElement.style.backgroundColor = ''; }}, 2000);
                    history.pushState(null, null, href);
                }}
            }}
        }});

        if (window.location.hash) {{
            setTimeout(() => {{
                const targetElement = document.getElementById(window.location.hash.substring(1));
                if (targetElement) {{
                    targetElement.scrollIntoView({{ block: 'start' }});
                }}
            }}, 100);
        }}
    </script>
</body>
</html>'''

    with open('AWH-1780-et-de.html', 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"Dictionary updated: AWH-1780-et-de.html")
    print(f"Letters: {', '.join(letters)}")

if __name__ == '__main__':
    main()

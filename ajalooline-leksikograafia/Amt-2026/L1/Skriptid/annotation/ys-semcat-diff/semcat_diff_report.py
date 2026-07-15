#!/usr/bin/env python3
# Created: 2026-07-06 20-01-59
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
"""Visual diff report: AMT Sem-Cat (before) vs ÜS YS-Sem-Cat (just added).

Reads AMT-Master_annotated_with_YS-semcat.json and writes semcat-diff-report.html —
a self-contained "collation sheet": historical annotation (EB Garamond) collated
against database tags (IBM Plex Mono), grouped by apparatus sigla:
  ≠ disjoint · ~ partial · + ÜS adds · = match · ∅ no ÜS data

Tokens are compared case-insensitively; the AMT subtype suffix (":staatus", ...)
is stripped to its base for matching, and a small documented equivalence map
bridges the two vocabularies (e.g. AGENT_TEGEVUS ~ in_tegija).

Usage: uv run python semcat_diff_report.py [annotated.json] [report.html]
"""
import html
import json
import sys
from pathlib import Path
from string import Template

BASE = Path("/Users/q/dev/gen/Katus-DEV/Katus-ALUSANDMED/YS-Master-semcat-diff/Sem-Cat-YS-to-AMT")
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "AMT-Master_annotated_with_YS-semcat.json"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE / "semcat-diff-report.html"

# documented cross-vocabulary equivalences (AMT-side token -> ÜS-side token)
EQUIV = {
    ("agent_tegevus", "in_tegija"),
    ("in_roll:sugulus", "in_sugulane"),
    ("asutus", "koht_asutus"),
    ("in", "inimene"),
    ("grp_inimene", "inimene"),
}
PLACEHOLDERS = {"NULL", "---", "???", ""}


def toks(s):
    return [t.strip() for t in (s or "").split(",") if t.strip() and t.strip() not in PLACEHOLDERS]


def amt_keys(tok):
    """Match keys for one AMT token: full lowercase + base before ':'."""
    t = tok.lower()
    return {t, t.split(":")[0]}


def classify_amt(tok, ys_tokens):
    ys = {y.lower() for y in ys_tokens}
    keys = amt_keys(tok)
    if keys & ys:
        return "shared"
    if any((k, y) in EQUIV for k in keys for y in ys):
        return "equiv"
    return "amt-only"


def classify_ys(tok, amt_tokens):
    y = tok.lower()
    akeys = set().union(*[amt_keys(a) for a in amt_tokens]) if amt_tokens else set()
    if y in akeys:
        return "shared"
    if any((k, y) in EQUIV for k in akeys):
        return "equiv"
    return "ys-new"


entries = json.load(SRC.open(encoding="utf-8"))["AMT-Master"]
groups = {k: [] for k in ("disjoint", "partial", "ys-adds", "match", "ys-null", "amt-empty")}

for e in entries:
    amt = toks(e.get("Sem-Cat"))
    ys = toks(e.get("YS-Sem-Cat"))
    a_cls = [(t, classify_amt(t, ys)) for t in amt]
    y_cls = [(t, classify_ys(t, amt)) for t in ys]
    if not ys:
        status = "ys-null"
    elif not amt:
        status = "amt-empty"
    else:
        amt_covered = all(c != "amt-only" for _, c in a_cls)
        ys_new = [t for t, c in y_cls if c == "ys-new"]
        any_shared = any(c != "amt-only" for _, c in a_cls)
        if amt_covered and not ys_new:
            status = "match"
        elif amt_covered:
            status = "ys-adds"
        elif any_shared:
            status = "partial"
        else:
            status = "disjoint"
    groups[status].append((e, a_cls, y_cls))

SECTIONS = [
    ("disjoint", "≠", "No overlap",
     "ÜS assigns entirely different categories — none of your tokens is confirmed."),
    ("partial", "~", "Partial overlap",
     "Some categories agree; others in your annotation are not confirmed by ÜS."),
    ("ys-adds", "+", "ÜS adds categories",
     "Your annotation is fully confirmed, and ÜS tags additional categories."),
    ("match", "=", "Full match",
     "Both sides agree, after case, subtype and equivalence normalisation."),
    ("ys-null", "∅", "No ÜS data",
     "The mapped ÜS word carries no semantic types in Ekilex."),
    ("amt-empty", "†", "Only ÜS has categories",
     "Your Sem-Cat was empty or a placeholder; ÜS supplies the only information."),
]

CHIP_TITLE = {
    "shared": "confirmed: present on both sides",
    "equiv": "equivalent label across the two vocabularies (see key)",
    "amt-only": "only in your Sem-Cat — not confirmed by ÜS",
    "ys-new": "new in ÜS — absent from your Sem-Cat",
}


def chip(tok, cls):
    return f'<span class="chip {cls}" title="{CHIP_TITLE[cls]}">{html.escape(tok)}</span>'


sections_html = []
for key, siglum, title, blurb in SECTIONS:
    items = groups[key]
    if not items:
        continue
    body = []
    for e, a_cls, y_cls in sorted(items, key=lambda x: x[0]["Amt-Master-ID"].strip().lower()):
        lemma = e["Amt-Master-ID"].strip()
        deft = html.escape(e.get("DEF_et") or "")
        ysl = html.escape((e.get("YS-lemma") or "").strip())
        before = " ".join(chip(t, c) for t, c in a_cls) or '<span class="none">—</span>'
        after = " ".join(chip(t, c) for t, c in y_cls) or '<span class="none">—</span>'
        body.append(
            f'<tr data-lemma="{html.escape(lemma.lower())}">'
            f'<td class="lemma" title="{deft}">{html.escape(lemma)}</td>'
            f'<td class="yslemma">{ysl}</td>'
            f'<td class="cats">{before}</td>'
            f'<td class="cats">{after}</td></tr>'
        )
    sections_html.append(
        f'<section id="{key}">'
        f'<h2><span class="siglum" aria-hidden="true">{siglum}</span>{html.escape(title)}'
        f'<span class="count">{len(items)}</span></h2>'
        f'<p class="blurb">{html.escape(blurb)}</p>'
        f'<div class="scroll"><table>'
        f'<thead><tr><th>Lemma</th><th>ÜS word</th>'
        f'<th>Sem-Cat · before</th><th>ÜS semantic types</th></tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div></section>'
    )

filters_html = "".join(
    f'<button class="filter" data-target="{key}" aria-pressed="true">'
    f'<span class="fsig">{siglum}</span><span class="flab">{html.escape(title)}</span>'
    f'<span class="fnum">{len(groups[key])}</span></button>'
    for key, siglum, title, _ in SECTIONS if groups[key]
)

n = len(entries)
thesis = (
    f"Of {n} lemmas: <b>{len(groups['match'])}</b> agree, "
    f"<b>{len(groups['ys-adds'])}</b> gain new categories from ÜS, "
    f"<b>{len(groups['disjoint'])}</b> diverge entirely, "
    f"<b>{len(groups['partial'])}</b> in part; "
    f"<b>{len(groups['ys-null'])}</b> have no ÜS tags."
)
equiv_str = " · ".join(f"{a.upper()} ~ {b}" for a, b in sorted(EQUIV))

page = Template(r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sem-Cat × ÜS — collation report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400..700;1,400..600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --paper:   #edeae2;
    --card:    #faf8f2;
    --ink:     #2b2418;
    --faded:   #8a8172;
    --rule:    #d9d3c4;
    --oxblood: #8c2b23;
    --oxwash:  #f6ebe7;
    --verdigris:#3e7d6c;
    --verwash: #e9f0ec;
    --moss:    #4b6446;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--paper); color: var(--ink);
    font: 400 16px/1.55 "IBM Plex Sans", system-ui, sans-serif;
  }
  .sheet { max-width: 1120px; margin: 0 auto; padding: 2.5rem 1.25rem 5rem; }

  /* ---------- masthead ---------- */
  header.mast { border-bottom: 3px double var(--ink); padding-bottom: 1.1rem; }
  .eyebrow {
    font: 500 12.5px/1 "IBM Plex Mono", monospace; letter-spacing: .18em;
    text-transform: uppercase; color: var(--oxblood); margin: 0 0 .8rem;
  }
  h1 {
    font: 600 clamp(28px, 4.5vw, 44px)/1.08 "EB Garamond", Georgia, serif;
    margin: 0; letter-spacing: .005em;
  }
  h1 .amp { font-style: italic; font-weight: 400; color: var(--faded); padding: 0 .12em; }
  .thesis {
    font: italic 400 clamp(17px, 2.4vw, 21px)/1.5 "EB Garamond", Georgia, serif;
    color: #4c4436; margin: .9rem 0 0; max-width: 62ch;
  }
  .thesis b { font-style: normal; font-weight: 600; color: var(--ink); }
  .provenance {
    font: 400 13.5px/1.6 "IBM Plex Mono", monospace; color: var(--faded); margin-top: .9rem;
  }

  /* ---------- toolbar ---------- */
  .toolbar {
    position: sticky; top: 0; z-index: 5;
    background: color-mix(in srgb, var(--paper) 92%, transparent);
    backdrop-filter: blur(4px);
    display: flex; flex-wrap: wrap; gap: .5rem; align-items: center;
    padding: .8rem 0 .7rem; border-bottom: 1px solid var(--rule); margin-bottom: 1.2rem;
  }
  .filter {
    display: inline-flex; align-items: baseline; gap: .45em;
    font: 500 14px/1 "IBM Plex Sans", sans-serif; color: var(--ink);
    background: var(--card); border: 1px solid var(--rule); border-radius: 3px;
    padding: .42rem .6rem; cursor: pointer;
  }
  .filter .fsig { font: 600 16px/1 "EB Garamond", serif; }
  .filter .fnum { font: 500 12.5px/1 "IBM Plex Mono", monospace; color: var(--faded); }
  .filter[aria-pressed="false"] { opacity: .38; text-decoration: line-through; }
  .filter:focus-visible, #q:focus-visible {
    outline: 2px solid var(--oxblood); outline-offset: 2px;
  }
  #q {
    margin-left: auto; min-width: 220px;
    font: 400 14.5px/1 "IBM Plex Mono", monospace; color: var(--ink);
    background: var(--card); border: 1px solid var(--rule); border-radius: 3px;
    padding: .45rem .6rem;
  }
  #q::placeholder { color: var(--faded); }

  /* ---------- key ---------- */
  .key {
    font-size: 14.5px; color: var(--faded); margin: 0 0 2.2rem; max-width: 90ch;
  }
  .key .chip { margin-right: .35rem; }

  /* ---------- sections ---------- */
  section { margin: 0 0 3rem; }
  section.hidden, section.empty { display: none; }
  h2 {
    font: 600 27px/1.15 "EB Garamond", Georgia, serif; margin: 0;
    display: flex; align-items: baseline; gap: .55rem;
  }
  .siglum {
    font-weight: 400; color: var(--oxblood);
    min-width: 1.2em; text-align: center;
  }
  .count {
    font: 500 13px/1 "IBM Plex Mono", monospace; color: var(--card);
    background: var(--oxblood); border-radius: 2px; padding: .28em .5em;
    transform: rotate(-1.5deg); align-self: center;
  }
  .blurb { margin: .35rem 0 .9rem; color: var(--faded); font-size: 14.5px; max-width: 70ch; }

  /* ---------- table ---------- */
  .scroll { overflow-x: auto; background: var(--card); border: 1px solid var(--rule); border-radius: 3px; }
  table { border-collapse: collapse; width: 100%; min-width: 720px; }
  th {
    font: 500 12px/1 "IBM Plex Mono", monospace; letter-spacing: .12em;
    text-transform: uppercase; color: var(--faded); text-align: left;
    padding: .6rem .75rem; border-bottom: 1px solid var(--rule);
  }
  td { padding: .55rem .75rem; border-top: 1px solid #eceadf; vertical-align: baseline; }
  tbody tr:first-child td { border-top: 0; }
  td.lemma {
    font: 600 19.5px/1.3 "EB Garamond", Georgia, serif;
    white-space: nowrap; cursor: help;
  }
  td.yslemma { font: 400 14.5px/1.5 "IBM Plex Mono", monospace; color: var(--faded); }
  td.cats { line-height: 1.9; }

  /* ---------- chips ---------- */
  .chip {
    display: inline-block; font: 500 13px/1 "IBM Plex Mono", monospace;
    padding: .28em .5em; border-radius: 2px; white-space: nowrap;
  }
  .shared   { color: var(--moss); background: transparent; border: 1px solid #bcc8b4; }
  .equiv    { color: var(--verdigris); background: var(--verwash); border: 1px dotted var(--verdigris); }
  .amt-only { color: var(--faded); background: transparent; border: 1px dashed #c9c2b2; }
  .ys-new   {
    color: var(--oxblood); background: var(--oxwash);
    border: 1.5px solid var(--oxblood); font-weight: 500;
    letter-spacing: .04em;
  }
  .none { color: var(--faded); }

  tr[hidden] { display: none; }

  @media (prefers-reduced-motion: no-preference) {
    .filter { transition: opacity .15s ease; }
  }
  @media (max-width: 720px) {
    .sheet { padding-top: 1.5rem; }
    #q { margin-left: 0; width: 100%; }
  }
</style></head><body>
<div class="sheet">

<header class="mast">
  <p class="eyebrow">Katus · AMT-Master person-lemmas · collated against EKI ÜS (Sõnaveeb)</p>
  <h1>Sem-Cat <span class="amp">against</span> ÜS semantic types</h1>
  <p class="thesis">$thesis</p>
  <p class="provenance">source: $src · $n entries · sigla: ≠ no overlap · ~ partial · + ÜS adds · = match · ∅ no ÜS data</p>
</header>

<div class="toolbar" role="group" aria-label="Filter by agreement class">
  $filters
  <input id="q" type="search" placeholder="find lemma…" aria-label="Find lemma">
</div>

<p class="key">
  Key: <span class="chip shared">confirmed by both</span>
  <span class="chip equiv">equivalent label</span>
  <span class="chip amt-only">only in your Sem-Cat</span>
  <span class="chip ys-new">new in ÜS</span><br>
  Matching is case-insensitive; AMT subtypes (IN_ROLL:staatus) match on their base (in_roll).
  Equivalences: $equiv. Hover a lemma for its DEF_et.
</p>

$sections

</div>
<script>
  document.querySelectorAll('.filter').forEach(function (b) {
    b.addEventListener('click', function () {
      var on = b.getAttribute('aria-pressed') === 'true';
      b.setAttribute('aria-pressed', String(!on));
      document.getElementById(b.dataset.target).classList.toggle('hidden', on);
    });
  });
  var q = document.getElementById('q');
  q.addEventListener('input', function () {
    var needle = q.value.trim().toLowerCase();
    document.querySelectorAll('section').forEach(function (sec) {
      var any = false;
      sec.querySelectorAll('tbody tr').forEach(function (tr) {
        var hit = needle === '' || tr.dataset.lemma.indexOf(needle) !== -1;
        tr.hidden = !hit;
        any = any || hit;
      });
      sec.classList.toggle('empty', needle !== '' && !any);
    });
  });
</script>
</body></html>
""").substitute(
    thesis=thesis,
    src=html.escape(SRC.name),
    n=n,
    filters=filters_html,
    equiv=html.escape(equiv_str),
    sections="".join(sections_html),
)

OUT.write_text(page, encoding="utf-8")
counts = {k: len(v) for k, v in groups.items() if v}
print(f"{len(entries)} entries -> {OUT}")
print("  " + ", ".join(f"{k}: {v}" for k, v in counts.items()))

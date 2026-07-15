# Created: 2026-07-07 14-22-51
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
"""Render an AMT-Master lemma-report .md into a self-contained HTML page.

Follows the LexLex viewer design language (paper / ink / rubric red).
Understands the report generator's Markdown subset: #/##/### headings, pipe
tables (with :--- alignment), bullet lists, paragraphs, --- rules, inline
`code` / **bold** / *italic*, ✅ / ❌ marks.

Enhancements: left rail with headline stats + section TOC (counts go rubric
red only where attention is owed), proportional bars in small count tables,
live filter + sticky header on the big lemma inventory, scrollable tables.

Usage: uv run python scripts/report_md_to_html.py [path/to/report.md]
       (default: latest AMT-Master_lemma-report_*.md in Review-JSON-AMT)
Output: same folder, same name with .html
"""
import glob
import html
import os
import re
import sys

DEFAULT_GLOB = "Katus-ALUSANDMED/Review-JSON-AMT/AMT-Master_lemma-report_*.md"


# --------------------------------------------------------------- md helpers
def esc(s):
    return html.escape(s, quote=False)


def inline(s):
    """Inline markdown on an already-escaped string."""
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", s)
    s = s.replace("✅", '<span class="ok">✓</span>')
    s = s.replace("❌", '<span class="bad">✗</span>')
    return s


def slug(text):
    t = re.sub(r"[^\w\s-]", "", text.lower().replace("·", " "))
    return re.sub(r"[\s_]+", "-", t).strip("-")[:60] or "s"


def badge_heading(text, attention):
    """Trailing (N) in a heading becomes a count badge; red when it needs work."""
    m = re.match(r"^(.*?)\s*\((\d[\d,]*)\)(.*)$", text)
    if not m:
        return inline(esc(text)), None
    body, n, rest = m.group(1), m.group(2), m.group(3).strip(" —-")
    red = attention and n not in ("0",)
    cls = "count rubric" if red else "count"
    out = inline(esc(body)) + f' <span class="{cls}">{n}</span>'
    if rest:
        out += f' <span class="hrest">{inline(esc(rest))}</span>'
    return out, n


# ------------------------------------------------------------------- parser
def parse(md_lines):
    """Yield blocks: ('h1'|'h2'|'h3'|'p'|'ul'|'table'|'hr', payload)."""
    i, n = 0, len(md_lines)
    while i < n:
        line = md_lines[i].rstrip("\n")
        if not line.strip():
            i += 1
            continue
        if line.startswith("### "):
            yield "h3", line[4:]; i += 1
        elif line.startswith("## "):
            yield "h2", line[3:]; i += 1
        elif line.startswith("# "):
            yield "h1", line[2:]; i += 1
        elif line.strip() == "---":
            yield "hr", None; i += 1
        elif line.lstrip().startswith("|"):
            rows = []
            while i < n and md_lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in
                         md_lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            yield "table", rows
        elif line.lstrip().startswith("- "):
            items = []
            while i < n and md_lines[i].lstrip().startswith("- "):
                items.append(md_lines[i].lstrip()[2:].rstrip("\n"))
                i += 1
            yield "ul", items
        else:
            para = []
            while (i < n and md_lines[i].strip()
                   and not md_lines[i].lstrip().startswith(("|", "- ", "#"))
                   and md_lines[i].strip() != "---"):
                para.append(md_lines[i].strip())
                i += 1
            yield "p", " ".join(para)


# ------------------------------------------------------------ table render
def render_table(rows, tid=None):
    if len(rows) < 2:
        return ""
    head, align_row, body = rows[0], rows[1], rows[2:]
    aligns = []
    for a in align_row:
        if a.endswith(":") and a.startswith(":"):
            aligns.append("center")
        elif a.endswith(":"):
            aligns.append("right")
        else:
            aligns.append("left")
    aligns += ["left"] * (len(head) - len(aligns))

    # proportional bars: small count tables (2–3 cols, col 2 all integers)
    bar_max = 0
    if 2 <= len(head) <= 3 and body:
        try:
            vals = [int(r[1].replace(",", "")) for r in body if len(r) > 1]
            if vals and len(vals) == len(body):
                bar_max = max(vals) or 1
        except ValueError:
            bar_max = 0

    big = len(body) > 60
    h = []
    if tid and big:
        h.append(f'<div class="filterbar"><input type="search" '
                 f'data-filter="{tid}" placeholder="filter rows…" '
                 f'aria-label="filter table rows"> '
                 f'<span class="fcount" id="{tid}-count">{len(body)} rows</span></div>')
    h.append(f'<div class="tablewrap{" tall" if big else ""}">')
    h.append(f'<table{f" id={chr(34)}{tid}{chr(34)}" if tid else ""}>')
    h.append("<thead><tr>" + "".join(
        f'<th class="al-{aligns[j]}">{inline(esc(c))}</th>'
        for j, c in enumerate(head)) + "</tr></thead><tbody>")
    for r in body:
        tds = []
        for j, c in enumerate(r):
            v = inline(esc(c))
            if bar_max and j == 1:
                w = 100 * int(c.replace(",", "")) / bar_max
                v = (f'<span class="barcell"><span class="bar" '
                     f'style="width:{w:.1f}%"></span><span class="barval">{v}'
                     f'</span></span>')
            tds.append(f'<td class="al-{aligns[j] if j < len(aligns) else "left"}">{v}</td>')
        h.append("<tr>" + "".join(tds) + "</tr>")
    h.append("</tbody></table></div>")
    return "\n".join(h)


# --------------------------------------------------------------------- main
def build(md_path):
    out_path = os.path.splitext(md_path)[0] + ".html"
    lines = open(md_path, encoding="utf-8").read().splitlines()

    title, toc, sections = "AMT-Master review", [], []
    cur = None                # current section dict: {id,title,html,level}
    in_omissions = False
    table_no = 0

    def close():
        if cur:
            sections.append(cur)

    stats = {}                # headline numbers for the rail
    text_all = "\n".join(lines)
    m = re.search(r"\*\*(\d+)\*\* concepts \(lemmas\)", text_all)
    if m:
        stats["concepts"] = m.group(1)
    m = re.search(r"Attested source-cells: \*\*(\d+)\*\* of \d+ "
                  r"\(([\d.]+%)\); linked \*\*(\d+)\*\* \(([\d.]+%)[^)]*\); "
                  r"attested-but-unlinked \*\*(\d+)\*\*", text_all)
    if m:
        stats.update(att=m.group(1), attpct=m.group(2), lnk=m.group(3),
                     lnkpct=m.group(4), tail=m.group(5))
    stats["sync_ok"] = text_all.count("| ✅ |")
    stats["sync_bad"] = text_all.count("| ❌ |")

    for kind, payload in parse(lines):
        if kind == "h1":
            title = payload
            continue
        if kind == "h2":
            close()
            in_omissions = "Omissions" in payload
            txt, _ = badge_heading(payload, False)
            sid = "s-" + slug(payload)
            cur = dict(id=sid, title=txt, html=[], subs=[])
            toc.append(dict(id=sid, title=txt, subs=[]))
            continue
        if cur is None:                       # preamble before first h2
            cur = dict(id="s-head", title=None, html=[], subs=[])
            toc.append(dict(id="s-head", title="Report inputs", subs=[]))
        if kind == "h3":
            txt, n = badge_heading(payload, in_omissions)
            sid = "s-" + slug(payload)
            cur["html"].append(f'<h3 id="{sid}">{txt}</h3>')
            toc[-1]["subs"].append(dict(id=sid, title=txt))
        elif kind == "table":
            table_no += 1
            cur["html"].append(render_table(payload, tid=f"t{table_no}"))
        elif kind == "ul":
            cur["html"].append("<ul>" + "".join(
                f"<li>{inline(esc(it))}</li>" for it in payload) + "</ul>")
        elif kind == "p":
            cur["html"].append(f"<p>{inline(esc(payload))}</p>")
        elif kind == "hr":
            cur["html"].append("<hr>")
    close()

    date = title.split("—")[-1].strip() if "—" in title else ""
    rail_stats = ""
    if stats.get("concepts"):
        tail = stats.get("tail", "0")
        tail_cls = "rubric" if tail != "0" else ""
        sync = (f'{stats["sync_ok"]}/{stats["sync_ok"] + stats["sync_bad"]}')
        sync_cls = "rubric" if stats["sync_bad"] else ""
        rail_stats = f"""
<div class="stats">
  <div class="stat"><span class="num">{stats["concepts"]}</span><span class="lab">concepts</span></div>
  <div class="stat"><span class="num">{stats.get("att","–")}</span><span class="lab">attested cells</span></div>
  <div class="stat"><span class="num">{stats.get("lnkpct","–")}</span><span class="lab">linked</span></div>
  <div class="stat"><span class="num {tail_cls}">{tail}</span><span class="lab">unlinked tail</span></div>
  <div class="stat"><span class="num {sync_cls}">{sync}</span><span class="lab">sync checks</span></div>
</div>"""

    toc_html = "<nav class=\"toc\">" + "".join(
        f'<a class="t2" href="#{t["id"]}">{t["title"]}</a>' + "".join(
            f'<a class="t3" href="#{s["id"]}">{s["title"]}</a>'
            for s in t["subs"])
        for t in toc) + "</nav>"

    body_html = "\n".join(
        f'<section class="panel" id="{s["id"]}">'
        + (f'<h2>{s["title"]}</h2>' if s["title"] else "")
        + "\n".join(s["html"]) + "</section>"
        for s in sections)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  :root{{
    --paper:#f2edde; --panel:#fbf9f0; --rail:#e9e3d1;
    --ink:#2a2215; --ink-strong:#171009;
    --line:#cec5ab; --line-soft:#e3dcc7; --hover:#e1dac2;
    --rubric:#96271f; --sepia:#867655; --verdigris:#3a6350;
    --serif:'Iowan Old Style','Palatino Linotype',Palatino,'Book Antiqua',Georgia,serif;
    --display:'Hoefler Text',Baskerville,'Iowan Old Style',Georgia,serif;
    --sans:'Avenir Next',Seravek,'Gill Sans','Helvetica Neue',Arial,sans-serif;
  }}
  body{{ font-family:var(--serif); background:var(--paper); color:var(--ink);
    font-size:15.5px; line-height:1.55; -webkit-font-smoothing:antialiased; }}
  header{{ border-bottom:3px double var(--line); padding:16px 24px 12px;
    display:flex; align-items:baseline; gap:18px; flex-wrap:wrap; }}
  h1{{ font-family:var(--display); font-size:1.55em; font-weight:600;
    color:var(--ink-strong); }}
  .legend{{ font-family:var(--sans); font-size:.72em; color:var(--sepia);
    margin-left:auto; }}
  .legend .dotmark{{ color:var(--rubric); font-weight:700; }}
  .wrap{{ display:flex; align-items:flex-start; }}
  aside{{ width:270px; min-width:270px; background:var(--rail);
    border-right:1px solid var(--line); position:sticky; top:0;
    max-height:100vh; overflow-y:auto; padding:16px 0 30px; }}
  .stats{{ padding:2px 18px 12px; border-bottom:1px solid var(--line);
    display:flex; flex-wrap:wrap; gap:10px 22px; }}
  .stat .num{{ font-family:var(--display); font-size:1.5em; font-weight:600;
    color:var(--ink-strong); display:block; line-height:1.15; }}
  .stat .num.rubric{{ color:var(--rubric); }}
  .stat .lab{{ font-family:var(--sans); font-size:.62em; text-transform:uppercase;
    letter-spacing:.1em; color:var(--sepia); }}
  .toc{{ display:block; padding:12px 0; }}
  .toc a{{ display:block; text-decoration:none; color:var(--ink);
    font-family:var(--sans); padding:4px 18px; border-left:3px solid transparent; }}
  .toc a:hover{{ background:var(--hover); }}
  .toc a.t2{{ font-size:.78em; letter-spacing:.04em; font-weight:600;
    text-transform:uppercase; margin-top:6px; }}
  .toc a.t3{{ font-size:.76em; color:var(--sepia); padding-left:30px; }}
  .toc .count{{ font-family:var(--sans); font-size:.85em; color:var(--sepia);
    background:var(--paper); border:1px solid var(--line); border-radius:9px;
    padding:0 7px; margin-left:5px; }}
  .toc .count.rubric, main .count.rubric{{ color:#faf6ec; background:var(--rubric);
    border-color:var(--rubric); }}
  main{{ flex:1; min-width:0; padding:22px 28px 80px; max-width:1080px;
    margin-inline:auto; }}
  .hrest{{ font-family:var(--serif); font-size:.9em; font-weight:400;
    font-style:italic; color:var(--sepia); text-transform:none;
    letter-spacing:0; }}
  .panel{{ background:var(--panel); border:1px solid var(--line);
    border-top:3px solid var(--ink-strong); box-shadow:0 2px 8px rgba(42,34,21,.05);
    padding:16px 22px 20px; margin:0 0 18px; }}
  h2{{ font-family:var(--display); font-size:1.3em; font-weight:600;
    color:var(--ink-strong); padding-bottom:8px; margin-bottom:10px;
    border-bottom:1px solid var(--line-soft); }}
  h3{{ font-family:var(--sans); font-size:.8em; text-transform:uppercase;
    letter-spacing:.1em; color:var(--ink-strong); margin:22px 0 8px; }}
  main .count{{ font-family:var(--sans); font-size:.85em; color:var(--sepia);
    background:var(--paper); border:1px solid var(--line); border-radius:9px;
    padding:0 7px; margin-left:6px; letter-spacing:0; }}
  p{{ margin:8px 0; max-width:75ch; }}
  ul{{ margin:8px 0 8px 22px; max-width:75ch; }}
  li{{ margin:3px 0; }}
  hr{{ border:0; border-top:1px solid var(--line-soft); margin:16px 0; }}
  code{{ font-family:ui-monospace,'SF Mono',Menlo,monospace; font-size:.86em;
    background:var(--paper); border:1px solid var(--line-soft);
    border-radius:3px; padding:0 4px; }}
  strong{{ color:var(--ink-strong); }}
  td strong{{ color:var(--rubric); }}
  em{{ color:inherit; }}
  .ok{{ color:var(--verdigris); font-weight:700; }}
  .bad{{ color:var(--rubric); font-weight:700; }}
  .tablewrap{{ overflow:auto; border:1px solid var(--line-soft); margin:10px 0; }}
  .tablewrap.tall{{ max-height:72vh; }}
  table{{ border-collapse:collapse; width:100%; font-size:.92em; }}
  th{{ font-family:var(--sans); font-size:.72em; text-transform:uppercase;
    letter-spacing:.08em; color:var(--sepia); font-weight:600;
    background:var(--panel); padding:7px 12px; border-bottom:2px solid var(--ink-strong);
    position:sticky; top:0; }}
  td{{ padding:5px 12px; border-bottom:1px solid var(--line-soft);
    vertical-align:top; }}
  tbody tr:hover{{ background:var(--hover); }}
  .al-right{{ text-align:right; }} .al-center{{ text-align:center; }}
  .barcell{{ display:flex; align-items:center; gap:8px; min-width:130px; }}
  .bar{{ display:inline-block; height:9px; background:var(--sepia);
    opacity:.55; min-width:2px; }}
  .barval{{ font-variant-numeric:tabular-nums; }}
  .filterbar{{ display:flex; align-items:center; gap:12px; margin:12px 0 0; }}
  .filterbar input{{ font-family:var(--serif); font-style:italic; font-size:.92em;
    padding:5px 11px; width:260px; border:1px solid var(--line);
    border-radius:2px; background:var(--panel); color:var(--ink); }}
  .filterbar input:focus-visible{{ outline:2px solid var(--ink-strong); outline-offset:1px; }}
  .fcount{{ font-family:var(--sans); font-size:.72em; color:var(--sepia);
    letter-spacing:.05em; }}
  :is(a,input):focus-visible{{ outline:2px solid var(--ink-strong); outline-offset:1px; }}
  @media (max-width:880px){{
    .wrap{{ display:block; }}
    aside{{ width:100%; min-width:0; position:static; max-height:none;
      border-right:0; border-bottom:1px solid var(--line); }}
    main{{ padding:16px 14px 60px; }}
  }}
  @media (prefers-reduced-motion:reduce){{ *{{ scroll-behavior:auto !important; }} }}
  html{{ scroll-behavior:smooth; }}
</style>
</head>
<body>
<header>
  <h1>{esc(title)}</h1>
  <span class="legend"><span class="dotmark">●</span> red marks items that still need attention</span>
</header>
<div class="wrap">
  <aside>
    {rail_stats}
    {toc_html}
  </aside>
  <main>
{body_html}
  </main>
</div>
<script>
document.querySelectorAll('input[data-filter]').forEach(inp=>{{
  const t=document.getElementById(inp.dataset.filter);
  const cnt=document.getElementById(inp.dataset.filter+'-count');
  const rows=[...t.tBodies[0].rows];
  inp.addEventListener('input',()=>{{
    const q=inp.value.trim().toLowerCase();
    let n=0;
    rows.forEach(r=>{{
      const show=!q||r.textContent.toLowerCase().includes(q);
      r.style.display=show?'':'none'; if(show)n++;
    }});
    if(cnt) cnt.textContent=n+' rows';
  }});
}});
</script>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {out_path} ({os.path.getsize(out_path)/1024:.0f} kB, "
          f"{len(sections)} sections, {table_no} tables)")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        cands = sorted(glob.glob(DEFAULT_GLOB))
        if not cands:
            sys.exit(f"no report found matching {DEFAULT_GLOB}")
        path = cands[-1]
    build(path)

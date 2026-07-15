# Created: 2026-07-07 14-33-47
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
"""Render the LexLex documentation (global-view/README.md) as a
self-contained HTML page in the LexLex design language.

Layout echoes the viewer itself: paper/ink/sepia palette with rubric-red
accents (#FF0000), red top rule, the LexLex masthead with a stamped tagline,
a sticky left rail holding the table of contents, and each numbered section
rendered as a full-width band whose spine carries the section number — the
same ledger metaphor as the viewer's workspace.

Markdown subset understood: #/##/### headings, pipe tables (with :---
alignment), fenced ``` code blocks, unordered lists (wrapped continuation
lines; nested ordered items), top-level ordered lists, paragraphs, --- rules,
inline `code` / **bold** / *italic*.

Usage:  uv run python scripts/build_doc.py
Output: Katus-ALUSANDMED/global-view/LexLex-doc.html
"""
import datetime
import html
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from build_viewer import VERSION

SRC = "Katus-ALUSANDMED/global-view/README.md"
OUT = "Katus-ALUSANDMED/global-view/LexLex-doc.html"


def esc(s):
    return html.escape(s, quote=False)


def inline(s):
    """Inline markdown on an already-escaped string."""
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", s)
    return s


def slug(text):
    t = re.sub(r"[^\w\s-]", "", text.lower().replace("·", " "))
    return re.sub(r"[\s_]+", "-", t).strip("-")[:60] or "s"


UL_RE = re.compile(r"^- (.*)$")
OL_RE = re.compile(r"^(\d+)\.\s+(.*)$")
NESTED_OL_RE = re.compile(r"^\s{2,}(\d+)\.\s+(.*)$")
CONT_RE = re.compile(r"^\s{2,}(\S.*)$")


def parse(lines):
    """Yield blocks: (kind, payload)."""
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].rstrip("\n")
        if not line.strip():
            i += 1
            continue
        if line.startswith("```"):
            code = []
            i += 1
            while i < n and not lines[i].startswith("```"):
                code.append(lines[i].rstrip("\n"))
                i += 1
            i += 1                                    # closing fence
            yield "code", "\n".join(code)
        elif line.startswith("### "):
            yield "h3", line[4:]; i += 1
        elif line.startswith("## "):
            yield "h2", line[3:]; i += 1
        elif line.startswith("# "):
            yield "h1", line[2:]; i += 1
        elif line.strip() in ("---", "***"):
            yield "hr", None; i += 1
        elif line.lstrip().startswith("|"):
            rows = []
            while i < n and lines[i].lstrip().startswith("|"):
                rows.append([c.strip() for c in
                             lines[i].strip().strip("|").split("|")])
                i += 1
            yield "table", rows
        elif UL_RE.match(line):
            # unordered list; items may wrap and may hold a nested ordered list
            items = []                                # each: {"text":…, "subs":[…]}
            while i < n and lines[i].strip():
                l = lines[i].rstrip("\n")
                m = UL_RE.match(l)
                mo = NESTED_OL_RE.match(l)
                mc = CONT_RE.match(l)
                if m:
                    items.append({"text": m.group(1), "subs": []})
                elif mo and items:
                    items[-1]["subs"].append(mo.group(2))
                elif mc and items:
                    if items[-1]["subs"]:
                        items[-1]["subs"][-1] += " " + mc.group(1)
                    else:
                        items[-1]["text"] += " " + mc.group(1)
                else:
                    break
                i += 1
            yield "ul", items
        elif OL_RE.match(line):
            items = []
            while i < n and lines[i].strip():
                l = lines[i].rstrip("\n")
                m = OL_RE.match(l)
                mc = CONT_RE.match(l)
                if m:
                    items.append(m.group(2))
                elif mc and items:
                    items[-1] += " " + mc.group(1)
                else:
                    break
                i += 1
            yield "ol", items
        else:
            para = []
            while (i < n and lines[i].strip()
                   and not lines[i].lstrip().startswith(("|", "- ", "#", "```"))
                   and not OL_RE.match(lines[i])
                   and lines[i].strip() not in ("---", "***")):
                para.append(lines[i].strip())
                i += 1
            yield "p", " ".join(para)


def render_table(rows):
    if len(rows) < 2:
        return ""
    head, align_row, body = rows[0], rows[1], rows[2:]
    aligns = []
    for a in align_row:
        if a.startswith(":") and a.endswith(":"):
            aligns.append("center")
        elif a.endswith(":"):
            aligns.append("right")
        else:
            aligns.append("left")
    aligns += ["left"] * (len(head) - len(aligns))
    h = ['<div class="tablewrap"><table>']
    h.append("<thead><tr>" + "".join(
        f'<th class="al-{aligns[j]}">{inline(esc(c))}</th>'
        for j, c in enumerate(head)) + "</tr></thead><tbody>")
    for r in body:
        h.append("<tr>" + "".join(
            f'<td class="al-{aligns[j] if j < len(aligns) else "left"}">'
            f'{inline(esc(c))}</td>' for j, c in enumerate(r)) + "</tr>")
    h.append("</tbody></table></div>")
    return "\n".join(h)


def render_ul(items):
    h = ["<ul>"]
    for it in items:
        h.append(f"<li>{inline(esc(it['text']))}")
        if it["subs"]:
            h.append("<ol>" + "".join(
                f"<li>{inline(esc(s))}</li>" for s in it["subs"]) + "</ol>")
        h.append("</li>")
    h.append("</ul>")
    return "".join(h)


def build():
    lines = open(SRC, encoding="utf-8").read().splitlines()

    intro = []            # blocks before the first h2 (rendered as lead)
    sections = []         # {"id","num","title","html"}
    toc = []
    cur = None

    def sink():
        return cur["html"] if cur else intro

    for kind, payload in parse(lines):
        if kind == "h1":
            continue                                   # masthead replaces it
        if kind == "h2":
            m = re.match(r"^(\d+)\.\s+(.*)$", payload)
            num, title = (m.group(1), m.group(2)) if m else ("·", payload)
            sid = "s-" + slug(payload)
            cur = {"id": sid, "num": num, "title": title, "html": []}
            sections.append(cur)
            toc.append({"id": sid, "num": num, "title": title, "subs": []})
            continue
        if kind == "h3":
            sid = "s-" + slug(payload)
            sink().append(f'<h3 id="{sid}">{inline(esc(payload))}</h3>')
            if toc:
                toc[-1]["subs"].append({"id": sid, "title": payload})
        elif kind == "code":
            sink().append(f"<pre><code>{esc(payload)}</code></pre>")
        elif kind == "table":
            sink().append(render_table(payload))
        elif kind == "ul":
            sink().append(render_ul(payload))
        elif kind == "ol":
            sink().append("<ol>" + "".join(
                f"<li>{inline(esc(it))}</li>" for it in payload) + "</ol>")
        elif kind == "p":
            sink().append(f"<p>{inline(esc(payload))}</p>")
        elif kind == "hr":
            if cur:                                    # section breaks come from h2
                cur["html"].append("<hr>")

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    tagline = (f"Documentation &middot;&middot;&middot; {VERSION} "
               f"&middot;&middot;&middot; {stamp}")

    toc_html = '<nav class="toc">' + "".join(
        f'<a class="t2" href="#{t["id"]}"><span class="tnum">{t["num"]}</span>'
        f'{inline(esc(t["title"]))}</a>' + "".join(
            f'<a class="t3" href="#{s["id"]}">{inline(esc(s["title"]))}</a>'
            for s in t["subs"])
        for t in toc) + "</nav>"

    bands = "\n".join(
        f'<section class="band" id="{s["id"]}">'
        f'<div class="spine">&sect;&nbsp;{s["num"]}</div>'
        f'<div class="bandmain"><h2>{inline(esc(s["title"]))}</h2>\n'
        + "\n".join(s["html"]) + "</div></section>"
        for s in sections)

    intro_html = "\n".join(intro)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LexLex — Lexicon Lexicorum Esthonicorum · documentation</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  :root{{
    --paper:#f2edde; --panel:#fbf9f0; --rail:#e9e3d1;
    --ink:#2a2215; --ink-strong:#171009;
    --line:#cec5ab; --line-soft:#e3dcc7; --hover:#e1dac2;
    --rubric:#FF0000; --sepia:#867655; --verdigris:#3a6350;
    --serif:'Iowan Old Style','Palatino Linotype',Palatino,'Book Antiqua',Georgia,serif;
    --display:'Hoefler Text',Baskerville,'Iowan Old Style',Georgia,serif;
    --sans:'Avenir Next',Seravek,'Gill Sans','Helvetica Neue',Arial,sans-serif;
  }}
  body{{ font-family:var(--serif); background:var(--paper); color:var(--ink);
    font-size:16px; line-height:1.55; -webkit-font-smoothing:antialiased; }}
  header{{ border-top:2px solid var(--rubric); border-bottom:3px double var(--line);
    padding:14px 22px 11px; display:flex; align-items:baseline; gap:18px;
    flex-wrap:wrap; }}
  h1{{ font-family:var(--display); font-size:1.5em; font-weight:600;
    color:var(--ink-strong); letter-spacing:.01em; }}
  h1 .rdash{{ color:var(--rubric); }}
  .tagline{{ font-family:var(--sans); font-size:.68em; letter-spacing:.14em;
    color:var(--sepia); white-space:nowrap; }}
  .openbtn{{ margin-left:auto; font-family:var(--sans); font-size:.72em;
    letter-spacing:.06em; text-transform:uppercase; text-decoration:none;
    border:1px solid var(--ink); color:var(--ink); border-radius:2px;
    padding:5px 12px; transition:background .12s ease, color .12s ease; }}
  .openbtn:hover{{ background:var(--ink-strong); color:var(--paper); }}
  .wrap{{ display:flex; align-items:flex-start; }}
  aside{{ width:290px; min-width:290px; background:var(--rail);
    border-right:1px solid var(--line); position:sticky; top:0;
    max-height:100vh; overflow-y:auto; padding:12px 0 30px; }}
  .toc a{{ display:block; text-decoration:none; color:var(--ink);
    font-family:var(--sans); padding:5px 18px; border-left:3px solid transparent; }}
  .toc a:hover{{ background:var(--hover); border-left-color:var(--rubric); }}
  .toc a.t2{{ font-size:.78em; letter-spacing:.05em; font-weight:600;
    text-transform:uppercase; margin-top:8px; }}
  .toc .tnum{{ display:inline-block; width:1.6em; color:var(--sepia);
    font-weight:400; }}
  .toc a.t3{{ font-size:.76em; color:var(--sepia); padding-left:44px;
    text-transform:none; }}
  main{{ flex:1; min-width:0; padding:20px 26px 90px; max-width:1000px; }}
  .lead{{ max-width:72ch; margin-bottom:20px; }}
  .lead p{{ margin:9px 0; }}
  .band{{ display:flex; background:var(--panel); border:1px solid var(--line);
    border-left:4px solid var(--line); box-shadow:0 2px 8px rgba(42,34,21,.05);
    margin:0 0 14px; }}
  .spine{{ width:76px; min-width:76px; padding:15px 0 12px 16px;
    border-right:1px solid var(--line-soft); font-family:var(--display);
    font-size:1.15em; color:var(--sepia); }}
  .bandmain{{ flex:1; min-width:0; padding:12px 20px 16px; }}
  h2{{ font-family:var(--display); font-size:1.3em; font-weight:600;
    color:var(--ink-strong); border-bottom:1px solid var(--line-soft);
    padding-bottom:7px; margin-bottom:8px; }}
  h3{{ font-family:var(--sans); font-size:.74em; text-transform:uppercase;
    letter-spacing:.12em; color:var(--ink-strong); margin:20px 0 6px; }}
  p{{ margin:8px 0; max-width:72ch; }}
  ul, ol{{ margin:8px 0 8px 22px; max-width:70ch; }}
  li{{ margin:4px 0; }}
  li ol{{ margin-top:6px; }}
  hr{{ border:0; border-top:1px solid var(--line-soft); margin:14px 0; }}
  code{{ font-family:ui-monospace,'SF Mono',Menlo,monospace; font-size:.86em;
    background:var(--paper); border:1px solid var(--line-soft);
    border-radius:3px; padding:0 4px; }}
  pre{{ background:var(--paper); border:1px solid var(--line-soft);
    padding:12px 16px; margin:10px 0; overflow-x:auto; max-width:100%; }}
  pre code{{ border:0; padding:0; background:transparent; font-size:.82em;
    line-height:1.45; }}
  strong{{ color:var(--ink-strong); }}
  em{{ color:inherit; }}
  .tablewrap{{ overflow-x:auto; border:1px solid var(--line-soft); margin:10px 0; }}
  table{{ border-collapse:collapse; width:100%; font-size:.92em; }}
  th{{ font-family:var(--sans); font-size:.72em; text-transform:uppercase;
    letter-spacing:.08em; color:var(--sepia); font-weight:600; text-align:left;
    background:var(--panel); padding:7px 12px;
    border-bottom:2px solid var(--ink-strong); }}
  td{{ padding:5px 12px; border-bottom:1px solid var(--line-soft);
    vertical-align:top; }}
  tbody tr:hover{{ background:var(--hover); }}
  .al-right{{ text-align:right; }} .al-center{{ text-align:center; }}
  :is(a,.openbtn):focus-visible{{ outline:2px solid var(--ink-strong);
    outline-offset:1px; }}
  ::-webkit-scrollbar{{ width:10px; height:10px; }}
  ::-webkit-scrollbar-thumb{{ background:var(--line); border-radius:5px; }}
  ::-webkit-scrollbar-track{{ background:transparent; }}
  html{{ scroll-behavior:smooth; }}
  @media (max-width:880px){{
    .wrap{{ display:block; }}
    aside{{ width:100%; min-width:0; position:static; max-height:none;
      border-right:0; border-bottom:1px solid var(--line); }}
    main{{ padding:16px 14px 60px; }}
    .spine{{ width:52px; min-width:52px; padding-left:12px; }}
  }}
  @media (prefers-reduced-motion:reduce){{
    html{{ scroll-behavior:auto; }} *{{ transition:none !important; }}
  }}
</style>
</head>
<body>
<header>
  <h1>LexLex <span class="rdash">&mdash;</span> Lexicon Lexicorum Esthonicorum</h1>
  <span class="tagline">{tagline}</span>
  <a class="openbtn" href="LexLex.html">Open LexLex &rarr;</a>
</header>
<div class="wrap">
  <aside>
    {toc_html}
  </aside>
  <main>
    <div class="lead">
{intro_html}
    </div>
{bands}
  </main>
</div>
</body>
</html>"""

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {OUT} ({os.path.getsize(OUT)/1024:.0f} kB, "
          f"{len(sections)} sections)")


if __name__ == "__main__":
    build()

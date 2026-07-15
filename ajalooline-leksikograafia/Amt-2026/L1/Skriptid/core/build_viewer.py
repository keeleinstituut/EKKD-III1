# Created: 2026-07-13 14-44-50
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
# Co-Authored-By: Claude Fable 5
"""Generate LexLex — Lexicon Lexicorum Esthonicorum, the self-contained
HTML dictionary portal for the Katus dataset.

One portable file (data embedded, vanilla JS, works from file://) offering:
  - Chronological-ledger workspace (canonical since v5, adopted from the v3
    experiment): every opened entry is a full-width band, master concept
    first, editions stacked below sorted by source year (1637→1780)
  - Per-dictionary browsing/search
  - Red attention markers on entries with "???" or attested-but-unlinked cells
  - A stubbed PDF-page hook for the future pop-up feature
"""
import datetime
import json
import os
import shutil
import sys

JD = "Katus-ALUSANDMED/json-all"
OUT = "Katus-ALUSANDMED/global-view/LexLex.html"
VERSION = "v9"  # v9 2026-07-13: 870 records after MJ deletions; DQ fix run + ÜS Sem-Cat fold-in + Sugu policy; comments extracted to archive
SOURCES = ["Stahl-1637", "Gutslaff-1648", "Göseken-1660",
           "Vestring-17XX", "Helle-1732", "Hupel-1780-est-ger"]
PLACE = {"---", "???", "NULL", "", " "}            # not a real attestation
EMPTY = {"NULL", ""}                                # drop when embedding


def has_q(v):
    return isinstance(v, str) and "???" in v


def strip_entry(e):
    """Drop empty scalar fields and empty lists; keep id always."""
    out = {}
    for k, v in e.items():
        if isinstance(v, str):
            if v in EMPTY:
                continue
            out[k] = v
        elif isinstance(v, list):
            if v:
                out[k] = v
        else:
            out[k] = v
    return out


def scan_q(e):
    """field labels (or 'mwu') whose value contains ???."""
    flags = []
    for k, v in e.items():
        if isinstance(v, str) and "???" in v:
            flags.append(k)
        elif isinstance(v, list):
            for m in v:
                if isinstance(m, dict) and any(has_q(x) for x in m.values()):
                    flags.append("mwu")
                    break
    return flags


def build(variants=()):
    master = json.load(open(f"{JD}/AMT-Master_annotated.json", encoding="utf-8"))["AMT-Master"]
    editions = {s: json.load(open(f"{JD}/{s}.json", encoding="utf-8"))[s] for s in SOURCES}

    # ---- master concepts (keep full, add flags) ----
    m_out = []
    for c in master:
        q = scan_q(c)
        unlinked = []
        for s in SOURCES:
            et = c.get(f"{s}-et", "NULL")
            if not isinstance(et, str):          # guard against stray [] / non-string cells
                et = "NULL"
            if et not in PLACE and not c.get(f"{s}-id"):
                unlinked.append(s)
        obj = strip_entry(c)
        if q:
            obj["_q"] = q
        if unlinked:
            obj["_u"] = unlinked
        m_out.append(obj)

    # ---- editions (strip empties, add ??? flag) ----
    e_out = {}
    for s in SOURCES:
        rows = []
        for e in editions[s]:
            q = scan_q(e)
            obj = strip_entry(e)
            if q:
                obj["_q"] = q
            rows.append(obj)
        e_out[s] = rows

    data = {"sources": SOURCES, "master": m_out, "ed": e_out}
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    js = js.replace("<", "\\u003c")                 # safe inside <script>

    html = TEMPLATE.replace("/*__DATA__*/", js)
    html = html.replace("__VER__", VERSION)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    html = html.replace("__TAG__", f"Prototype &middot;&middot;&middot; {VERSION} "
                                   f"&middot;&middot;&middot; {stamp}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    n_mflag = sum(1 for c in m_out if "_q" in c or "_u" in c)
    n_eflag = sum(1 for s in SOURCES for e in e_out[s] if "_q" in e)
    print(f"wrote {OUT}")
    print(f"  size: {os.path.getsize(OUT) / 1024 / 1024:.1f} MB")
    print(f"  master concepts: {len(m_out)} ({n_mflag} flagged red)")
    for s in SOURCES:
        print(f"  {s:22} {len(e_out[s]):6} entries")
    print(f"  edition entries flagged red (???): {n_eflag}")

    vcopy = OUT.replace(".html", f"_{VERSION}.html")
    shutil.copyfile(OUT, vcopy)
    print(f"wrote {vcopy} (versioned snapshot)")

    # ---- experimental workspace-layout variants (override block appended) ----
    for v in variants:
        vhtml = html.replace("LexLex — Lexicon Lexicorum Esthonicorum",
                             f"LexLex — Lexicon Lexicorum Esthonicorum · {v}")
        vhtml = vhtml.replace("</body>", VARIANTS[v] + "\n</body>")
        vout = OUT.replace(".html", f"_{v}.html")
        with open(vout, "w", encoding="utf-8") as f:
            f.write(vhtml)
        print(f"wrote {vout} ({os.path.getsize(vout) / 1024 / 1024:.1f} MB)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="et">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LexLex — Lexicon Lexicorum Esthonicorum · __VER__</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  :root{
    --paper:#f2edde; --panel:#fbf9f0; --rail:#e9e3d1;
    --ink:#2a2215; --ink-strong:#171009;
    --line:#cec5ab; --line-soft:#e3dcc7; --hover:#e1dac2; --sel:#d9d1b6;
    --rubric:#FF0000; --sepia:#867655; --verdigris:#3a6350;
    --gram:#5c5140; --dialect:#7d3554;
    --serif:'Iowan Old Style','Palatino Linotype',Palatino,'Book Antiqua',Georgia,serif;
    --display:'Hoefler Text',Baskerville,'Iowan Old Style',Georgia,serif;
    --sans:'Avenir Next',Seravek,'Gill Sans','Helvetica Neue',Arial,sans-serif;
  }
  html,body{ height:100%; }
  body{ font-family:var(--serif); background:var(--paper); color:var(--ink);
    font-size:16px; line-height:1.55; display:flex; flex-direction:column;
    overflow:hidden; -webkit-font-smoothing:antialiased; }
  header{ background:var(--paper); border-top:2px solid var(--rubric);
    border-bottom:3px double var(--line); padding:14px 20px 11px; }
  .htop{ display:flex; align-items:baseline; gap:18px; flex-wrap:wrap; }
  h1{ font-family:var(--display); font-size:1.5em; font-weight:600;
    color:var(--ink-strong); letter-spacing:.01em; }
  h1 .rdash{ color:var(--rubric); }
  .tagline{ font-family:var(--sans); font-size:.68em; font-weight:400;
    letter-spacing:.14em; color:var(--sepia); white-space:nowrap; }
  .legend{ font-family:var(--sans); font-size:.72em; letter-spacing:.02em;
    color:var(--sepia); margin-left:auto; }
  .dot{ color:var(--rubric); font-weight:700; }
  nav{ display:flex; gap:7px; margin-top:11px; flex-wrap:wrap; align-items:center;}
  nav button{ font-family:var(--sans); font-size:.72em; text-transform:uppercase;
    letter-spacing:.09em; color:var(--ink); background:transparent;
    border:1px solid var(--line); border-radius:2px; padding:6px 12px; cursor:pointer;
    transition:background .12s ease, color .12s ease; }
  nav button:hover{ background:var(--hover); }
  nav button.active{ background:var(--ink-strong); color:var(--paper); border-color:var(--ink-strong);}
  nav button.flagtog{ margin-left:8px; }
  nav button.flagtog.on{ background:transparent; color:var(--rubric); border-color:var(--rubric); font-weight:600; }
  nav button.hidetog.on{ background:var(--ink-strong); color:var(--paper); border-color:var(--ink-strong); }
  #search{ font-family:var(--serif); font-size:.95em; font-style:italic; padding:6px 12px;
    width:250px; border:1px solid var(--line); border-radius:2px;
    background:var(--panel); color:var(--ink); }
  #search::placeholder{ color:var(--sepia); }
  nav select{ font-family:var(--sans); font-size:.75em; padding:6px 6px; color:var(--ink);
    background:var(--panel); border:1px solid var(--line); border-radius:2px; cursor:pointer; }
  :is(nav button, #search, nav select, .pager button, .pager input):focus-visible{
    outline:2px solid var(--ink-strong); outline-offset:1px; }
  .badq{ outline:2px solid var(--rubric); }
  .layout{ flex:1; min-height:0; display:flex; }
  .browser{ width:330px; min-width:330px; background:var(--rail);
    border-right:1px solid var(--line); overflow-y:auto; }
  .bcount{ font-family:var(--sans); font-size:.7em; letter-spacing:.06em;
    text-transform:uppercase; color:var(--sepia); padding:10px 16px;
    border-bottom:1px solid var(--line); }
  .item{ padding:9px 16px 9px 13px; border-bottom:1px solid var(--line-soft);
    border-left:3px solid transparent; cursor:pointer; transition:background .12s ease; }
  .item:hover{ background:var(--hover); }
  .item.sel{ background:var(--sel); border-left-color:var(--rubric); }
  .item .code{ font-weight:600; color:var(--ink-strong); }
  .item .sub{ font-size:.82em; color:var(--sepia); font-style:italic;
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .pager{ display:flex; justify-content:space-between; align-items:center;
    font-family:var(--sans); padding:10px 16px; font-size:.78em; color:var(--sepia); }
  .pager button{ font-family:var(--sans); cursor:pointer; border:1px solid var(--line);
    background:var(--panel); color:var(--ink); border-radius:2px; padding:2px 9px; }
  .pager button:hover{ background:var(--hover); }
  .pager button:disabled{ opacity:.35; cursor:default; }
  .pager .grp{ display:flex; align-items:center; gap:4px; }
  .pager input.jump{ width:48px; font-family:var(--sans); font-size:1em; text-align:center;
    border:1px solid var(--line); border-radius:2px; padding:1px 3px;
    background:var(--panel); color:var(--ink); }
  .workspace{ flex:1; display:block; overflow-y:auto; padding:16px 22px; }
  .band{ display:flex; background:var(--panel); border:1px solid var(--line);
    border-left:4px solid var(--line); box-shadow:0 2px 8px rgba(42,34,21,.05);
    margin:0 0 12px; }
  .band.master{ border-left-color:var(--ink-strong); }
  .band.flag{ border-left-color:var(--rubric); }
  .spine{ width:108px; min-width:108px; padding:13px 0 12px 14px;
    border-right:1px solid var(--line-soft); font-family:var(--sans); font-size:.72em;
    text-transform:uppercase; letter-spacing:.08em; color:var(--ink); line-height:1.5; }
  .spine .syear{ display:block; color:var(--sepia); letter-spacing:.06em; }
  .spine .src.master{ font-size:.85em; letter-spacing:.14em; color:var(--paper);
    background:var(--ink-strong); padding:2px 7px; border-radius:2px; }
  .bandmain{ flex:1; min-width:0; padding:9px 16px 12px; }
  .bandtop{ display:flex; align-items:baseline; gap:10px;
    border-bottom:1px solid var(--line-soft); padding-bottom:6px; margin-bottom:6px; }
  .bandtop .hw{ font-family:var(--display); font-size:1.25em; font-weight:600;
    color:var(--ink-strong); flex:1; line-height:1.25; }
  .bandtop .x{ cursor:pointer; color:var(--sepia); font-size:1.15em; padding:0 4px;
    align-self:center; }
  .bandtop .x:hover{ color:var(--rubric); }
  .flds{ display:flex; flex-wrap:wrap; gap:2px 28px; align-items:baseline; }
  .fld{ margin:3px 0; max-width:48ch; }
  .bandfull{ flex-basis:100%; margin-top:5px; }
  .mwuitem .et{ font-weight:600; }
  .lab{ font-family:var(--sans); font-size:.62em; text-transform:uppercase;
    letter-spacing:.12em; color:var(--sepia); display:block; margin-bottom:1px; }
  .val{ color:var(--ink); }
  .v-de{ font-style:italic; }
  .v-gram{ color:var(--gram); }
  .v-dialect{ color:var(--dialect); }
  .v-example{ color:var(--verdigris); font-style:italic; }
  .v-meta{ color:var(--sepia); font-style:italic; font-size:.93em; }
  .mwu{ border-left:2px solid var(--line); padding:4px 0 4px 11px; margin:8px 0; }
  .mwu .et{ font-weight:600; }
  .srctable{ margin-top:6px; }
  .band .srctable{ columns:2; column-gap:36px; }
  .band .srcrow{ break-inside:avoid; }
  .srcrow{ display:flex; gap:10px; padding:6px 0; border-top:1px solid var(--line-soft);
    align-items:baseline; }
  .srcrow .sname{ width:92px; min-width:92px; font-family:var(--sans); font-size:.7em;
    text-transform:uppercase; letter-spacing:.08em; color:var(--ink); line-height:1.5; }
  .srcrow .sname .syear{ display:block; font-size:.92em; color:var(--sepia);
    letter-spacing:.06em; }
  .srcrow .sform{ flex:1; }
  .lnk{ color:var(--ink-strong); text-decoration:underline;
    text-decoration-color:var(--line); text-underline-offset:3px; cursor:pointer;
    transition:color .12s ease, text-decoration-color .12s ease; }
  .lnk:hover{ color:var(--rubric); text-decoration-color:var(--rubric); }
  .none{ color:#b3a98d; font-style:italic; }
  .chip{ display:inline-block; font-family:var(--sans); font-size:.72em;
    letter-spacing:.03em; background:var(--paper); border:1px solid var(--line);
    border-radius:2px; padding:2px 9px; margin:3px 5px 0 0; cursor:pointer;
    transition:background .12s ease; }
  .chip:hover{ background:var(--hover); }
  .chip.strong{ font-weight:600; border-color:var(--ink); }
  .chip.weak{ font-size:.6em; color:var(--gram); background:transparent; }
  .btn{ font-family:var(--sans); font-size:.72em; letter-spacing:.06em;
    text-transform:uppercase; cursor:pointer; margin:10px 6px 2px 0;
    border:1px solid var(--ink); background:transparent; color:var(--ink);
    border-radius:2px; padding:5px 12px; transition:background .12s ease, color .12s ease; }
  .btn:hover{ background:var(--ink-strong); color:var(--paper); }
  .pdf{ font-family:var(--sans); font-size:.65em; letter-spacing:.04em; color:var(--sepia);
    border:1px dashed var(--line); border-radius:2px; padding:2px 8px; cursor:pointer;
    background:transparent; align-self:center; white-space:nowrap; }
  .pdf:hover{ background:var(--paper); color:var(--ink); }
  .empty-ws{ color:var(--sepia); font-style:italic; padding:48px; max-width:52ch; }
  .qbadge{ color:var(--rubric); font-weight:700; cursor:help; }
  body.hideflags .dot, body.hideflags .qbadge{ display:none; }
  body.hideflags .band.flag{ border-left-color:var(--line); }
  body.hideflags .band.master.flag{ border-left-color:var(--ink-strong); }
  .srctag{ display:inline-block; font-family:var(--sans); font-size:.6em;
    text-transform:uppercase; letter-spacing:.1em; color:var(--sepia);
    border:1px solid var(--line); padding:0 5px; border-radius:2px;
    vertical-align:2px; margin-right:2px; }
  .srctag.master{ color:var(--paper); background:var(--ink-strong); border-color:var(--ink-strong); }
  ::-webkit-scrollbar{ width:10px; height:10px; }
  ::-webkit-scrollbar-thumb{ background:var(--line); border-radius:5px; }
  ::-webkit-scrollbar-track{ background:transparent; }
  @media (max-width:1000px){ .band .srctable{ columns:1; } }
  @media (max-width:760px){
    .layout{ flex-direction:column; }
    .browser{ width:100%; min-width:0; max-height:38vh; border-right:0;
      border-bottom:1px solid var(--line); }
    .legend{ margin-left:0; }
  }
  @media (prefers-reduced-motion:reduce){ *{ transition:none !important; } }
</style>
</head>
<body>
<header>
  <div class="htop">
    <h1>LexLex <span class="rdash">&mdash;</span> Lexicon Lexicorum Esthonicorum</h1>
    <span class="tagline">__TAG__</span>
    <span class="legend"><span class="dot">&#9679;</span> marks entries needing attention — "???" or an attested form not yet linked</span>
  </div>
  <nav id="modes"></nav>
</header>
<div class="layout">
  <div class="browser">
    <div class="bcount" id="bcount"></div>
    <div id="items"></div>
    <div class="pager" id="pager"></div>
  </div>
  <div class="workspace" id="workspace">
    <div class="empty-ws">Pick a concept or dictionary entry on the left to open it here. Use the source links inside a master concept to open editions side by side.</div>
  </div>
</div>

<script id="data" type="application/json">/*__DATA__*/</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const SOURCES = DATA.sources;
const PLACE = new Set(['---','???','NULL','','&nbsp;',' ']);
// lookups
const MID = {};            // master id -> concept
const MBYCODE = {};        // Amt-Master-ID -> concept
const EDID = {};           // edition id -> {src, e}
DATA.master.forEach(c => { MID[c.id]=c; MBYCODE[c['Amt-Master-ID']]=c; });
for (const s of SOURCES) DATA.ed[s].forEach(e => { EDID[e.id]={src:s, e}; });
// combined pool for the "All data" search tab
const ALLITEMS = DATA.master.concat(...SOURCES.map(s=>DATA.ed[s]));
const isMaster = o => !!o.id && o.id.slice(0,3)==='am-';

const ED_LABELS = {
  'headword-modern':'Modern form','explanation':'Explanation',
  'pos':'Part of speech','grammar':'Grammar',
  'latin':'Latin','meaning-et':'Meaning (et)','syn-et':'Synonym (et)',
  'syn-de':'Synonym (de)','example-et':'Example (et)','example-de':'Example (de)',
  'variant':'Variant','dialect':'Dialect','regional':'Regional','usage':'Usage',
  'xref':'Cross-ref','page':'Page','comment':'Comment'
};
const ED_ORDER = ['headword-modern','explanation','pos','grammar','latin','meaning-et','syn-et',
  'syn-de','example-et','example-de','variant','dialect','regional','usage',
  'xref','page','comment'];
const VCLASS = { 'dialect':'v-dialect','regional':'v-dialect','example-et':'v-example',
  'example-de':'v-example','grammar':'v-gram','pos':'v-gram','comment':'v-meta',
  'usage':'v-meta','meaning-et':'v-meta','latin':'v-meta','variant':'v-gram',
  'explanation':'v-meta' };
const MASTER_FIELDS = [['Amt-Cat','Category'],['Sem-Cat','Semantic cat.'],
  ['Teema','Theme'],['Sugu','Gender'],['DEF_en','Definition (en)'],
  ['DEF_et','Definition (et)'],['Cross-source count','Cross-source count'],
  ['Comment-1','Comment 1'],['Comment-2','Comment 2'],['Comment-3','Comment 3']];

let state = { mode:'master', dict:SOURCES[0], q:'', match:'contains', page:0, panels:[], flaggedOnly:false, hideFlags:false };
const PAGE = 120;
const esc = s => (s==null?'':String(s)).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

// ---------- search matching (contains / starts / ends / wildcard* / regex) ----------
// state.q is already lowercased; haystacks are lowercased by the caller.
function globToRegex(g){
  const body = g.replace(/[.+^${}()|[\]\\]/g,'\\$&').replace(/\*/g,'.*').replace(/\?/g,'.');
  return new RegExp('^'+body+'$','i');
}
function makeMatcher(){
  const q = state.q;
  if(!q) return {ok:true, fn:()=>true};
  const m = state.match;
  if(m==='starts') return {ok:true, fn:h=>h.startsWith(q)};
  if(m==='ends')   return {ok:true, fn:h=>h.endsWith(q)};
  if(m==='wild'){ try{ const rx=globToRegex(q); return {ok:true, fn:h=>rx.test(h)}; }
                  catch(e){ return {ok:false, fn:()=>false}; } }
  if(m==='regex'){ try{ const rx=new RegExp(q,'i'); return {ok:true, fn:h=>rx.test(h)}; }
                   catch(e){ return {ok:false, fn:()=>false}; } }
  return {ok:true, fn:h=>h.includes(q)};   // contains
}
const flaggedM = c => (c._q && c._q.length) || (c._u && c._u.length);
const flaggedE = e => e._q && e._q.length;

// ---------- top nav ----------
function renderModes(){
  const nav = document.getElementById('modes');
  let h = `<button data-m="master" class="${state.mode==='master'?'active':''}">Master concepts</button>`;
  h += `<button data-m="all" class="${state.mode==='all'?'active':''}" title="search master concepts and all six dictionaries together">All data</button>`;
  for (const s of SOURCES)
    h += `<button data-m="dict" data-s="${s}" class="${state.mode==='dict'&&state.dict===s?'active':''}">${esc(s)}</button>`;
  h += `<input id="search" placeholder="search…" value="${esc(state.q)}">`;
  h += `<select id="matchmode" title="how the search matches the headword">
    <option value="contains">contains</option>
    <option value="starts">starts with</option>
    <option value="ends">ends with</option>
    <option value="wild">wildcard *</option>
    <option value="regex">regex</option></select>`;
  h += `<button id="flagtog" class="flagtog ${state.flaggedOnly?'on':''}" title="show only entries that need attention (??? or unlinked)">&#9873; flagged only</button>`;
  h += `<button id="hidetog" class="hidetog ${state.hideFlags?'on':''}" title="hide all &#9679; attention markers for a clean view">&#9673; hide flags</button>`;
  nav.innerHTML = h;
  nav.querySelectorAll('button[data-m]').forEach(b=>b.onclick=()=>{
    state.mode=b.dataset.m; if(b.dataset.s) state.dict=b.dataset.s;
    state.q=''; state.page=0; renderModes(); renderBrowser();
  });
  document.getElementById('flagtog').onclick=()=>{
    state.flaggedOnly=!state.flaggedOnly; state.page=0; renderModes(); renderBrowser();
  };
  document.getElementById('hidetog').onclick=(ev)=>{
    state.hideFlags=!state.hideFlags;
    document.body.classList.toggle('hideflags', state.hideFlags);
    ev.currentTarget.classList.toggle('on', state.hideFlags);
  };
  const sel = document.getElementById('matchmode');
  sel.value = state.match;
  sel.onchange = ()=>{ state.match=sel.value; state.page=0; renderBrowser(); };
  document.body.classList.toggle('hideflags', state.hideFlags);
  const inp = document.getElementById('search');
  inp.oninput = ()=>{ state.q=inp.value.trim().toLowerCase(); state.page=0; renderBrowser(); };
  inp.focus();
}

// ---------- browser list ----------
function currentList(){
  const q = state.q;
  const {ok, fn} = makeMatcher();
  const inp = document.getElementById('search');
  if (inp) inp.classList.toggle('badq', !!q && !ok);   // flag an invalid regex
  const matchMaster = c =>
    fn((c['Amt-Master-ID']||'').toLowerCase()) ||
    fn((c.DEF_et||'').toLowerCase()) ||
    fn((c.DEF_en||'').toLowerCase()) ||
    SOURCES.some(s=>fn((c[s+'-et']||'').toLowerCase()));
  const matchEdition = e =>
    fn((e['headword-et']||'').toLowerCase()) ||
    fn((e['equiv-de']||'').toLowerCase());
  let arr;
  if (state.mode==='master'){
    arr = DATA.master;
    if (state.flaggedOnly) arr = arr.filter(flaggedM);
    if (q) arr = arr.filter(matchMaster);
  } else if (state.mode==='all'){
    arr = ALLITEMS;
    if (state.flaggedOnly) arr = arr.filter(o=> isMaster(o)?flaggedM(o):flaggedE(o));
    if (q) arr = arr.filter(o=> isMaster(o)?matchMaster(o):matchEdition(o));
  } else {
    arr = DATA.ed[state.dict];
    if (state.flaggedOnly) arr = arr.filter(flaggedE);
    if (q) arr = arr.filter(matchEdition);
  }
  return arr;
}
function renderBrowser(){
  const list = currentList();
  const pages = Math.max(1, Math.ceil(list.length/PAGE));
  if(state.page>=pages) state.page=pages-1;
  if(state.page<0) state.page=0;
  document.getElementById('bcount').innerHTML =
    `${list.length} ${state.mode==='master'?'concepts':state.mode==='all'?'results':'entries'}`
    + (state.flaggedOnly?` <span class="dot">&#9679;</span> flagged`:'')
    + (state.q?` matching “${esc(state.q)}”`:'');
  const start = state.page*PAGE, slice = list.slice(start, start+PAGE);
  const box = document.getElementById('items');
  const showTag = state.mode==='all';
  box.innerHTML = slice.map((o)=>{
    const asMaster = state.mode==='all' ? isMaster(o) : state.mode==='master';
    if (asMaster){
      const flag = flaggedM(o)?' <span class="dot">&#9679;</span>':'';
      const sub = esc((o.DEF_et||o.DEF_en||'').slice(0,70));
      const tag = showTag ? '<span class="srctag master">master</span> ' : '';
      return `<div class="item" data-k="master" data-id="${o.id}">
        <span class="code">${tag}${esc(o['Amt-Master-ID'])}</span>${flag}
        <div class="sub">${sub}</div></div>`;
    } else {
      const flag = flaggedE(o)?' <span class="dot">&#9679;</span>':'';
      const hw = esc(o['headword-et']||'—');
      const sub = esc((o['equiv-de']||'').slice(0,70));
      const tag = showTag ? `<span class="srctag">${esc((o.source||'').split('-')[0])}</span> ` : '';
      return `<div class="item" data-k="edition" data-id="${o.id}">
        <span class="code">${tag}${hw}</span>${flag}
        <div class="sub">${sub}</div></div>`;
    }
  }).join('');
  box.querySelectorAll('.item').forEach(el=>el.onclick=()=>{
    box.querySelectorAll('.item').forEach(x=>x.classList.remove('sel'));
    el.classList.add('sel');
    openPanel(el.dataset.k, el.dataset.id, true);
  });
  const atFirst=state.page<=0, atLast=state.page>=pages-1;
  document.getElementById('pager').innerHTML = pages>1 ?
    `<div class="grp">
       <button ${atFirst?'disabled':''} data-p="first" title="first page">&laquo;</button>
       <button ${atFirst?'disabled':''} data-p="prev" title="previous">&lsaquo;</button>
     </div>
     <span>page <input class="jump" id="jump" value="${state.page+1}" title="jump to page"> / ${pages}</span>
     <div class="grp">
       <button ${atLast?'disabled':''} data-p="next" title="next">&rsaquo;</button>
       <button ${atLast?'disabled':''} data-p="last" title="last page">&raquo;</button>
     </div>` : '';
  const gotoPage = p=>{ state.page=Math.max(0,Math.min(pages-1,p)); renderBrowser(); };
  document.querySelectorAll('#pager button[data-p]').forEach(b=>b.onclick=()=>{
    const p=b.dataset.p;
    gotoPage(p==='first'?0 : p==='last'?pages-1 : p==='prev'?state.page-1 : state.page+1);
  });
  const jmp=document.getElementById('jump');
  if(jmp) jmp.onchange=()=>{ const n=parseInt(jmp.value,10);
    if(!isNaN(n)) gotoPage(n-1); else jmp.value=state.page+1; };
}

// ---------- workspace panels ----------
function openPanel(kind, id, reset){
  if (reset) state.panels = [];
  if (!state.panels.some(p=>p.kind===kind && p.id===id))
    state.panels.push({kind, id});
  renderWorkspace();
}
function closePanel(kind, id){
  state.panels = state.panels.filter(p=>!(p.kind===kind&&p.id===id));
  renderWorkspace();
}
function qbadge(flags){
  return flags&&flags.length ? ` <span class="qbadge" title="needs attention: ${esc(flags.join(', '))}">&#9679;</span>`:'';
}
// chronological ledger (canonical v3 layout): each opened entry is a
// full-width band, stacked master-first then by source year (1637→1780).
function wireWs(ws){
  ws.querySelectorAll('[data-act]').forEach(el=>{
    const a=el.dataset.act;
    if(a==='close') el.onclick=()=>closePanel(el.dataset.kind, el.dataset.id);
    if(a==='open-ed') el.onclick=()=>openPanel('edition', el.dataset.id, false);
    if(a==='open-master') el.onclick=()=>openPanel('master', el.dataset.id, false);
    if(a==='open-all') el.onclick=()=>{
      const c=MID[el.dataset.id];
      SOURCES.forEach(s=>(c[s+'-id']||[]).forEach(eid=>openPanel('edition',eid,false)));
    };
    if(a==='pdf') el.onclick=()=>openPdf(el.dataset.src, el.dataset.page);
  });
}
function fld(lab, val, cls){
  if(!val||PLACE.has(val)) return '';
  const q=(''+val).includes('???')?' <span class="dot">&#9679;</span>':'';
  return `<div class="fld"><span class="lab">${lab}</span><span class="val ${cls||''}">${esc(val)}${q}</span></div>`;
}

// ---------- "Also worth checking" suggestions (rank 2) ----------
// Heuristic lookup across ALL data, no embeddings and deliberately no POS
// filter: normalized old-orthography Estonian forms (exact match + prefix
// kinship) plus shared German equivalent tokens (synonym signal across
// sources). Meant to invite browsing beyond the exact linked matches.
const DE_STOP=new Set(['aber','alle','allerley','also','ander','andere','auch','beym',
  'damit','dann','dass','dem','denen','derer','desgleichen','diese','dieser','dieses',
  'ding','durch','eine','einem','einen','einer','eines','etwas','gantz','ganz','haben',
  'hat','ist','item','kann','machen','man','mein','mich','nach','nicht','noch','oder',
  'ohne','sein','seine','sehr','sich','sind','soll','uber','über','und','vom','von',
  'vor','was','wenn','werden','wie','wird','zum','zur']);
function asText(s){
  if(typeof s==='string') return s;
  return Array.isArray(s)?s.join(' '):'';
}
function normEt(s){
  s=asText(s).toLowerCase().replace(/\(.*?\)/g,' ').replace(/[^a-zäöüõšž]+/g,' ').trim();
  if(!s) return [];
  return s.split(' ').map(w=>w.replace(/w/g,'v').replace(/y/g,'i').replace(/(.)\1+/g,'$1'))
          .filter(w=>w.length>=3);
}
function normDe(s){
  s=asText(s).toLowerCase().replace(/ß/g,'ss');
  return s.split(/[^a-zäöü]+/).filter(t=>t.length>=4&&!DE_STOP.has(t));
}
let SUGG=null;
function buildSugg(){
  const t0=performance.now();
  SUGG={et:new Map(), p4:new Map(), de:new Map(), keys:new Map()};
  const add=(m,k,id)=>{ let a=m.get(k); if(!a){a=[];m.set(k,a);} a.push(id); };
  for(const o of ALLITEMS){
    const ets=new Set(), des=new Set();
    if(isMaster(o)){
      normEt(o['Amt-Master-ID']).forEach(t=>ets.add(t));
      for(const s of SOURCES){
        normEt(o[s+'-et']).forEach(t=>ets.add(t));
        normDe(o[s+'-de']).forEach(t=>des.add(t));
      }
    } else {
      for(const k of ['headword-et','variant','syn-et','headword-modern'])
        normEt(o[k]).forEach(t=>ets.add(t));
      for(const k of ['equiv-de','syn-de'])
        normDe(o[k]).forEach(t=>des.add(t));
    }
    SUGG.keys.set(o.id,{ets,des});
    ets.forEach(t=>{ add(SUGG.et,t,o.id); if(t.length>=4) add(SUGG.p4,t.slice(0,4),o.id); });
    des.forEach(t=>add(SUGG.de,t,o.id));
  }
  console.log(`see-also index: ${ALLITEMS.length} items in ${Math.round(performance.now()-t0)} ms`);
}
function suggestions(id, excl){
  if(!SUGG) buildSugg();
  const own=SUGG.keys.get(id); if(!own) return [];
  const score=new Map();
  const bump=(oid,n)=>{ if(oid===id||excl.has(oid)) return; score.set(oid,(score.get(oid)||0)+n); };
  own.ets.forEach(t=>{
    (SUGG.et.get(t)||[]).forEach(oid=>bump(oid,3));           // same normalized form
    if(t.length>=4){
      const bucket=SUGG.p4.get(t.slice(0,4))||[];
      if(bucket.length<=400) for(const oid of bucket){        // orthographic kin
        if(oid===id||excl.has(oid)) continue;
        for(const u of SUGG.keys.get(oid).ets)
          if(u!==t&&(u.startsWith(t)||t.startsWith(u))){ bump(oid,2); break; }
      }
    }
  });
  own.des.forEach(t=>{                                        // shared German gloss
    const a=SUGG.de.get(t)||[];
    if(a.length<=150) a.forEach(oid=>bump(oid,2));
  });
  return [...score.entries()].sort((a,b)=>b[1]-a[1]).slice(0,24).map(([oid])=>oid);
}
function suggLabel(oid){
  if(oid.slice(0,3)==='am-'){
    const c=MID[oid]; return c?'master: '+c['Amt-Master-ID']:'';
  }
  const r=EDID[oid]; if(!r) return '';
  const hw=[r.e['headword-et'],r.e['headword-modern'],r.e['equiv-de']]
    .find(v=>v&&!PLACE.has(v))||'—';
  return r.src.split('-')[0]+': '+hw;
}
function suggChip(oid, cls){
  const lab=suggLabel(oid); if(!lab) return '';
  const act=oid.slice(0,3)==='am-'?'open-master':'open-ed';
  return `<span class="chip ${cls}" data-act="${act}" data-id="${oid}">${esc(lab)}</span>`;
}
function suggBlock(id, excl){
  try{
    const seen=new Set(), chips=[];
    for(const oid of suggestions(id, excl)){
      const lab=suggLabel(oid);
      if(!lab||seen.has(lab)) continue;               // dedupe same-looking chips
      seen.add(lab);
      chips.push(suggChip(oid,'weak'));
    }
    return chips.length?`<div class="bandfull"><span class="lab">Also worth checking</span>${chips.join('')}</div>`:'';
  }catch(err){ console.warn('see-also suggestions failed for', id, err); return ''; }
}
function srctable(c){
  let st='<div class="srctable bandfull">';
  for(const s of SOURCES){
    const et=c[s+'-et']||'---', de=c[s+'-de']||'', ids=c[s+'-id']||[];
    const attested=!PLACE.has(et);
    let form;
    if(ids.length){
      form=ids.map(eid=>`<span class="lnk" data-act="open-ed" data-id="${eid}">${esc(et)}</span>`).join(' &middot; ');
      if(ids.length>1) form+=` <span class="v-meta">(${ids.length})</span>`;
    } else if(attested){
      form=`<span>${esc(et)}</span> <span class="dot" title="attested but not linked">&#9679;</span>`;
    } else {
      form=`<span class="none">&mdash;</span>`;
    }
    const q=(et.includes('???')||de.includes('???'))?' <span class="dot">&#9679;</span>':'';
    st+=`<div class="srcrow"><span class="sname">${s.split('-')[0]}<span class="syear">${s.split('-')[1]||''}</span></span>
      <span class="sform">${form}${q}${PLACE.has(de)?'':`<div class="v-de">${esc(de)}</div>`}</span></div>`;
  }
  return st+'</div>';
}
function masterBand(c, id){
  if(!c) return '';
  let flds='';
  for(const [k,lab] of MASTER_FIELDS) flds+=fld(lab, c[k]);
  const anyLink=SOURCES.some(s=>(c[s+'-id']||[]).length);
  const allBtn=anyLink?`<button class="btn" data-act="open-all" data-id="${c.id}">Open all linked editions &rarr;</button>`:'';
  // rank 2: lookup suggestions beyond the concept's own linked editions
  const excl=new Set();
  for(const s of SOURCES) (c[s+'-id']||[]).forEach(eid=>excl.add(eid));
  const worth=suggBlock(c.id, excl);
  return `<div class="band master ${flaggedM(c)?'flag':''}">
    <div class="spine"><span class="src master">master</span></div>
    <div class="bandmain">
      <div class="bandtop"><span class="hw">${esc(c['Amt-Master-ID'])}${qbadge(c._q||[])}</span>
        <span class="x" data-act="close" data-kind="master" data-id="${id}">&times;</span></div>
      <div class="flds">${flds}${srctable(c)}${worth}</div>${allBtn}
    </div></div>`;
}
function editionBand(id){
  const rec=EDID[id]; if(!rec) return '';
  const {src,e}=rec;
  let flds=fld('German', e['equiv-de'], 'v-de');
  for(const k of ED_ORDER) flds+=fld(ED_LABELS[k], e[k], VCLASS[k]||'');
  let mw='';
  if(e.mwu&&e.mwu.length){
    mw=`<div class="bandfull"><span class="lab">Multi-word units (${e.mwu.length})</span>`+
      e.mwu.map(m=>{
        const q=Object.values(m).some(x=>(''+x).includes('???'))?' <span class="dot">&#9679;</span>':'';
        return `<span class="mwuitem"><span class="et">${esc(m['mwu-et']||'')}</span>${q} <span class="v-de">${esc(m['mwu-de']||'')}</span></span>`;
      }).join(' &nbsp;&middot;&nbsp; ')+'</div>';
  }
  const mids=e['master-id']||[];
  const back=mids.length?'<div class="bandfull">'+mids.map(mid=>{const c=MID[mid];
    return `<span class="chip" data-act="open-master" data-id="${mid}">${esc(c?c['Amt-Master-ID']:mid)}</span>`;}).join('')+'</div>':'';
  // rank 1: siblings actually linked through the same master concept(s)
  const sibIds=new Set();
  for(const mid of mids){ const c=MID[mid]; if(!c) continue;
    for(const s2 of SOURCES) (c[s2+'-id']||[]).forEach(eid=>{ if(eid!==id) sibIds.add(eid); }); }
  const sa=sibIds.size?`<div class="bandfull"><span class="lab">See also (same concept)</span>`+
    [...sibIds].map(eid=>suggChip(eid,'strong')).join('')+'</div>':'';
  // rank 2: heuristic lookup suggestions, excluding self + rank-1 + own masters
  const excl=new Set([...sibIds,...mids]);
  const worth=suggBlock(id, excl);
  const pdf=e.page&&!PLACE.has(e.page)?
    `<button class="pdf" data-act="pdf" data-src="${src}" data-page="${esc(e.page)}">&#128196; p.${esc(e.page)}</button>`:'';
  return `<div class="band ${flaggedE(e)?'flag':''}">
    <div class="spine">${esc(src.split('-')[0])}<span class="syear">${esc(src.split('-')[1]||'')}</span></div>
    <div class="bandmain">
      <div class="bandtop"><span class="hw">${esc(e['headword-et']||'&mdash;')}${qbadge(e._q||[])}</span>${pdf}
        <span class="x" data-act="close" data-kind="edition" data-id="${id}">&times;</span></div>
      <div class="flds">${flds}${mw}${back}${sa}${worth}</div>
    </div></div>`;
}
function bandOrder(p){
  if(p.kind==='master') return -1;
  const r=EDID[p.id]; return r?SOURCES.indexOf(r.src):99;
}
function renderWorkspace(){
  const ws=document.getElementById('workspace');
  if(!state.panels.length){ ws.innerHTML='<div class="empty-ws">Nothing open.</div>'; return; }
  const panels=state.panels.slice().sort((a,b)=>bandOrder(a)-bandOrder(b));
  ws.innerHTML=panels.map(p=>p.kind==='master'?masterBand(MID[p.id],p.id):editionBand(p.id)).join('');
  wireWs(ws);
}

// ---------- PDF stub (future pop-up) ----------
function openPdf(src, page){
  alert('PDF page pop-up — coming soon.\n\nSource: '+src+'\nPage: '+page+
        '\n\n(The viewer already passes the source + page here; the PDF overlay will hook in at openPdf().)');
}

// ---------- deep links (#concept=<Amt-Master-ID> or #entry=<edition-id>) ----------
function openFromHash(){
  const h = decodeURIComponent((location.hash||'').replace(/^#/,''));
  if(!h) return;
  const eq=h.indexOf('='); if(eq<0) return;
  const k=h.slice(0,eq), v=h.slice(eq+1);
  if(k==='concept'){
    const c=MBYCODE[v]||MID[v];
    if(c){ state.mode='master'; openPanel('master',c.id,true);
      SOURCES.forEach(s=>(c[s+'-id']||[]).forEach(eid=>openPanel('edition',eid,false)));
      renderModes(); renderBrowser(); }
  } else if(k==='entry' && EDID[v]){ openPanel('edition',v,true); }
  else if(k==='flagged'){ state.flaggedOnly=(v!=='0'); state.page=0; renderModes(); renderBrowser(); }
  else if(k==='mode' && (v==='all'||v==='master')){ state.mode=v; state.q=''; state.page=0; renderModes(); renderBrowser(); }
}
window.addEventListener('hashchange', openFromHash);

renderModes(); renderBrowser(); openFromHash();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# v2 — synoptic grid: one column per opened entry, fields aligned in shared
#      rows behind a sticky label gutter (differences sit on the same line).
# ---------------------------------------------------------------------------
V2_BLOCK = r"""<style>
  .workspace{ display:block; padding:0; overflow:auto; }
  .syn{ display:grid; min-width:min-content; }
  .sy-corner{ position:sticky; top:0; left:0; z-index:4; background:var(--rail);
    border-bottom:2px solid var(--ink-strong); border-right:1px solid var(--line); }
  .sy-lab{ position:sticky; left:0; z-index:2; background:var(--rail);
    font-family:var(--sans); font-size:.62em; text-transform:uppercase; letter-spacing:.12em;
    color:var(--sepia); padding:11px 14px 8px; border-bottom:1px solid var(--line-soft);
    border-right:1px solid var(--line); }
  .sy-head{ position:sticky; top:0; z-index:3; background:var(--panel);
    border-bottom:2px solid var(--ink-strong); border-left:1px solid var(--line);
    padding:10px 14px; display:flex; align-items:baseline; gap:8px; }
  .sy-head.flag{ box-shadow:inset 4px 0 0 var(--rubric); }
  body.hideflags .sy-head.flag{ box-shadow:none; }
  .sy-head .hw{ font-family:var(--display); font-size:1.15em; font-weight:600;
    color:var(--ink-strong); flex:1; line-height:1.25; }
  .sy-head .src, .sy-head .x{ align-self:center; }
  .sy-head .src{ font-family:var(--sans); font-size:.6em; text-transform:uppercase;
    letter-spacing:.12em; color:var(--sepia); border:1px solid var(--line);
    padding:2px 7px; border-radius:2px; white-space:nowrap; }
  .sy-head .src.master{ color:var(--paper); background:var(--ink-strong);
    border-color:var(--ink-strong); }
  .sy-head .x{ cursor:pointer; color:var(--sepia); font-size:1.15em; padding:0 4px; }
  .sy-head .x:hover{ color:var(--rubric); }
  .sy-cell{ background:var(--panel); border-bottom:1px solid var(--line-soft);
    border-left:1px solid var(--line); padding:7px 14px; font-size:.95em; }
  .sy-cell.blank{ background:var(--paper); }
  .sy-cell .mwu{ margin:4px 0; }
</style>
<script>
function wireWs(ws){
  ws.querySelectorAll('[data-act]').forEach(el=>{
    const a=el.dataset.act;
    if(a==='close') el.onclick=()=>closePanel(el.dataset.kind, el.dataset.id);
    if(a==='open-ed') el.onclick=()=>openPanel('edition', el.dataset.id, false);
    if(a==='open-master') el.onclick=()=>openPanel('master', el.dataset.id, false);
    if(a==='open-all') el.onclick=()=>{
      const c=MID[el.dataset.id];
      SOURCES.forEach(s=>(c[s+'-id']||[]).forEach(eid=>openPanel('edition',eid,false)));
    };
    if(a==='pdf') el.onclick=()=>openPdf(el.dataset.src, el.dataset.page);
  });
}
function v2cols(){
  return state.panels.map(p=>{
    if(p.kind==='master'){ const c=MID[p.id]; return c?{kind:'master',id:p.id,c}:null; }
    const r=EDID[p.id]; return r?{kind:'edition',id:p.id,src:r.src,e:r.e}:null;
  }).filter(Boolean);
}
function v2head(col){
  if(col.kind==='master'){
    const c=col.c;
    return `<div class="sy-head ${flaggedM(c)?'flag':''}"><span class="src master">master</span>
      <span class="hw">${esc(c['Amt-Master-ID'])}${qbadge(c._q||[])}</span>
      <span class="x" data-act="close" data-kind="master" data-id="${col.id}">&times;</span></div>`;
  }
  const {src,e}=col;
  const pdf = e.page&&!PLACE.has(e.page)?`<button class="pdf" data-act="pdf" data-src="${src}" data-page="${esc(e.page)}">p.${esc(e.page)}</button>`:'';
  return `<div class="sy-head ${flaggedE(e)?'flag':''}"><span class="src">${esc(src.split('-')[0])} ${esc(src.split('-')[1]||'')}</span>
    <span class="hw">${esc(e['headword-et']||'&mdash;')}${qbadge(e._q||[])}</span>${pdf}
    <span class="x" data-act="close" data-kind="edition" data-id="${col.id}">&times;</span></div>`;
}
function v2srctable(c){
  let st='<div class="srctable">';
  for(const s of SOURCES){
    const et=c[s+'-et']||'---', de=c[s+'-de']||'', ids=c[s+'-id']||[];
    const attested=!PLACE.has(et);
    let form;
    if(ids.length){
      form=ids.map(eid=>`<span class="lnk" data-act="open-ed" data-id="${eid}">${esc(et)}</span>`).join(' &middot; ');
      if(ids.length>1) form+=` <span class="v-meta">(${ids.length})</span>`;
    } else if(attested){
      form=`<span>${esc(et)}</span> <span class="dot" title="attested but not linked">&#9679;</span>`;
    } else {
      form=`<span class="none">&mdash;</span>`;
    }
    const q=(et.includes('???')||de.includes('???'))?' <span class="dot">&#9679;</span>':'';
    st+=`<div class="srcrow"><span class="sname">${s.split('-')[0]}<span class="syear">${s.split('-')[1]||''}</span></span>
      <span class="sform">${form}${q}${PLACE.has(de)?'':`<div class="v-de">${esc(de)}</div>`}</span></div>`;
  }
  return st+'</div>';
}
function v2mwu(e){
  if(!e.mwu||!e.mwu.length) return '';
  return e.mwu.map(m=>{
    const q=Object.values(m).some(x=>(''+x).includes('???'))?' <span class="dot">&#9679;</span>':'';
    return `<div class="mwu"><span class="et">${esc(m['mwu-et']||'')}</span>${q}<div class="v-de">${esc(m['mwu-de']||'')}</div></div>`;
  }).join('');
}
function v2links(col){
  if(col.kind==='master'){
    const c=col.c;
    const anyLink=SOURCES.some(s=>(c[s+'-id']||[]).length);
    return anyLink?`<button class="btn" data-act="open-all" data-id="${c.id}">Open all linked editions &rarr;</button>`:'';
  }
  const mids=col.e['master-id']||[];
  return mids.map(mid=>{const c=MID[mid];
    return `<span class="chip" data-act="open-master" data-id="${mid}">${esc(c?c['Amt-Master-ID']:mid)}</span>`;}).join('');
}
function v2cell(col,k){
  if(col.kind==='master'){
    const c=col.c;
    if(k==='sources') return v2srctable(c);
    if(k==='links') return v2links(col);
    if(k==='equiv-de'||k==='mwu'||ED_ORDER.includes(k)) return '';
    const v=c[k];
    return (v&&!PLACE.has(v))?`<span class="val">${esc(v)}</span>`:'';
  }
  const e=col.e;
  if(k==='sources') return '';
  if(k==='links') return v2links(col);
  if(k==='mwu') return v2mwu(e);
  if(k==='equiv-de') return (e['equiv-de']&&!PLACE.has(e['equiv-de']))?`<span class="val v-de">${esc(e['equiv-de'])}</span>`:'';
  if(ED_ORDER.includes(k)){
    const v=e[k];
    return (v&&!PLACE.has(v))?`<span class="val ${VCLASS[k]||''}">${esc(v)}${(''+v).includes('???')?' <span class="dot">&#9679;</span>':''}</span>`:'';
  }
  return '';
}
function renderWorkspace(){
  const ws=document.getElementById('workspace');
  const cols=v2cols();
  if(!cols.length){ ws.innerHTML='<div class="empty-ws">Nothing open.</div>'; return; }
  const order=[], labels={};
  const push=(k,l)=>{ if(!(k in labels)){ order.push(k); labels[k]=l; } };
  MASTER_FIELDS.forEach(([k,l])=>push(k,l));
  push('sources','Cited forms');
  push('equiv-de','German');
  ED_ORDER.forEach(k=>push(k,ED_LABELS[k]));
  push('mwu','Multi-word units');
  push('links','Linked to');
  let h=`<div class="syn" style="grid-template-columns:150px repeat(${cols.length},minmax(240px,330px))">`;
  h+='<div class="sy-corner"></div>'+cols.map(v2head).join('');
  for(const k of order){
    const cells=cols.map(c=>v2cell(c,k));
    if(cells.every(x=>!x)) continue;
    h+=`<div class="sy-lab">${labels[k]}</div>`+cells.map(x=>`<div class="sy-cell ${x?'':'blank'}">${x}</div>`).join('');
  }
  h+='</div>';
  ws.innerHTML=h;
  wireWs(ws);
  ws.scrollLeft=ws.scrollWidth;
}
if(state.panels.length) renderWorkspace();
</script>"""

VARIANTS = {"v2": V2_BLOCK}   # v3 was promoted to the base layout in v5


if __name__ == "__main__":
    build(tuple(a for a in sys.argv[1:] if a in VARIANTS))

# Created: 2026-07-08 17-25-00
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
"""
Step 4 — build a single-file HTML viewer for the parsed dictionary.

Reads json/all_entries.json + json/manifest.json and produces
APTSK_sonastik.html in the project root: a self-contained page (the data
travels inside the file as gzip+base64, unpacked in the browser with
DecompressionStream) styled after LexLex_v6.html — paper palette, rubric-red
(#FF0000) accents, browser rail + workspace bands — with an Advanced search
panel able to query every field of the JSON, "in all content", regex,
wildcards, diacritic folding, numeric ranges, source presence and flag
filters, plus CSV/JSON export of the filtered set.

Run:  uv run python APTSK-scripts/aptsk_04_build_html.py
"""

import base64
import gzip
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "APTSK_sonastik.html"

entries = json.loads((ROOT / "json" / "all_entries.json").read_text())
manifest = json.loads((ROOT / "json" / "manifest.json").read_text())

payload = {"meta": manifest, "entries": entries["entries"]}
raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
b64 = base64.b64encode(gzip.compress(raw, 9)).decode()

generated = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
count = f"{entries['entry_count']:,}".replace(",", " ")

TEMPLATE = r"""<!DOCTYPE html>
<html lang="et">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>APTSK — Eesti vanema piiblitõlke sõnastik 1600–1739</title>
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
    border-bottom:3px double var(--line); padding:12px 20px 10px; }
  .htop{ display:flex; align-items:baseline; gap:18px; flex-wrap:wrap; }
  h1{ font-family:var(--display); font-size:1.4em; font-weight:600;
    color:var(--ink-strong); letter-spacing:.01em; }
  h1 .rdash{ color:var(--rubric); }
  .tagline{ font-family:var(--sans); font-size:.66em; font-weight:400;
    letter-spacing:.14em; color:var(--sepia); white-space:nowrap; }
  .legend{ font-family:var(--sans); font-size:.72em; letter-spacing:.02em;
    color:var(--sepia); margin-left:auto; }
  .dot{ color:var(--rubric); font-weight:700; }
  nav{ display:flex; gap:6px; margin-top:10px; flex-wrap:wrap; align-items:center;}
  nav button{ font-family:var(--sans); font-size:.7em; text-transform:uppercase;
    letter-spacing:.07em; color:var(--ink); background:transparent;
    border:1px solid var(--line); border-radius:2px; padding:5px 10px; cursor:pointer;
    transition:background .12s ease, color .12s ease; }
  nav button:hover{ background:var(--hover); }
  nav button.active{ background:var(--ink-strong); color:var(--paper); border-color:var(--ink-strong);}
  nav button.letter{ padding:5px 8px; min-width:30px; }
  nav button.flagtog.on{ background:transparent; color:var(--rubric); border-color:var(--rubric); font-weight:600; }
  #search{ font-family:var(--serif); font-size:.95em; font-style:italic; padding:5px 12px;
    width:230px; border:1px solid var(--line); border-radius:2px;
    background:var(--panel); color:var(--ink); }
  #search::placeholder{ color:var(--sepia); }
  nav select, .adv select{ font-family:var(--sans); font-size:.75em; padding:5px 6px; color:var(--ink);
    background:var(--panel); border:1px solid var(--line); border-radius:2px; cursor:pointer; }
  :is(nav button,#search,nav select,.adv select,.adv input,.adv button,.pager button,.pager input):focus-visible{
    outline:2px solid var(--ink-strong); outline-offset:1px; }
  .badq{ outline:2px solid var(--rubric); }

  /* ---------- advanced search panel ---------- */
  .adv{ display:none; background:var(--rail); border-bottom:3px double var(--line);
    padding:12px 20px 14px; font-family:var(--sans); font-size:.8em; }
  .adv.open{ display:block; }
  .adv h2{ font-family:var(--sans); font-size:.85em; text-transform:uppercase;
    letter-spacing:.14em; color:var(--ink-strong); margin-bottom:8px; }
  .adv h2 .rq{ color:var(--rubric); }
  .advgrid{ display:flex; flex-wrap:wrap; gap:18px 34px; align-items:flex-start; }
  .advcol{ min-width:280px; }
  .advcol h3{ font-size:.72em; text-transform:uppercase; letter-spacing:.12em;
    color:var(--sepia); margin:0 0 6px; border-bottom:1px solid var(--line-soft);
    padding-bottom:3px; }
  .crow{ display:flex; gap:5px; margin:4px 0; align-items:center; flex-wrap:wrap; }
  .crow select{ max-width:190px; }
  .crow input[type=text]{ font-family:var(--serif); font-style:italic; font-size:1.05em;
    padding:4px 8px; border:1px solid var(--line); border-radius:2px;
    background:var(--panel); color:var(--ink); width:170px; }
  .crow .del{ cursor:pointer; color:var(--sepia); border:none; background:none;
    font-size:1.1em; padding:0 4px; }
  .crow .del:hover{ color:var(--rubric); }
  .adv .btn2{ font-family:var(--sans); font-size:.78em; letter-spacing:.06em;
    text-transform:uppercase; cursor:pointer; border:1px solid var(--ink);
    background:transparent; color:var(--ink); border-radius:2px; padding:4px 11px;
    transition:background .12s, color .12s; margin:6px 6px 0 0; }
  .adv .btn2:hover{ background:var(--ink-strong); color:var(--paper); }
  .adv .btn2.primary{ background:var(--ink-strong); color:var(--paper); }
  .adv .btn2.primary:hover{ background:var(--rubric); border-color:var(--rubric); }
  .adv label.ck{ display:inline-flex; gap:5px; align-items:center; margin:3px 12px 3px 0;
    cursor:pointer; color:var(--ink); }
  .srcpick{ display:flex; flex-wrap:wrap; gap:4px; }
  .srcpick .chip{ user-select:none; }
  .chip{ display:inline-block; font-family:var(--sans); font-size:.9em;
    letter-spacing:.03em; background:var(--paper); border:1px solid var(--line);
    border-radius:2px; padding:2px 9px; margin:2px 4px 0 0; cursor:pointer;
    transition:background .12s ease; }
  .chip:hover{ background:var(--hover); }
  .chip.req{ border-color:var(--rubric); color:var(--rubric); font-weight:700; }
  .chip.exc{ text-decoration:line-through; opacity:.55; }
  .fl{ display:flex; gap:6px; align-items:center; margin:3px 0; }
  .fl span{ width:150px; color:var(--ink); }
  .numrange input{ width:70px; font-family:var(--sans); padding:4px 6px;
    border:1px solid var(--line); border-radius:2px; background:var(--panel); }
  .advsum{ margin-top:10px; font-size:.85em; color:var(--sepia); }
  .advsum b{ color:var(--rubric); }

  /* ---------- layout ---------- */
  .layout{ flex:1; min-height:0; display:flex; }
  .browser{ width:330px; min-width:330px; background:var(--rail);
    border-right:1px solid var(--line); overflow-y:auto; }
  .bcount{ font-family:var(--sans); font-size:.7em; letter-spacing:.06em;
    text-transform:uppercase; color:var(--sepia); padding:10px 16px;
    border-bottom:1px solid var(--line); }
  .item{ padding:8px 16px 8px 13px; border-bottom:1px solid var(--line-soft);
    border-left:3px solid transparent; cursor:pointer; transition:background .12s ease; }
  .item:hover{ background:var(--hover); }
  .item.sel{ background:var(--sel); border-left-color:var(--rubric); }
  .item .code{ font-weight:600; color:var(--ink-strong); }
  .item .code sup{ font-size:.7em; color:var(--sepia); }
  .item .cnt{ float:right; font-family:var(--sans); font-size:.72em; color:var(--sepia); }
  .item .sub{ font-size:.8em; color:var(--sepia); font-style:italic;
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

  /* ---------- entry band ---------- */
  .band{ display:flex; background:var(--panel); border:1px solid var(--line);
    border-left:4px solid var(--ink-strong); box-shadow:0 2px 8px rgba(42,34,21,.05);
    margin:0 0 12px; }
  .band.flag{ border-left-color:var(--rubric); }
  .band.refband{ border-left-color:var(--sepia); }
  .spine{ width:108px; min-width:108px; padding:13px 0 12px 14px;
    border-right:1px solid var(--line-soft); font-family:var(--sans); font-size:.72em;
    text-transform:uppercase; letter-spacing:.08em; color:var(--ink); line-height:1.5; }
  .spine .letterbig{ font-family:var(--display); font-size:2.1em; color:var(--ink-strong);
    display:block; line-height:1.1; text-transform:none; }
  .spine .syear{ display:block; color:var(--sepia); letter-spacing:.06em;
    text-transform:none; }
  .bandmain{ flex:1; min-width:0; padding:9px 16px 12px; }
  .bandtop{ display:flex; align-items:baseline; gap:10px; flex-wrap:wrap;
    border-bottom:1px solid var(--line-soft); padding-bottom:6px; margin-bottom:6px; }
  .bandtop .hw{ font-family:var(--display); font-size:1.3em; font-weight:600;
    color:var(--ink-strong); line-height:1.25; }
  .bandtop .hw sup{ color:var(--rubric); font-size:.62em; }
  .bandtop .gl{ color:var(--gram); font-size:.95em; }
  .bandtop .gl i{ font-style:italic; }
  .bandtop .tot{ margin-left:auto; font-family:var(--sans); font-size:.7em;
    letter-spacing:.06em; color:var(--paper); background:var(--ink-strong);
    padding:2px 8px; border-radius:2px; white-space:nowrap; }
  .bandtop .x{ cursor:pointer; color:var(--sepia); font-size:1.15em; padding:0 4px;
    align-self:center; }
  .bandtop .x:hover{ color:var(--rubric); }
  .lab{ font-family:var(--sans); font-size:.62em; text-transform:uppercase;
    letter-spacing:.12em; color:var(--sepia); display:block; margin:7px 0 1px; }
  .srctable{ margin-top:2px; }
  .band .srctable{ columns:2; column-gap:36px; }
  .band .srcrow{ break-inside:avoid; }
  .srcrow{ display:flex; gap:10px; padding:5px 0; border-top:1px solid var(--line-soft);
    align-items:baseline; }
  .srcrow .sname{ width:64px; min-width:64px; font-family:var(--sans); font-size:.7em;
    text-transform:uppercase; letter-spacing:.08em; color:var(--ink); line-height:1.5;
    cursor:help; }
  .srcrow .sname .syear{ display:block; font-size:.92em; color:var(--sepia);
    letter-spacing:.06em; }
  .srcrow .sform{ flex:1; font-style:italic; }
  .srcrow .scount{ font-family:var(--sans); font-size:.78em; color:var(--sepia);
    white-space:nowrap; }
  .qbadge{ color:var(--rubric); font-weight:700; cursor:help; font-style:normal; }
  .ex{ border-left:2px solid var(--line); padding:4px 0 4px 11px; margin:8px 0; }
  .ex .xt{ color:var(--verdigris); font-style:italic; }
  .ex .exhl{ font-weight:700; }
  .srctag{ display:inline-block; font-family:var(--sans); font-size:.6em;
    text-transform:uppercase; letter-spacing:.1em; color:var(--sepia);
    border:1px solid var(--line); padding:0 5px; border-radius:2px;
    vertical-align:2px; margin-right:6px; cursor:help; }
  .refchip{ display:inline-block; font-family:var(--sans); font-size:.66em;
    letter-spacing:.04em; color:var(--verdigris); border:1px solid var(--line);
    border-radius:2px; padding:0 7px; margin:2px 0 0 6px; cursor:pointer; }
  .refchip:hover{ border-color:var(--rubric); color:var(--rubric); }
  .lnk{ color:var(--ink-strong); text-decoration:underline;
    text-decoration-color:var(--line); text-underline-offset:3px; cursor:pointer;
    transition:color .12s ease, text-decoration-color .12s ease; }
  .lnk:hover{ color:var(--rubric); text-decoration-color:var(--rubric); }
  .warnline{ color:var(--rubric); font-family:var(--sans); font-size:.78em;
    margin-top:7px; }
  .rawbox{ margin-top:9px; }
  .rawbox summary{ font-family:var(--sans); font-size:.68em; text-transform:uppercase;
    letter-spacing:.1em; color:var(--sepia); cursor:pointer; }
  .rawbox pre{ font-family:var(--serif); font-size:.88em; white-space:pre-wrap;
    background:var(--paper); border:1px dashed var(--line); padding:8px 12px;
    margin-top:5px; color:var(--gram); }
  .btnrow{ margin-top:9px; }
  .pdf{ font-family:var(--sans); font-size:.65em; letter-spacing:.04em; color:var(--sepia);
    border:1px dashed var(--line); border-radius:2px; padding:2px 8px; cursor:pointer;
    background:transparent; white-space:nowrap; margin-right:6px; }
  .pdf:hover{ background:var(--paper); color:var(--ink); }
  .empty-ws{ color:var(--sepia); font-style:italic; padding:48px; max-width:56ch; }
  .wstools{ font-family:var(--sans); font-size:.7em; color:var(--sepia); margin:0 0 10px;
    display:flex; gap:10px; align-items:center; }
  .wstools .pdf{ margin:0; }
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
    <h1>APTSK <span class="rdash">&mdash;</span> Eesti vanema piiblitõlke sõnastik 1600&ndash;1739</h1>
    <span class="tagline">@@COUNT@@ artiklit &middot;&middot;&middot; genereeritud @@GENERATED@@</span>
    <span class="legend"><span class="dot">&#9679;</span> marks entries whose print shows an anomaly (see warnings inside the entry)</span>
  </div>
  <nav id="modes"></nav>
</header>

<div class="adv" id="advpanel">
  <h2>Advanced search <span class="rq">&para;</span></h2>
  <div class="advgrid">
    <div class="advcol" style="flex:1 1 420px">
      <h3>Conditions — <select id="combine"><option value="AND">match ALL</option><option value="OR">match ANY</option></select></h3>
      <div id="crows"></div>
      <button class="btn2" id="addrow">+ add condition</button>
      <div style="margin-top:8px">
        <label class="ck"><input type="checkbox" id="cs"> case sensitive</label>
        <label class="ck"><input type="checkbox" id="fold"> ignore diacritics (t&auml;hh &rarr; tahh)</label>
      </div>
    </div>
    <div class="advcol">
      <h3>Source presence <span title="click a source to cycle: neutral → must have (red) → must NOT have (struck)">&#9432;</span></h3>
      <div class="srcpick" id="srcpick"></div>
      <h3 style="margin-top:12px">Total frequency</h3>
      <div class="numrange">
        <input type="text" id="tmin" placeholder="min"> &ndash;
        <input type="text" id="tmax" placeholder="max">
      </div>
    </div>
    <div class="advcol">
      <h3>Flags</h3>
      <div id="flagsel"></div>
    </div>
  </div>
  <div>
    <button class="btn2 primary" id="applyadv">Apply</button>
    <button class="btn2" id="clearadv">Clear</button>
    <button class="btn2" id="expjson">&darr; JSON of results</button>
    <button class="btn2" id="expcsv">&darr; CSV of results</button>
  </div>
  <div class="advsum" id="advsum"></div>
</div>

<div class="layout">
  <div class="browser">
    <div class="bcount" id="bcount">Unpacking data&hellip;</div>
    <div id="items"></div>
    <div class="pager" id="pager"></div>
  </div>
  <div class="workspace" id="workspace">
    <div class="empty-ws">Pick an entry on the left to open it here. Every
    cross-reference (<i>vt</i>, <i>vt ka</i>) is clickable, as is each verse
    reference &mdash; the latter finds all entries quoting that verse.
    Open the <b>Advanced search</b> to query any field of the data, including
    the raw printed lines, with regex, wildcards and numeric ranges.</div>
  </div>
</div>

<script id="data" type="application/octet-stream">@@PAYLOAD@@</script>
<script>
'use strict';
let ENTRIES=[], META={}, BYID={}, BYHW={};
const SRC_ORDER=['Ml','Rs','St','GtVT','GtUT','Gt','Bl','WT','Vr','Mn','PR'];
const SRC_YEAR={Ml:'1600–1606',Rs:'1632',St:'1638',GtVT:'käsikiri',GtUT:'käsikiri',
  Gt:'käsikiri',Bl:'käsikiri',WT:'1686',Vr:'käsikiri',Mn:'käsikiri',PR:'1739'};
const esc=s=>(s==null?'':String(s)).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fold=s=>s.normalize('NFD').replace(/[̀-ͯ]/g,'');
const PAGE=150;

let state={ letter:'all', q:'', match:'contains', refMode:'all', flaggedOnly:false,
  page:0, panels:[], advOn:false, advOpen:false };
let adv={ combine:'AND', rows:[], srcs:{}, flags:{}, cs:false, fold:false, tmin:'', tmax:'' };
let LETTERS=[];

/* ---------------- field registry: every field of the JSON ---------------- */
function srcSet(e){ const s=new Set();
  (e.source_counts||[]).forEach(x=>s.add(x.source));
  (e.examples||[]).forEach(x=>{ if(x.source) s.add(x.source); }); return s; }
function allText(e){ if(!e._all) e._all=JSON.stringify(e); return [e._all]; }
const FIELDS=[
 ['*','★ all content (whole record)','text', allText],
 ['headword','headword','text', e=>e.headwords&&e.headwords.length?e.headwords:(e.headword?[e.headword]:[])],
 ['homonym','homonym number','num', e=>e.homonym==null?[]:[e.homonym]],
 ['gloss','gloss <…>','text', e=>e.gloss?[e.gloss]:[]],
 ['letter','letter section','text', e=>e.letter?[e.letter]:[]],
 ['id','entry id','text', e=>[e.id]],
 ['total_count','total frequency','num', e=>e.total_count==null?[]:[e.total_count]],
 ['sc.source','statistics: source','text', e=>(e.source_counts||[]).map(x=>x.source)],
 ['sc.form','statistics: subform','text', e=>(e.source_counts||[]).flatMap(x=>(x.forms&&x.forms.length?x.forms:(x.form?[x.form]:[])))],
 ['sc.form_raw','statistics: subform (raw, with ?/*)','text', e=>(e.source_counts||[]).map(x=>x.form_raw).filter(Boolean)],
 ['sc.count','statistics: per-source count','num', e=>(e.source_counts||[]).map(x=>x.count).filter(v=>v!=null)],
 ['ex.source','example: source','text', e=>(e.examples||[]).map(x=>x.source).filter(Boolean)],
 ['ex.text','example: quotation text','text', e=>(e.examples||[]).map(x=>x.text)],
 ['ex.highlighted','example: highlighted form','text', e=>(e.examples||[]).flatMap(x=>x.highlighted||[])],
 ['ex.references','example: verse reference','text', e=>(e.examples||[]).flatMap(x=>x.references||[])],
 ['see','vt target (reference entries)','text', e=>(e.see||[]).map(x=>x.form)],
 ['see_also','vt ka items','text', e=>e.see_also?[].concat(e.see_also.same_form||[],e.see_also.other_form||[]):[]],
 ['see_also.raw','vt ka (raw line)','text', e=>e.see_also&&e.see_also.raw?[e.see_also.raw]:[]],
 ['gloss_rich','gloss segments','text', e=>(e.gloss_rich||[]).map(x=>x.text)],
 ['pages','PDF page','num', e=>e.pages||[]],
 ['raw','raw printed lines','text', e=>e.raw_lines||[]],
 ['warnings','warnings','text', e=>e.warnings||[]],
 ['unparsed','unparsed remainder','text', e=>e.unparsed?[e.unparsed]:[]],
 ['hd.form','headword detail: form','text', e=>(e.headword_details||[]).map(x=>x.form)],
 ['reference_only','is a vt-reference entry','bool', e=>[!!e.reference_only]],
 ['counts_omitted','counts omitted in print','bool', e=>[!!e.counts_omitted]],
 ['has_warnings','has warnings','bool', e=>[!!(e.warnings&&e.warnings.length)]],
 ['uncertain','has ?-marked (uncertain) subform','bool', e=>[(e.source_counts||[]).some(x=>x.uncertain)]],
 ['orig','has *-marked (orig. spelling) subform','bool', e=>[(e.source_counts||[]).some(x=>x.original_spelling)]],
 ['foreign','has foreign (bold-italic) component','bool', e=>[(e.headword_details||[]).some(x=>x.foreign)]],
 ['has_examples','has examples','bool', e=>[!!(e.examples&&e.examples.length)]],
 ['has_see_also','has vt ka','bool', e=>[!!e.see_also]],
];
const FIELDMAP={}; FIELDS.forEach(f=>FIELDMAP[f[0]]={k:f[0],label:f[1],type:f[2],get:f[3]});
const OPS={
 text:[['contains','contains'],['ncontains','does not contain'],['starts','starts with'],
   ['ends','ends with'],['exact','is exactly'],['wild','wildcard *?'],
   ['regex','regex'],['empty','is empty'],['nonempty','is not empty']],
 num:[['eq','='],['ne','≠'],['gt','>'],['ge','≥'],['lt','<'],['le','≤'],
   ['empty','is empty'],['nonempty','is not empty']],
 bool:[['true','is yes'],['false','is no']]
};
const FLAGFIELDS=['reference_only','counts_omitted','has_warnings','uncertain','orig',
  'foreign','has_examples','has_see_also'];

/* ---------------- matching ---------------- */
function globToRegex(g,flags){
  const body=g.replace(/[.+^${}()|[\]\\]/g,'\\$&').replace(/\*/g,'.*').replace(/\?/g,'.');
  return new RegExp('^'+body+'$',flags);
}
function nrm(s){ s=String(s); if(!adv.cs) s=s.toLowerCase(); if(adv.fold) s=fold(s); return s; }
function evalRow(e,row){
  const f=FIELDMAP[row.f]; if(!f) return true;
  const vals=f.get(e);
  if(f.type==='bool') return row.op==='true'? vals.some(v=>v===true) : vals.some(v=>v===false)||!vals.length;
  if(f.type==='num'){
    if(row.op==='empty') return !vals.length;
    if(row.op==='nonempty') return !!vals.length;
    const n=parseFloat(String(row.v).replace(',','.')); if(isNaN(n)) return true;
    const cmp={eq:v=>v===n, ne:v=>v!==n, gt:v=>v>n, ge:v=>v>=n, lt:v=>v<n, le:v=>v<=n}[row.op];
    return row.op==='ne'? vals.every(cmp) : vals.some(cmp);
  }
  if(row.op==='empty') return !vals.length||vals.every(v=>!String(v).trim());
  if(row.op==='nonempty') return vals.some(v=>String(v).trim());
  const q=nrm(row.v||''); if(!q&&row.op!=='regex') return true;
  const strs=vals.map(nrm);
  switch(row.op){
    case 'contains': return strs.some(s=>s.includes(q));
    case 'ncontains': return !strs.some(s=>s.includes(q));
    case 'starts': return strs.some(s=>s.startsWith(q));
    case 'ends': return strs.some(s=>s.endsWith(q));
    case 'exact': return strs.some(s=>s===q);
    case 'wild': try{ const rx=globToRegex(row.v,adv.cs?'':'i'); return vals.some(v=>rx.test(String(v))); }catch(_){ return false; }
    case 'regex': try{ const rx=new RegExp(row.v,adv.cs?'':'i'); return vals.some(v=>rx.test(String(v))); }catch(_){ return false; }
  }
  return true;
}
function advHit(e){
  for(const s in adv.srcs){ const have=srcSet(e).has(s);
    if(adv.srcs[s]===1&&!have) return false;
    if(adv.srcs[s]===-1&&have) return false; }
  for(const k in adv.flags){ const v=adv.flags[k]; if(v==='any') continue;
    const got=FIELDMAP[k].get(e)[0]===true;
    if(v==='yes'&&!got) return false; if(v==='no'&&got) return false; }
  const tmin=parseFloat(adv.tmin), tmax=parseFloat(adv.tmax);
  if(!isNaN(tmin)&&!(e.total_count>=tmin)) return false;
  if(!isNaN(tmax)&&!(e.total_count<=tmax)) return false;
  const active=adv.rows.filter(r=>r.f&&(r.v!==''||['empty','nonempty','true','false'].includes(r.op)));
  if(active.length){
    if(adv.combine==='AND'){ for(const r of active) if(!evalRow(e,r)) return false; }
    else { let any=false; for(const r of active) if(evalRow(e,r)){ any=true; break; } if(!any) return false; }
  }
  return true;
}
function quickMatcher(){
  const q=fold(state.q.toLowerCase()); if(!q) return null;
  const m=state.match;
  if(m==='starts') return h=>h.startsWith(q);
  if(m==='ends') return h=>h.endsWith(q);
  if(m==='wild'){ try{ const rx=globToRegex(q,'i'); return h=>rx.test(h);}catch(_){ return ()=>false; } }
  if(m==='regex'){ try{ const rx=new RegExp(q,'i'); return h=>rx.test(h);}catch(_){ return ()=>false; } }
  return h=>h.includes(q);
}
function quickHay(e){
  if(!e._hay){
    const parts=[].concat(e.headwords||[], e.gloss?[e.gloss]:[],
      (e.source_counts||[]).flatMap(x=>x.forms||[]),
      (e.see||[]).map(x=>x.form),
      e.see_also?(e.see_also.same_form||[]).concat(e.see_also.other_form||[]):[]);
    e._hay=parts.filter(Boolean).map(x=>fold(String(x).toLowerCase()));
  }
  return e._hay;   // array: anchors (^ $, starts/ends) apply per component
}
function currentList(){
  const qm=quickMatcher();
  const out=[];
  for(const e of ENTRIES){
    if(state.letter!=='all'&&e.letter!==state.letter) continue;
    if(state.refMode==='full'&&e.reference_only) continue;
    if(state.refMode==='refs'&&!e.reference_only) continue;
    if(state.flaggedOnly&&!(e.warnings&&e.warnings.length)) continue;
    if(qm&&!quickHay(e).some(qm)) continue;
    if(state.advOn&&!advHit(e)) continue;
    out.push(e);
  }
  return out;
}

/* ---------------- top nav ---------------- */
function renderModes(){
  const nav=document.getElementById('modes');
  let h=`<button data-l="all" class="letter ${state.letter==='all'?'active':''}">K&otilde;ik</button>`;
  for(const L of LETTERS)
    h+=`<button data-l="${esc(L)}" class="letter ${state.letter===L?'active':''}">${esc(L)}</button>`;
  h+=`<input id="search" placeholder="quick search: headword, subform, gloss&hellip;" value="${esc(state.q)}">`;
  h+=`<select id="match">
    <option value="contains"${state.match==='contains'?' selected':''}>contains</option>
    <option value="starts"${state.match==='starts'?' selected':''}>starts</option>
    <option value="ends"${state.match==='ends'?' selected':''}>ends</option>
    <option value="wild"${state.match==='wild'?' selected':''}>wildcard</option>
    <option value="regex"${state.match==='regex'?' selected':''}>regex</option></select>`;
  h+=`<select id="refmode">
    <option value="all"${state.refMode==='all'?' selected':''}>all entries</option>
    <option value="full"${state.refMode==='full'?' selected':''}>full articles</option>
    <option value="refs"${state.refMode==='refs'?' selected':''}>vt-references</option></select>`;
  h+=`<button id="flagged" class="flagtog ${state.flaggedOnly?'on':''}" title="only entries with print-anomaly warnings">&#9679; flagged</button>`;
  h+=`<button id="advtog" class="flagtog ${state.advOpen?'on':''}">Advanced search</button>`;
  h+=`<button id="lucky" title="open a random entry from the current result set">&#9884; random</button>`;
  h+=`<button id="about">About</button>`;
  nav.innerHTML=h;
  nav.querySelectorAll('button.letter').forEach(b=>b.onclick=()=>{ state.letter=b.dataset.l; state.page=0; render(); });
  const si=document.getElementById('search');
  si.oninput=()=>{ state.q=si.value.trim(); state.page=0; renderList();
    si.classList.toggle('badq', state.match==='regex'&&!validRx(state.q)); };
  si.onkeydown=ev=>{ if(ev.key==='Enter'){ const l=currentList(); if(l.length===1) openEntry(l[0].id); } };
  document.getElementById('match').onchange=ev=>{ state.match=ev.target.value; state.page=0; renderList(); };
  document.getElementById('refmode').onchange=ev=>{ state.refMode=ev.target.value; state.page=0; renderList(); };
  document.getElementById('flagged').onclick=()=>{ state.flaggedOnly=!state.flaggedOnly; state.page=0; render(); };
  document.getElementById('advtog').onclick=()=>{ state.advOpen=!state.advOpen;
    document.getElementById('advpanel').classList.toggle('open',state.advOpen); renderModes(); };
  document.getElementById('lucky').onclick=()=>{ const l=currentList();
    if(l.length) openEntry(l[Math.floor(Math.random()*l.length)].id); };
  document.getElementById('about').onclick=openAbout;
}
function validRx(q){ try{ new RegExp(q); return true; }catch(_){ return false; } }

/* ---------------- advanced panel ---------------- */
function rowTemplate(row,i){
  const f=FIELDMAP[row.f]||FIELDMAP['*'];
  let h=`<div class="crow" data-i="${i}"><select class="rf">`;
  for(const fd of FIELDS) h+=`<option value="${fd[0]}"${fd[0]===row.f?' selected':''}>${esc(fd[1])}</option>`;
  h+=`</select><select class="rop">`;
  for(const op of OPS[f.type]) h+=`<option value="${op[0]}"${op[0]===row.op?' selected':''}>${esc(op[1])}</option>`;
  h+=`</select>`;
  if(f.type!=='bool'&&!['empty','nonempty'].includes(row.op))
    h+=`<input type="text" class="rv" value="${esc(row.v)}" placeholder="${f.type==='num'?'number':'value'}">`;
  h+=`<button class="del" title="remove condition">&times;</button></div>`;
  return h;
}
function renderAdv(){
  const box=document.getElementById('crows');
  if(!adv.rows.length) adv.rows.push({f:'*',op:'contains',v:''});
  box.innerHTML=adv.rows.map(rowTemplate).join('');
  box.querySelectorAll('.crow').forEach(cr=>{
    const i=+cr.dataset.i, row=adv.rows[i];
    cr.querySelector('.rf').onchange=ev=>{ row.f=ev.target.value;
      row.op=OPS[FIELDMAP[row.f].type][0][0]; renderAdv(); };
    cr.querySelector('.rop').onchange=ev=>{ row.op=ev.target.value; renderAdv(); };
    const rv=cr.querySelector('.rv'); if(rv) rv.oninput=()=>{ row.v=rv.value;
      rv.classList.toggle('badq', row.op==='regex'&&!validRx(row.v)); };
    cr.querySelector('.del').onclick=()=>{ adv.rows.splice(i,1); renderAdv(); };
  });
  const sp=document.getElementById('srcpick');
  sp.innerHTML=SRC_ORDER.map(s=>{
    const st=adv.srcs[s]||0;
    return `<span class="chip ${st===1?'req':st===-1?'exc':''}" data-s="${s}" title="${esc((META.text_source_abbreviations||{})[s]||s)}">${s}${st===1?' &#10003;':st===-1?' &#10007;':''}</span>`;
  }).join('');
  sp.querySelectorAll('.chip').forEach(c=>c.onclick=()=>{
    const s=c.dataset.s, cur=adv.srcs[s]||0;
    if(cur===0) adv.srcs[s]=1; else if(cur===1) adv.srcs[s]=-1; else delete adv.srcs[s];
    renderAdv(); });
  const fs=document.getElementById('flagsel');
  fs.innerHTML=FLAGFIELDS.map(k=>`<div class="fl"><span>${esc(FIELDMAP[k].label)}</span>
    <select data-k="${k}"><option value="any"${(adv.flags[k]||'any')==='any'?' selected':''}>any</option>
    <option value="yes"${adv.flags[k]==='yes'?' selected':''}>yes</option>
    <option value="no"${adv.flags[k]==='no'?' selected':''}>no</option></select></div>`).join('');
  fs.querySelectorAll('select').forEach(s=>s.onchange=()=>{ adv.flags[s.dataset.k]=s.value; });
}
function wireAdv(){
  document.getElementById('combine').onchange=ev=>{ adv.combine=ev.target.value; };
  document.getElementById('cs').onchange=ev=>{ adv.cs=ev.target.checked; };
  document.getElementById('fold').onchange=ev=>{ adv.fold=ev.target.checked; };
  document.getElementById('tmin').oninput=ev=>{ adv.tmin=ev.target.value; };
  document.getElementById('tmax').oninput=ev=>{ adv.tmax=ev.target.value; };
  document.getElementById('addrow').onclick=()=>{ adv.rows.push({f:'*',op:'contains',v:''}); renderAdv(); };
  document.getElementById('applyadv').onclick=()=>{ state.advOn=true; state.page=0; render();
    const n=currentList().length;
    document.getElementById('advsum').innerHTML=`Advanced filter is <b>active</b> &mdash; ${n} matching entr${n===1?'y':'ies'}.`; };
  document.getElementById('clearadv').onclick=()=>{ adv={combine:'AND',rows:[],srcs:{},flags:{},cs:false,fold:false,tmin:'',tmax:''};
    state.advOn=false; state.page=0;
    document.getElementById('cs').checked=false; document.getElementById('fold').checked=false;
    document.getElementById('tmin').value=''; document.getElementById('tmax').value='';
    document.getElementById('combine').value='AND';
    document.getElementById('advsum').textContent='';
    renderAdv(); render(); };
  document.getElementById('expjson').onclick=()=>{
    const list=currentList().map(e=>{ const c={...e}; delete c._all; delete c._hay; delete c._i; return c; });
    download('aptsk_results.json', JSON.stringify(list,null,1), 'application/json'); };
  document.getElementById('expcsv').onclick=()=>{
    const cq=v=>'"'+String(v==null?'':v).replace(/"/g,'""')+'"';
    const head=['id','letter','headword','homonym','gloss','total_count','reference_only',
      'sources','n_examples','pages','warnings'];
    const lines=[head.join(',')];
    for(const e of currentList())
      lines.push([e.id,e.letter,(e.headwords||[]).join(' | '),e.homonym??'',e.gloss??'',
        e.total_count??'',e.reference_only,
        (e.source_counts||[]).map(x=>x.source+(x.form?' '+x.form:'')+(x.count!=null?' '+x.count:'')).join('; '),
        (e.examples||[]).length,(e.pages||[]).join(' '),(e.warnings||[]).join('; ')].map(cq).join(','));
    download('aptsk_results.csv','﻿'+lines.join('\n'),'text/csv'); };
}
function download(name,text,mime){
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([text],{type:mime+';charset=utf-8'}));
  a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),4000);
}

/* ---------------- browser rail ---------------- */
function hwHtml(e){
  let h=esc(e.headwords&&e.headwords.length?e.headwords.join(', '):e.headword);
  if(e.homonym) h+=`<sup>${e.homonym}</sup>`;
  return h;
}
function renderList(){
  const list=currentList();
  const pages=Math.max(1,Math.ceil(list.length/PAGE));
  if(state.page>=pages) state.page=pages-1;
  const slice=list.slice(state.page*PAGE,(state.page+1)*PAGE);
  document.getElementById('bcount').innerHTML=
    `${list.length} entr${list.length===1?'y':'ies'}${state.advOn?' &middot; advanced filter on':''}`;
  document.getElementById('items').innerHTML=slice.map(e=>{
    const flag=e.warnings&&e.warnings.length?' <span class="dot">&#9679;</span>':'';
    const sub=e.reference_only
      ? 'vt '+esc((e.see||[]).map(x=>x.form).join(', '))
      : esc(e.gloss||((e.source_counts||[]).map(x=>x.source).join(' ')));
    const cnt=e.total_count!=null?`<span class="cnt">${e.total_count}</span>`:'';
    const sel=state.panels[0]===e.id?' sel':'';
    return `<div class="item${sel}" data-id="${esc(e.id)}">${cnt}<span class="code">${hwHtml(e)}</span>${flag}<div class="sub">${sub}</div></div>`;
  }).join('');
  document.querySelectorAll('.item').forEach(it=>it.onclick=()=>openEntry(it.dataset.id));
  const pg=document.getElementById('pager');
  if(pages>1){
    pg.innerHTML=`<button id="pprev"${state.page===0?' disabled':''}>&larr;</button>
      <span class="grp">page <input class="jump" id="pjump" value="${state.page+1}"> / ${pages}</span>
      <button id="pnext"${state.page>=pages-1?' disabled':''}>&rarr;</button>`;
    document.getElementById('pprev').onclick=()=>{ state.page--; renderList(); };
    document.getElementById('pnext').onclick=()=>{ state.page++; renderList(); };
    const j=document.getElementById('pjump');
    j.onchange=()=>{ const v=parseInt(j.value,10);
      if(v>=1&&v<=pages){ state.page=v-1; renderList(); } };
  } else pg.innerHTML='';
}

/* ---------------- entry rendering ---------------- */
function resolveWord(word,hom){
  const k=word.toLowerCase();
  let ids=BYHW[k+(hom||'')]||BYHW[k];
  return ids&&ids.length?ids[0]:null;
}
function linkWord(w,hom){
  const id=resolveWord(w,hom);
  const label=esc(w)+(hom?`<sup>${hom}</sup>`:'');
  return id?`<span class="lnk" data-go="${esc(id)}">${label}</span>`:label;
}
function glossHtml(e){
  if(!e.gloss_rich||!e.gloss_rich.length) return e.gloss?esc(e.gloss):'';
  return e.gloss_rich.map(p=>p.italic?`<i>${esc(p.text)}</i>`:esc(p.text)).join('');
}
function exText(x){
  let t=x.text;
  const hls=[...new Set(x.highlighted||[])].filter(Boolean).sort((a,b)=>b.length-a.length);
  hls.forEach(h=>{ t=t.split(h).join('\u0001'+h+'\u0002'); });
  return esc(t).replace(/\u0001/g,'<b class="exhl">').replace(/\u0002/g,'</b>');
}
function bandHtml(e){
  const flag=e.warnings&&e.warnings.length;
  let h=`<div class="band${flag?' flag':''}${e.reference_only?' refband':''}" data-band="${esc(e.id)}">`;
  h+=`<div class="spine"><span class="letterbig">${esc(e.letter||'')}</span>
      <span class="syear">PDF lk ${(e.pages||[]).join('–')}</span>
      <span class="syear">${esc(e.id)}</span></div><div class="bandmain">`;
  h+=`<div class="bandtop"><span class="hw">${hwHtml(e)}</span>`;
  const gl=glossHtml(e); if(gl) h+=`<span class="gl">&lsaquo;${gl}&rsaquo;</span>`;
  if(e.total_count!=null) h+=`<span class="tot">${e.total_count}&times;</span>`;
  else if(e.counts_omitted) h+=`<span class="tot" title="the dictionary deliberately prints no frequencies for this entry">&mdash;</span>`;
  h+=`<span class="x" data-x="${esc(e.id)}" title="close">&times;</span></div>`;

  if(e.reference_only){
    h+=`<span class="lab">vt</span><div>`+(e.see||[]).map(t=>{
      let m=linkWord(t.form,t.homonym);
      if(t.original_spelling) m+='<span class="qbadge" title="original orthography">*</span>';
      return m; }).join(', ')+`</div>`;
  } else {
    if(e.source_counts&&e.source_counts.length){
      h+=`<span class="lab">Attestation by source</span><div class="srctable">`;
      for(const s of e.source_counts){
        h+=`<div class="srcrow"><span class="sname" title="${esc((META.text_source_abbreviations||{})[s.source]||'')}">${esc(s.source)}<span class="syear">${esc(SRC_YEAR[s.source]||'')}</span></span>
        <span class="sform">${s.form?esc(s.form):'<span style="font-style:normal;color:var(--sepia)">= headword</span>'}`
        +(s.uncertain?'<span class="qbadge" title="lemma uncertain — only inflected forms attested">?</span>':'')
        +(s.original_spelling?'<span class="qbadge" title="given in original orthography (spelling unreliable)">*</span>':'')
        +`</span><span class="scount">${s.count!=null?s.count+'×':'—'}</span></div>`;
      }
      h+=`</div>`;
    }
    if(e.examples&&e.examples.length){
      h+=`<span class="lab">Examples</span>`;
      for(const x of e.examples){
        h+=`<div class="ex">`;
        h+=x.source?`<span class="srctag" title="${esc((META.text_source_abbreviations||{})[x.source]||'')}">${esc(x.source)}</span>`
                   :`<span class="srctag" title="source label missing in print">?</span>`;
        h+=`<span class="xt">${exText(x)}</span>`;
        for(const r of (x.references||[]))
          h+=`<span class="refchip" data-ref="${esc(r)}" title="find all entries quoting ${esc(r)}">${esc(r)}</span>`;
        h+=`</div>`;
      }
    }
    if(e.see_also){
      h+=`<span class="lab">vt ka</span><div>`;
      h+=(e.see_also.same_form||[]).map(w=>seeAlsoChip(w)).join(' ');
      if((e.see_also.other_form||[]).length)
        h+=` <span style="color:var(--sepia)">&brvbar;</span> `+
           e.see_also.other_form.map(w=>seeAlsoChip(w)).join(' ');
      h+=`</div>`;
    }
  }
  if(e.unparsed) h+=`<div class="warnline">unparsed remainder kept verbatim: &ldquo;${esc(e.unparsed)}&rdquo;</div>`;
  if(flag) h+=`<div class="warnline">&#9679; ${e.warnings.map(esc).join(' &middot; ')}</div>`;
  h+=`<details class="rawbox"><summary>printed lines (verbatim)</summary><pre>${esc((e.raw_lines||[]).join('\n'))}</pre></details>`;
  h+=`<div class="btnrow"><button class="pdf" data-cite="${esc(e.id)}">copy citation</button></div>`;
  h+=`</div></div>`;
  return h;
}
function seeAlsoChip(w){
  const m=/^(.*?)(\d+)?$/.exec(w); const word=m[1], hom=m[2]?+m[2]:null;
  if(word.endsWith('-'))
    return `<span class="chip" data-pref="${esc(word.slice(0,-1))}" title="compound prefix — click to search compounds">${esc(w)}</span>`;
  const id=resolveWord(word,hom);
  return `<span class="chip${id?'':' weak'}" ${id?`data-go="${esc(id)}"`:''}>${esc(w)}</span>`;
}
function renderWorkspace(){
  const ws=document.getElementById('workspace');
  if(!state.panels.length){
    ws.innerHTML=`<div class="empty-ws">Pick an entry on the left to open it here.</div>`;
    return;
  }
  let h='';
  if(state.panels.length>1)
    h+=`<div class="wstools"><button class="pdf" id="closeall">close all (${state.panels.length})</button></div>`;
  h+=state.panels.map(id=>BYID[id]?bandHtml(BYID[id]):'').join('');
  ws.innerHTML=h;
  const ca=document.getElementById('closeall');
  if(ca) ca.onclick=()=>{ state.panels=[]; renderWorkspace(); renderList(); };
  ws.querySelectorAll('[data-x]').forEach(x=>x.onclick=()=>{
    state.panels=state.panels.filter(p=>p!==x.dataset.x); renderWorkspace(); renderList(); });
  ws.querySelectorAll('[data-go]').forEach(l=>l.onclick=()=>openEntry(l.dataset.go));
  ws.querySelectorAll('[data-ref]').forEach(c=>c.onclick=()=>searchRef(c.dataset.ref));
  ws.querySelectorAll('[data-pref]').forEach(c=>c.onclick=()=>{
    state.q=c.dataset.pref; state.match='starts'; state.page=0; renderModes(); renderList(); });
  ws.querySelectorAll('[data-cite]').forEach(b=>b.onclick=()=>{
    const e=BYID[b.dataset.cite];
    const cite=`${(e.headwords||[]).join(', ')}${e.homonym?'¹²³'.charAt(e.homonym-1)||('('+e.homonym+')'):''} — Eesti vanema piiblitõlke sõnastik 1600–1739. EKSA 2025, PDF lk ${(e.pages||[]).join('–')}.`;
    navigator.clipboard&&navigator.clipboard.writeText(cite);
    b.textContent='copied ✓'; setTimeout(()=>b.textContent='copy citation',1400); });
}
function openEntry(id){
  if(!BYID[id]) return;
  state.panels=state.panels.filter(p=>p!==id); state.panels.unshift(id);
  renderWorkspace(); renderList();
  document.getElementById('workspace').scrollTop=0;
}
function searchRef(ref){
  state.advOpen=true; state.advOn=true;
  document.getElementById('advpanel').classList.add('open');
  adv.rows=[{f:'ex.references',op:'exact',v:ref}];
  adv.srcs={}; adv.flags={}; adv.tmin=''; adv.tmax='';
  state.letter='all'; state.q=''; state.page=0;
  state.flaggedOnly=false; state.refMode='all';
  renderModes(); renderAdv(); render();
  const n=currentList().length;
  document.getElementById('advsum').innerHTML=
    `Showing all entries quoting <b>${esc(ref)}</b> &mdash; ${n} entr${n===1?'y':'ies'}.`;
}

/* ---------------- about band ---------------- */
function openAbout(){
  const ws=document.getElementById('workspace');
  const m=META;
  let h=`<div class="band"><div class="spine"><span class="letterbig">&sect;</span>
    <span class="syear">about</span></div><div class="bandmain">
    <div class="bandtop"><span class="hw">${esc(m.title||'APTSK')}</span>
    <span class="tot">${ENTRIES.length} entries</span></div>
    <p style="max-width:70ch">${esc(m.publisher||'')} &middot; Editors: ${(m.editors||[]).map(esc).join(', ')}.
    Extracted from <i>${esc(m.source_pdf||'')}</i> (PDF pages ${(m.dictionary_pages||[]).join('–')})
    on ${esc(m.extracted_at||'')} by ${esc(m.author_of_extraction||'')}.
    Every printed line is preserved verbatim inside each entry (&ldquo;printed lines&rdquo; fold-out);
    the validation suite proves character-exact coverage of the whole dictionary body.</p>
    <span class="lab">Sources</span><div class="srctable">`;
  for(const s of SRC_ORDER){ const d=(m.text_source_abbreviations||{})[s]; if(!d) continue;
    h+=`<div class="srcrow"><span class="sname">${s}<span class="syear">${esc(SRC_YEAR[s]||'')}</span></span>
      <span class="sform" style="font-style:normal">${esc(d)}</span></div>`; }
  h+=`</div><span class="lab">Subform markers</span>
    <p><span class="qbadge">?</span> ${esc((m.subform_markers||{})['?']||'')} &nbsp;&middot;&nbsp;
    <span class="qbadge">*</span> ${esc((m.subform_markers||{})['*']||'')}</p>
    <span class="lab">Letters</span><p>`+
    LETTERS.map(L=>`${esc(L)}&thinsp;<span style="color:var(--sepia)">${(m.letters||{})[L]||''}</span>`).join(' &middot; ')+
    `</p></div></div>`;
  ws.innerHTML=h+ws.innerHTML;
}

/* ---------------- boot ---------------- */
function render(){ renderModes(); renderList(); }
function init(){
  ENTRIES.forEach((e,i)=>{ e._i=i; BYID[e.id]=e;
    const hws=(e.headwords&&e.headwords.length?e.headwords:[e.headword]).filter(Boolean);
    for(const hw of hws){
      const k=hw.toLowerCase();
      (BYHW[k+(e.homonym||'')]=BYHW[k+(e.homonym||'')]||[]).push(e.id);
      if(e.homonym)(BYHW[k]=BYHW[k]||[]).push(e.id);
    }
    if(!LETTERS.includes(e.letter)) LETTERS.push(e.letter);
  });
  renderAdv(); wireAdv(); render();
}
async function boot(){
  try{
    if(typeof DecompressionStream==='undefined')
      throw new Error('this viewer needs a modern browser (DecompressionStream)');
    const b64=document.getElementById('data').textContent.trim();
    const bin=atob(b64); const bytes=new Uint8Array(bin.length);
    for(let i=0;i<bin.length;i++) bytes[i]=bin.charCodeAt(i);
    const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    const text=await new Response(stream).text();
    const D=JSON.parse(text);
    ENTRIES=D.entries||[]; META=D.meta||{};
    init();
  }catch(err){
    document.getElementById('bcount').textContent='Failed to load data: '+err.message;
  }
}
boot();
</script>
</body>
</html>
"""

html = (TEMPLATE
        .replace("@@PAYLOAD@@", b64)
        .replace("@@GENERATED@@", generated)
        .replace("@@COUNT@@", count))
OUT.write_text(html, encoding="utf-8")
print(f"entries embedded : {len(payload['entries'])}")
print(f"payload          : {len(raw)/1e6:.1f} MB json -> {len(b64)/1e6:.1f} MB base64(gzip)")
print(f"written          : {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")

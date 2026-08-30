# Created: 2026-07-13 12-20-53
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
"""Extensive lemma/linkage review of AMT-Master_annotated.json, cross-checked
against the dataset embedded in LexLex (the canonical viewer).

Verifies JSON <-> viewer sync, then reports: attestation & linkage stats per
source, coverage distributions, category breakdowns, attention flags, omissions
(attested-but-unlinked cells, missing fields, ??? cells, cross-source-count
mismatches) and a full per-lemma inventory.

Output: Katus-ALUSANDMED/Review-JSON-AMT/AMT-Master_lemma-report_<YYYYMMDD>.md
"""
import collections
import datetime
import json
import os

MASTER_PATH = "Katus-ALUSANDMED/json-all/AMT-Master_annotated.json"
VIEWER_PATH = "Katus-ALUSANDMED/global-view/LexLex.html"
OUT_DIR = "Katus-ALUSANDMED/Review-JSON-AMT"

SOURCES = ["Stahl-1637", "Gutslaff-1648", "Göseken-1660",
           "Vestring-17XX", "Helle-1732", "Hupel-1780-est-ger"]
SHORT = {"Stahl-1637": "St", "Gutslaff-1648": "Gu", "Göseken-1660": "Gö",
         "Vestring-17XX": "Ve", "Helle-1732": "He", "Hupel-1780-est-ger": "Hu"}
PLACE = {"---", "???", "NULL", "", " "}          # not a real attestation

MASTER_FIELDS = ["Amt-Cat", "Sem-Cat", "Teema", "Sugu", "DEF_en", "DEF_et",
                 "Cross-source count", "Comment-1", "Comment-2", "Comment-3"]


def cellstr(c, k):
    v = c.get(k, "NULL")
    return v if isinstance(v, str) else "NULL"


def md_esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def scan_q(e):
    """Field names whose value contains ??? (mirrors build_viewer.scan_q)."""
    flags = []
    for k, v in e.items():
        if isinstance(v, str) and "???" in v:
            flags.append(k)
        elif isinstance(v, list):
            for m in v:
                if isinstance(m, dict) and any(
                        isinstance(x, str) and "???" in x for x in m.values()):
                    flags.append("mwu")
                    break
    return flags


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "–"


def load_viewer_data(path):
    html = open(path, encoding="utf-8").read()
    tag = '<script id="data" type="application/json">'
    i = html.index(tag) + len(tag)
    j = html.index("</script>", i)
    return json.loads(html[i:j])


def main():
    today = datetime.date.today().strftime("%Y%m%d")
    out_path = os.path.join(OUT_DIR, f"AMT-Master_lemma-report_{today}.md")

    master = json.load(open(MASTER_PATH, encoding="utf-8"))["AMT-Master"]
    vdata = load_viewer_data(VIEWER_PATH)
    ved = vdata["ed"]
    vmaster = {c["id"]: c for c in vdata["master"]}

    m_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(MASTER_PATH))
    v_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(VIEWER_PATH))

    # ------------------------------------------------------------- sync check
    sync = []
    sync.append(("concept count JSON vs viewer",
                 len(master) == len(vdata["master"]),
                 f"{len(master)} vs {len(vdata['master'])}"))
    ids_json = {c["id"] for c in master}
    ids_view = set(vmaster)
    sync.append(("master `id` sets identical", ids_json == ids_view,
                 f"{len(ids_json & ids_view)} shared, "
                 f"{len(ids_json ^ ids_view)} differing"))
    flag_mismatch = 0
    for c in master:
        vq = set(vmaster.get(c["id"], {}).get("_q", []))
        if set(scan_q(c)) != vq:
            flag_mismatch += 1
        unl = {s for s in SOURCES
               if cellstr(c, f"{s}-et") not in PLACE and not c.get(f"{s}-id")}
        vu = set(vmaster.get(c["id"], {}).get("_u", []))
        if unl != vu:
            flag_mismatch += 1
    sync.append(("recomputed ???/unlinked flags match embedded `_q`/`_u`",
                 flag_mismatch == 0, f"{flag_mismatch} mismatches"))
    # forward link targets must exist among viewer edition ids
    ed_ids = {s: {e["id"] for e in ved[s]} for s in SOURCES}
    dangling_fwd = [(c["Amt-Master-ID"], s, eid)
                    for c in master for s in SOURCES
                    for eid in (c.get(f"{s}-id") or []) if eid not in ed_ids[s]]
    sync.append(("all forward `<Source>-id` targets exist in editions",
                 not dangling_fwd, f"{len(dangling_fwd)} dangling"))
    # symmetry master->edition vs edition->master
    fwd_pairs = {(c["id"], eid) for c in master for s in SOURCES
                 for eid in (c.get(f"{s}-id") or [])}
    back_pairs = {(mid, e["id"]) for s in SOURCES for e in ved[s]
                  for mid in (e.get("master-id") or [])}
    sync.append(("forward/back link pairs symmetric",
                 fwd_pairs == back_pairs,
                 f"{len(fwd_pairs)} forward vs {len(back_pairs)} back, "
                 f"{len(fwd_pairs ^ back_pairs)} asymmetric (§6.7–6.8)"))
    fwd_only = fwd_pairs - back_pairs
    back_only = back_pairs - fwd_pairs
    midx = {c["id"]: c for c in master}
    edidx = {e["id"]: (s, e) for s in SOURCES for e in ved[s]}

    # ---------------------------------------------------------- master basics
    amid_dupes = [k for k, n in collections.Counter(
        c["Amt-Master-ID"] for c in master).items() if n > 1]
    id_dupes = [k for k, n in collections.Counter(
        c["id"] for c in master).items() if n > 1]
    bad_prefix = [c["Amt-Master-ID"] for c in master
                  if not str(c.get("id", "")).startswith("am-")]

    field_fill = {k: sum(1 for c in master if cellstr(c, k) not in PLACE)
                  for k in MASTER_FIELDS}

    cat_c = collections.Counter(cellstr(c, "Amt-Cat") or "—" for c in master)
    sugu_c = collections.Counter(cellstr(c, "Sugu") or "—" for c in master)
    teema_c = collections.Counter(cellstr(c, "Teema") or "—" for c in master)
    sem_c = collections.Counter(cellstr(c, "Sem-Cat") or "—" for c in master)

    # ------------------------------------------------- per-source attestation
    per_src = {}
    unlinked_cells = []          # (source, concept, et, de)
    q_cells = []                 # (source, concept, field, value)
    asym_cells = []              # et placeholder but de real
    for s in SOURCES:
        st = dict(att=0, lnk=0, unl=0, dash=0, qqq=0, null=0,
                  de_att=0, refs=0, multi=0, maxids=0)
        for c in master:
            et, de = cellstr(c, f"{s}-et"), cellstr(c, f"{s}-de")
            ids = c.get(f"{s}-id") or []
            st["refs"] += len(ids)
            if len(ids) > 1:
                st["multi"] += 1
                st["maxids"] = max(st["maxids"], len(ids))
            if de not in PLACE:
                st["de_att"] += 1
            if et not in PLACE:
                st["att"] += 1
                if ids:
                    st["lnk"] += 1
                else:
                    st["unl"] += 1
                    unlinked_cells.append((s, c, et, de))
            elif et == "---":
                st["dash"] += 1
            elif et == "???":
                st["qqq"] += 1
            else:
                st["null"] += 1
            if et in PLACE and de not in PLACE:
                asym_cells.append((s, c, et, de))
            for k, v in ((f"{s}-et", et), (f"{s}-de", de)):
                if v != "???" and "???" in v:
                    q_cells.append((s, c, k, v))
        per_src[s] = st

    # --------------------------------------------------------------- coverage
    def n_att(c):
        return sum(1 for s in SOURCES if cellstr(c, f"{s}-et") not in PLACE)

    def n_lnk(c):
        return sum(1 for s in SOURCES if c.get(f"{s}-id"))

    cov_att = collections.Counter(n_att(c) for c in master)
    cov_lnk = collections.Counter(n_lnk(c) for c in master)
    all6 = sorted(c["Amt-Master-ID"] for c in master if n_att(c) == 6)
    zero = sorted((c["Amt-Master-ID"], cellstr(c, "Amt-Cat"))
                  for c in master if n_att(c) == 0)

    xsc_mismatch = []
    for c in master:
        raw = c.get("Cross-source count", "NULL")   # str or int (D8 2026-07-13)
        try:
            stored = int(str(raw))
        except ValueError:
            stored = None
        comp = n_att(c)
        if stored is None or stored != comp:
            xsc_mismatch.append((c["Amt-Master-ID"], raw, comp))

    # ------------------------------------------------------- edition coverage
    ed_stats = {}
    for s in SOURCES:
        entries = ved[s]
        linked = [e for e in entries if e.get("master-id")]
        mwu_total = sum(len(e.get("mwu") or []) for e in entries)
        mwu_linked = sum(len(e.get("mwu") or []) for e in linked)
        multi_master = sum(1 for e in linked if len(e["master-id"]) > 1)
        ed_stats[s] = dict(total=len(entries), linked=len(linked),
                           mwu=mwu_total, mwu_l=mwu_linked, mm=multi_master)

    distinct_targets = {s: len({eid for c in master
                                for eid in (c.get(f"{s}-id") or [])})
                        for s in SOURCES}

    # ------------------------------------------------------------------ flags
    flagged_m = [(c, scan_q(c),
                  [s for s in SOURCES if cellstr(c, f"{s}-et") not in PLACE
                   and not c.get(f"{s}-id")])
                 for c in master]
    flagged_m = [(c, q, u) for c, q, u in flagged_m if q or u]
    flagged_e = [(s, e) for s in SOURCES for e in ved[s] if e.get("_q")]

    missing_def = [(c["Amt-Master-ID"],
                    cellstr(c, "DEF_en") in PLACE,
                    cellstr(c, "DEF_et") in PLACE)
                   for c in master
                   if cellstr(c, "DEF_en") in PLACE
                   or cellstr(c, "DEF_et") in PLACE]

    # =================================================================== emit
    L = []
    A = L.append
    tot_att = sum(per_src[s]["att"] for s in SOURCES)
    tot_lnk = sum(per_src[s]["lnk"] for s in SOURCES)
    tot_unl = sum(per_src[s]["unl"] for s in SOURCES)

    A(f"# AMT-Master lemma & linkage review — {datetime.date.today().isoformat()}")
    A("")
    A("Cross-check of the annotated Master concept table against the dataset "
      "embedded in LexLex, the canonical viewer.")
    A("")
    A("| input | path | modified | size |")
    A("|---|---|---|---|")
    A(f"| Master JSON | `{MASTER_PATH}` | {m_mtime:%Y-%m-%d %H:%M} | "
      f"{os.path.getsize(MASTER_PATH)/1024:.0f} kB |")
    A(f"| LexLex viewer | `{VIEWER_PATH}` | {v_mtime:%Y-%m-%d %H:%M} | "
      f"{os.path.getsize(VIEWER_PATH)/1024/1024:.1f} MB |")
    A("")
    A("**Conventions.** A source cell counts as *attested* when `<Source>-et` "
      "is none of the placeholders `---` / `???` / `NULL` / empty (placeholders "
      "are not attestations). *Linked* = non-empty `<Source>-id` list "
      "(UUIDv7 links into the harmonized edition JSONs). Linking is "
      "precision-first: unresolved cells stay unlinked rather than being "
      "force-matched.")
    A("")

    A("## 1 · JSON ↔ viewer sync")
    A("")
    A("| check | result | detail |")
    A("|---|---|---|")
    for name, ok, detail in sync:
        A(f"| {name} | {'✅' if ok else '❌'} | {detail} |")
    A("")

    A("## 2 · Master table overview")
    A("")
    A(f"- **{len(master)}** concepts (lemmas) in the Master.")
    A(f"- `Amt-Master-ID` unique: {'yes' if not amid_dupes else f'NO — dupes: {amid_dupes}'}; "
      f"`id` unique: {'yes' if not id_dupes else f'NO — dupes: {id_dupes}'}; "
      f"non-`am-` ids: {len(bad_prefix)}.")
    A(f"- Attested source-cells: **{tot_att}** of {6*len(master)} "
      f"({pct(tot_att, 6*len(master))}); linked **{tot_lnk}** "
      f"({pct(tot_lnk, tot_att)} of attested); attested-but-unlinked "
      f"**{tot_unl}** (§6.1).")
    A("")
    A("### 2.1 Field completeness")
    A("")
    A(f"| field | filled | % of {len(master)} |")
    A("|---|---:|---:|")
    for k in MASTER_FIELDS:
        A(f"| {k} | {field_fill[k]} | {pct(field_fill[k], len(master))} |")
    A("")

    A("### 2.2 Category distributions")
    A("")
    A("**Amt-Cat** (K1 core professions · K2 occasional/role-like doings · "
      "K3 societal roles/status):")
    A("")
    A("| Amt-Cat | concepts | % |")
    A("|---|---:|---:|")
    for k, n in cat_c.most_common():
        A(f"| {md_esc(k)} | {n} | {pct(n, len(master))} |")
    A("")
    A("**Sugu (gender):**")
    A("")
    A("| Sugu | concepts |")
    A("|---|---:|")
    for k, n in sugu_c.most_common():
        A(f"| {md_esc(k)} | {n} |")
    A("")
    A(f"**Teema** ({len(teema_c)} distinct values):")
    A("")
    A("| Teema | concepts |")
    A("|---|---:|")
    for k, n in teema_c.most_common():
        A(f"| {md_esc(k)} | {n} |")
    A("")
    A(f"**Sem-Cat** ({len(sem_c)} distinct values, top 30):")
    A("")
    A("| Sem-Cat | concepts |")
    A("|---|---:|")
    for k, n in sem_c.most_common(30):
        A(f"| {md_esc(k)} | {n} |")
    A("")

    A("## 3 · Attestation & linkage per source")
    A("")
    A("| source | attested | linked | linked % | attested-unlinked | `---` | `???` | NULL/empty | de-cell attested | id refs | multi-id cells | max ids |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in SOURCES:
        t = per_src[s]
        A(f"| {s} | {t['att']} | {t['lnk']} | {pct(t['lnk'], t['att'])} | "
          f"{t['unl']} | {t['dash']} | {t['qqq']} | {t['null']} | "
          f"{t['de_att']} | {t['refs']} | {t['multi']} | {t['maxids']} |")
    A(f"| **total** | **{tot_att}** | **{tot_lnk}** | "
      f"**{pct(tot_lnk, tot_att)}** | **{tot_unl}** | "
      f"{sum(per_src[s]['dash'] for s in SOURCES)} | "
      f"{sum(per_src[s]['qqq'] for s in SOURCES)} | "
      f"{sum(per_src[s]['null'] for s in SOURCES)} | "
      f"{sum(per_src[s]['de_att'] for s in SOURCES)} | "
      f"{sum(per_src[s]['refs'] for s in SOURCES)} | "
      f"{sum(per_src[s]['multi'] for s in SOURCES)} | |")
    A("")
    A("A *multi-id cell* holds several comma/semicolon-separated forms or a "
      "headword the edition repeats; one Master cell may therefore link to "
      "several edition entries.")
    A("")

    A("## 4 · Cross-source coverage")
    A("")
    A("| attested in N sources | concepts | linked in N sources | concepts |")
    A("|---:|---:|---:|---:|")
    for n in range(7):
        A(f"| {n} | {cov_att.get(n, 0)} | {n} | {cov_lnk.get(n, 0)} |")
    A("")
    A(f"### 4.1 Concepts attested in all six sources ({len(all6)})")
    A("")
    A(", ".join(f"`{md_esc(x)}`" for x in all6) or "—")
    A("")
    A(f"### 4.2 Concepts attested nowhere ({len(zero)})")
    A("")
    if zero:
        A("These carry analytic annotation but no dictionary evidence in the "
          "six editions:")
        A("")
        A("| Amt-Master-ID | Amt-Cat |")
        A("|---|---|")
        for amid, cat in zero:
            A(f"| {md_esc(amid)} | {md_esc(cat)} |")
    else:
        A("None — every concept is attested in at least one source.")
    A("")
    A("### 4.3 `Cross-source count` field vs recomputed attestation")
    A("")
    if xsc_mismatch:
        A(f"{len(xsc_mismatch)} mismatches (stored ≠ recomputed):")
        A("")
        A("| Amt-Master-ID | stored | recomputed |")
        A("|---|---:|---:|")
        for amid, raw, comp in xsc_mismatch:
            A(f"| {md_esc(amid)} | {md_esc(raw)} | {comp} |")
    else:
        A(f"All {len(master)} stored values match the recomputed per-concept "
          "attestation count. ✅")
    A("")

    A("## 5 · Edition-side view (lemmas reflected in the Master)")
    A("")
    A("| edition | entries | linked to Master | % of edition | distinct entries targeted by Master | multi-concept entries | MWUs (total) | MWUs in linked entries |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in SOURCES:
        d = ed_stats[s]
        A(f"| {s} | {d['total']} | {d['linked']} | "
          f"{pct(d['linked'], d['total'])} | {distinct_targets[s]} | "
          f"{d['mm']} | {d['mwu']} | {d['mwu_l']} |")
    A(f"| **total** | **{sum(ed_stats[s]['total'] for s in SOURCES)}** | "
      f"**{sum(ed_stats[s]['linked'] for s in SOURCES)}** | | "
      f"**{sum(distinct_targets[s] for s in SOURCES)}** | "
      f"{sum(ed_stats[s]['mm'] for s in SOURCES)} | "
      f"{sum(ed_stats[s]['mwu'] for s in SOURCES)} | "
      f"{sum(ed_stats[s]['mwu_l'] for s in SOURCES)} |")
    A("")
    A("*Linked to Master* counts edition entries whose `master-id` back-link "
      "list is non-empty; *multi-concept entries* belong to more than one "
      "Master concept. The Master is a thematic (person/role) selection, so "
      "low percentages are expected for full dictionaries.")
    A("")

    A("## 6 · Omissions & attention items")
    A("")
    A(f"### 6.1 Attested but unlinked cells ({len(unlinked_cells)}) — the open review tail")
    A("")
    if unlinked_cells:
        A("| source | Amt-Master-ID | et form | de |")
        A("|---|---|---|---|")
        for s, c, et, de in unlinked_cells:
            A(f"| {SHORT[s]} {s.split('-')[1]} | {md_esc(c['Amt-Master-ID'])} | "
              f"{md_esc(et)} | {md_esc(de if de not in PLACE else '')} |")
    else:
        A("None. ✅")
    A("")
    A(f"### 6.2 Cells containing `???` inside a value ({len(q_cells)})")
    A("")
    if q_cells:
        A("| source | Amt-Master-ID | field | value |")
        A("|---|---|---|---|")
        for s, c, k, v in q_cells:
            A(f"| {SHORT[s]} | {md_esc(c['Amt-Master-ID'])} | {k} | {md_esc(v)} |")
    else:
        A("None. ✅")
    A("")
    A(f"### 6.3 Flagged Master concepts ({len(flagged_m)})")
    A("")
    if flagged_m:
        A("| Amt-Master-ID | ??? in fields | attested-but-unlinked in |")
        A("|---|---|---|")
        for c, q, u in flagged_m:
            A(f"| {md_esc(c['Amt-Master-ID'])} | {md_esc(', '.join(q)) or ''} | "
              f"{', '.join(SHORT[s] for s in u)} |")
    else:
        A("None. ✅")
    A("")
    A(f"### 6.4 Flagged edition entries ({len(flagged_e)})")
    A("")
    if flagged_e:
        A("| edition | headword | fields with ??? |")
        A("|---|---|---|")
        for s, e in flagged_e:
            A(f"| {s} | {md_esc(e.get('headword-et', '—'))} | "
              f"{md_esc(', '.join(e['_q']))} |")
    else:
        A("None. ✅")
    A("")
    A(f"### 6.5 Concepts missing a definition ({len(missing_def)})")
    A("")
    if missing_def:
        A("| Amt-Master-ID | DEF_en missing | DEF_et missing |")
        A("|---|:---:|:---:|")
        for amid, en, et in missing_def:
            A(f"| {md_esc(amid)} | {'•' if en else ''} | {'•' if et else ''} |")
    else:
        A("None — every concept has both definitions. ✅")
    A("")
    A(f"### 6.6 Asymmetric cells — et is a placeholder but de has content ({len(asym_cells)})")
    A("")
    if asym_cells:
        A("| source | Amt-Master-ID | et | de |")
        A("|---|---|---|---|")
        for s, c, et, de in asym_cells:
            A(f"| {SHORT[s]} | {md_esc(c['Amt-Master-ID'])} | {md_esc(et)} | "
              f"{md_esc(de)} |")
    else:
        A("None. ✅")
    A("")

    A(f"### 6.7 Dangling forward links ({len(dangling_fwd)})")
    A("")
    A("`<Source>-id` values that do not exist in the edition the column "
      "belongs to. If the id resolves in *another* edition, it was placed in "
      "the wrong column:")
    A("")
    if dangling_fwd:
        A("| column | Amt-Master-ID | id | actually resolves to |")
        A("|---|---|---|---|")
        for amid, s, eid in dangling_fwd:
            hit = edidx.get(eid)
            where = (f"{hit[0]}: {md_esc(hit[1].get('headword-et', '—'))} "
                     f"“{md_esc(hit[1].get('equiv-de', ''))}”"
                     if hit else "nothing — id unknown in all editions")
            A(f"| {s} | {md_esc(amid)} | `{eid}` | {where} |")
    else:
        A("None. ✅")
    A("")
    A(f"### 6.8 Asymmetric link pairs — forward without back-link ({len(fwd_only)}) / back without forward ({len(back_only)})")
    A("")
    A("The Master's `<Source>-id` and the editions' `master-id` should mirror "
      "each other; hand-added links on one side need the reverse-link pass "
      "re-run to restore symmetry.")
    A("")
    if fwd_only:
        A("**Forward-only** (in Master, missing in edition `master-id`):")
        A("")
        A("| Amt-Master-ID | edition entry | headword |")
        A("|---|---|---|")
        for mid, eid in sorted(fwd_only,
                               key=lambda p: midx[p[0]]["Amt-Master-ID"]):
            s, e = edidx.get(eid, ("?", None))
            hw = e.get("headword-et", "—") if e else "**id missing**"
            A(f"| {md_esc(midx[mid]['Amt-Master-ID'])} | "
              f"{SHORT.get(s, s)} `{eid}` | {md_esc(hw)} |")
        A("")
    if back_only:
        A("**Back-only** (edition claims a Master concept that does not link "
          "back):")
        A("")
        A("| edition entry | headword | claims Amt-Master-ID |")
        A("|---|---|---|")
        for mid, eid in sorted(back_only):
            s, e = edidx.get(eid, ("?", None))
            hw = e.get("headword-et", "—") if e else "?"
            amid = midx[mid]["Amt-Master-ID"] if mid in midx else f"`{mid}` (unknown)"
            A(f"| {SHORT.get(s, s)} `{eid}` | {md_esc(hw)} | {md_esc(amid)} |")
        A("")
    if not fwd_only and not back_only:
        A("None — fully symmetric. ✅")
        A("")

    A("## 7 · Full lemma inventory")
    A("")
    A("Per source: **L** linked · **A** attested but unlinked · **?** cell is "
      "`???` · — not attested. Sources in chronological order "
      "St 1637 · Gu 1648 · Gö 1660 · Ve 17XX · He 1732 · Hu 1780.")
    A("")
    A("| Amt-Master-ID | Amt-Cat | Sugu | St | Gu | Gö | Ve | He | Hu | n att |")
    A("|---|---|---|---|---|---|---|---|---|---:|")
    for c in master:
        marks = []
        for s in SOURCES:
            et = cellstr(c, f"{s}-et")
            if et == "???":
                marks.append("?")
            elif et in PLACE:
                marks.append("—")
            elif c.get(f"{s}-id"):
                marks.append("L")
            else:
                marks.append("**A**")
        A(f"| {md_esc(c['Amt-Master-ID'])} | {md_esc(cellstr(c, 'Amt-Cat'))} | "
          f"{md_esc(cellstr(c, 'Sugu'))} | " + " | ".join(marks) +
          f" | {n_att(c)} |")
    A("")
    A("---")
    A(f"*Generated by `scripts/report_master_review.py` on "
      f"{datetime.datetime.now():%Y-%m-%d %H:%M}.*")
    A("")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"wrote {out_path} ({os.path.getsize(out_path)/1024:.0f} kB, "
          f"{len(L)} lines)")
    print(f"  sync: {sum(1 for _, ok, _ in sync if ok)}/{len(sync)} checks OK")
    print(f"  attested {tot_att}, linked {tot_lnk}, unlinked tail {tot_unl}")
    print(f"  flagged concepts {len(flagged_m)}, flagged edition entries "
          f"{len(flagged_e)}, xsc mismatches {len(xsc_mismatch)}")


if __name__ == "__main__":
    main()

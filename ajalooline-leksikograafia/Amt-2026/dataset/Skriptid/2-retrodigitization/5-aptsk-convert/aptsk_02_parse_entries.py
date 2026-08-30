# Created: 2026-07-08 16-18-21
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
"""
Step 2/3 — parse styled lines into structured dictionary entries.

Reads build/spans.jsonl (produced by aptsk_01_extract_spans.py), segments the
line stream into dictionary articles and parses each article into the JSON
schema documented in PLAN.md. Nothing is dropped: every printed line lands
verbatim in exactly one entry's "raw_lines", and any text the structural
parser cannot place is kept under "unparsed" plus a warning.

Output: json/entries_<LETTER>.json (one per section), json/all_entries.json,
        json/manifest.json

Run:  uv run python APTSK-scripts/aptsk_02_parse_entries.py
"""

import json
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "build" / "spans.jsonl"
JSON_DIR = ROOT / "json"

SOURCES = ["GtVT", "GtUT", "Ml", "Rs", "St", "Bl", "WT", "Vr", "Mn", "PR"]
# 'Gt' (unspecified Gutslaff ms.) and 'RS' (misprint for Rs) occur a few times
SRC_RE = re.compile(r"(GtVT|GtUT|GUT|Gt|RS|Ml|Rs|St|Bl|WT|Vr|Mn|PR)")
SRC_NORMALIZE = {"RS": "Rs", "GUT": "GtUT"}

SOURCE_LEGEND = {
    "Ml": "Georg Mülleri jutluste piiblitsitaadid (Jutluseraamat, Ilmamaa 2007)",
    "Rs": "Joachim Rossihniuse perikoobid (1632; toim. V. Reimann 1898)",
    "St": "Heinrich Stahl, Hand- und Hauszbuch, Dritter Theil (Tallinn 1638)",
    "GtVT": "Johannes Gutslaffi Vana Testamendi tõlkekäsikiri (EKM EKLA, f 192, m 326:1)",
    "GtUT": "Johannes Gutslaffi Uue Testamendi tõlkekäsikiri (RA f 1187, n 2, s 5323)",
    "Bl": "Christoph Blume Uue Testamendi tõlkekatke käsikiri (RA f 1187, n 2, s 5323)",
    "WT": "Wastne Testament (Riia 1686)",
    "Vr": "Andreas ja Adrian Virginiuse Vana Testamendi tõlkekäsikiri (EKM EKLA, f 192, m 358:1)",
    "Mn": "Müncheni käsikiri, Johann Hornungi UT tõlke ümberkirjutus (Codex livo-esthnicus)",
    "PR": "Piibli Ramat (esimene eesti täispiibel, Tallinn 1739)",
}

# HL (bold italic) appears in headwords too: foreign-language components are
# set in bold italics (jähvke, hofmeister, schilt ...)
BOLD_ROLES = {"HW", "SUP", "HL"}


# ----------------------------------------------------------------------------
# character/role stream assembly
# ----------------------------------------------------------------------------

def line_chars(spans):
    """One line's spans -> parallel char/role lists; COMB inherits neighbour role."""
    chars, roles = [], []
    for role, text in spans:
        for ch in text:
            chars.append(ch)
            roles.append(role)
    for i, r in enumerate(roles):           # combining marks join the run they sit in
        if r == "COMB":
            roles[i] = roles[i - 1] if i else (roles[i + 1] if i + 1 < len(roles) else "IT")
    return chars, roles


def build_stream(lines):
    """
    Join an entry's lines into one char/role stream.
    Dehyphenation: soft hyphens (U+00AD) are always removed; a hard '-' at a
    line end is removed when the next line starts lower-case (heuristic;
    raw_lines keep the original). line_no[i] = source line of stream char i.
    """
    chars, roles, line_no = [], [], []
    softs = 0
    prev_soft_end = False
    for li, spans in enumerate(lines):
        lchars, lroles = line_chars(spans)
        soft_end = "".join(lchars).rstrip().endswith("­")
        # strip soft hyphens
        keep = [(c, r) for c, r in zip(lchars, lroles) if c != "­"]
        softs += len(lchars) - len(keep)
        if not keep:
            prev_soft_end = prev_soft_end or soft_end
            continue
        if chars:
            nxt = keep[0][0]
            if prev_soft_end:
                pass          # discretionary break: tight join; a remaining
                              # '-' before it was a real compound hyphen
            elif chars[-1] == "-" and nxt.islower():
                chars.pop(); roles.pop(); line_no.pop()      # hyphenation artefact
            elif chars[-1] == "-":
                pass                                          # real hyphen: tight join
            elif chars[-1] != " " and nxt != " ":
                chars.append(" "); roles.append(roles[-1]); line_no.append(li)
        for c, r in keep:
            chars.append(c); roles.append(r); line_no.append(li)
        prev_soft_end = soft_end
    return "".join(chars), roles, line_no, softs


# ----------------------------------------------------------------------------
# entry parsing
# ----------------------------------------------------------------------------

def split_word_homonym(text, roles):
    """'Aabel1' with SUP-role digit -> ('Aabel', 1); keeps other digits intact."""
    word, hom = [], []
    for ch, r in zip(text, roles):
        (hom if r == "SUP" else word).append(ch)
    homonym = int("".join(hom)) if hom else None
    return "".join(word).strip(), homonym


def parse_words_list(text, roles):
    """Comma-separated word list -> [{'form', 'homonym', 'uncertain',
    'original_spelling'}]; the ?/* markers are stripped into flags."""
    items, buf_c, buf_r = [], [], []
    for ch, r in zip(text, roles):
        if ch == ",":
            if "".join(buf_c).strip():
                items.append((buf_c[:], buf_r[:]))
            buf_c, buf_r = [], []
        else:
            buf_c.append(ch); buf_r.append(r)
    if "".join(buf_c).strip():
        items.append((buf_c, buf_r))
    out = []
    for cs, rs in items:
        w, h = split_word_homonym("".join(cs), rs)
        uncertain, orig = w.startswith("?"), w.endswith("*")
        out.append({"form": w.lstrip("?").rstrip("*"), "homonym": h,
                    "uncertain": uncertain, "original_spelling": orig,
                    "foreign": any(r == "HL" for r in rs)})
    return out


# totals and counts may be printed with thousands groups: '10 230'
COUNT_RE = r"(\d{1,3}(?: \d{3})+|\d+)"


class EntryParser:
    def __init__(self, text, roles, line_no, entry):
        self.t, self.r, self.ln = text, roles, line_no
        self.n = len(text)
        self.pos = 0
        self.e = entry

    # -- helpers -------------------------------------------------------------
    def warn(self, msg):
        self.e["warnings"].append(msg)

    def skip_ws(self):
        while self.pos < self.n and self.t[self.pos].isspace():
            self.pos += 1

    def match_source(self, at):
        """Return (source, end) if a REG-role source label starts at `at`."""
        m = SRC_RE.match(self.t, at)
        if not m:
            return None
        if any(self.r[i] not in ("REG",) for i in range(m.start(), m.end())):
            return None
        end = m.end()
        # reject a label glued to more roman letters ('Mln…'); italic letters
        # right after are a missing-space misprint ('GtVTninck …') — accept
        if end < self.n and self.t[end].isalpha() and self.r[end] == "REG":
            return None
        return SRC_NORMALIZE.get(m.group(1), m.group(1)), end

    # -- header --------------------------------------------------------------
    def parse_header(self):
        e = self.e
        self.skip_ws()
        start = self.pos
        # bold zone = headword list; punctuation/markers between bold words are
        # sometimes set in roman type, so admit ' ,*?~' regardless of role.
        # Stop at '<' (gloss) or at a roman-type digit (the total count).
        while self.pos < self.n:
            ch, r = self.t[self.pos], self.r[self.pos]
            if ch == "<":                    # gloss opens (its '<' is bold once)
                break
            if r in BOLD_ROLES:
                self.pos += 1
            elif ch.isdigit():
                break                        # total count reached
            elif ch in " ,*?~":
                self.pos += 1
            else:
                break
        bold_t = self.t[start:self.pos]
        bold_r = self.r[start:self.pos]

        vt = re.search(r"(?:^|\s)vt\s", bold_t)
        if vt:
            left_t, left_r = bold_t[:vt.start()], bold_r[:vt.start()]
            right_t, right_r = bold_t[vt.end():], bold_r[vt.end():]
            e["reference_only"] = True
            heads = parse_words_list(left_t, left_r)
            e["headwords"] = [h["form"] for h in heads]
            e["headword"] = e["headwords"][0] if e["headwords"] else ""
            e["homonym"] = heads[0]["homonym"] if heads else None
            e["headword_details"] = heads
            e["see"] = parse_words_list(right_t, right_r)
            self.skip_ws()
            return

        heads = parse_words_list(bold_t, bold_r)
        if not heads:
            self.warn("no headword parsed")
            e["headword"], e["headwords"], e["homonym"] = "", [], None
        else:
            e["headwords"] = [h["form"] for h in heads]
            e["headword"] = heads[0]["form"]
            e["homonym"] = heads[0]["homonym"]
        e["headword_details"] = heads

        self.skip_ws()
        self.try_gloss()
        self.skip_ws()
        # total frequency (may use thousands groups: '10 230'; tolerate a
        # stray '-' misprint before it)
        m = re.match(r"[-–]?\s*" + COUNT_RE, self.t[self.pos:])
        if m:
            e["total_count"] = int(m.group(1).replace(" ", ""))
            self.pos += m.end()
            self.skip_ws()
            if e["gloss"] is None and self.pos < self.n and (
                    self.t[self.pos] == "<" or self.r[self.pos] in ("G", "GI")):
                self.warn("gloss printed after total count")
                self.try_gloss()
            return
        # gloss misprinted fully in 9 pt roman without '<': 'urge auk> 1'
        mg = re.match(r"([^<>]{1,60})>", self.t[self.pos:])
        if e["gloss"] is None and mg:
            self.warn("gloss without opening '<'")
            e["gloss"] = mg.group(1).strip()
            e["gloss_rich"] = [{"text": e["gloss"], "italic": False}]
            self.pos += mg.end()
            self.skip_ws()
            m = re.match(COUNT_RE, self.t[self.pos:])
            if m:
                e["total_count"] = int(m.group(1).replace(" ", ""))
                self.pos += m.end()
                return
        self.warn("no total count found after headword")

    def try_gloss(self):
        e = self.e
        # optional gloss <...>; tolerate a missing '<' (gloss recognisable by
        # its 8 pt type) and a missing '>' (gloss ends where 8 pt type ends)
        has_open = self.pos < self.n and self.t[self.pos] == "<"
        has_gloss_type = self.pos < self.n and self.r[self.pos] in ("G", "GI")
        if has_open or has_gloss_type:
            gstart = self.pos + 1 if has_open else self.pos
            close = self.t.find(">", self.pos)
            if close == -1:
                self.warn("unclosed gloss")
                close = gstart
                while close < self.n and (self.r[close] in ("G", "GI")
                                          or self.t[close].isspace()):
                    close += 1
                if close == gstart:
                    # gloss misprinted in 9 pt roman: end it at the total
                    # count, i.e. the first roman number closing its line
                    for m9 in re.finditer(COUNT_RE, self.t[gstart:]):
                        j0, j1 = gstart + m9.start(), gstart + m9.end()
                        if (all(self.r[i] == "REG" for i in range(j0, j1))
                                and (j1 >= self.n
                                     or self.line_of(self.next_nonspace(j1))
                                     > self.line_of(j1 - 1))):
                            close = j0
                            break
                    else:
                        close = self.n
                while close > gstart and self.t[close - 1].isspace():
                    close -= 1
            if not has_open:
                self.warn("gloss without opening '<'")
            gtxt = self.t[gstart:close]
            groles = self.r[gstart:close]
            e["gloss"] = gtxt.strip()
            rich, cur, cur_it = [], [], None
            for ch, r in zip(gtxt, groles):
                it = r in ("GI", "IT", "HL")
                if cur_it is None or it == cur_it:
                    cur.append(ch); cur_it = it
                else:
                    rich.append({"text": "".join(cur), "italic": cur_it})
                    cur, cur_it = [ch], it
            if cur:
                rich.append({"text": "".join(cur), "italic": bool(cur_it)})
            e["gloss_rich"] = rich
            self.pos = close + 1 if (close < self.n and self.t[close] == ">") else close

    # -- per-source statistics -----------------------------------------------
    def line_of(self, i):
        return self.ln[min(i, self.n - 1)] if self.n else 0

    def next_nonspace(self, i):
        while i < self.n and self.t[i].isspace():
            i += 1
        return i

    def count_boundary_ok(self, digits_end):
        """A statistics count must be followed by , . ; a line break, another
        source label, 'vt ka' or the end — else the digit belongs to a verse
        reference in an example ('… mehs; 2Sm23:29)')."""
        k = self.next_nonspace(digits_end)
        if k >= self.n or self.t[k] in ",.;":
            return True
        if self.match_source(k) or (self.t.startswith("vt ka", k)
                                    and self.r[k] == "REG"):
            return True
        return self.line_of(k) > self.line_of(digits_end - 1)

    def parse_source_counts(self):
        e = self.e
        while True:
            self.skip_ws()
            save = self.pos
            got = self.match_source(self.pos)
            if not got:
                return
            src, p = got
            content_end = p        # end of label/last form, for line checks
            # optional italic subform, possibly '~'-separated variants
            # ('Ml eb ~ is ~ emme ~ evat 175'); the '~' may be roman type
            chunks, fstart = [], None
            while True:
                while p < self.n and self.t[p].isspace():
                    p += 1
                if fstart is None:
                    fstart = p
                q = p
                # the ?/* markers around a form may be roman type ('?möth'),
                # as may the leading '-' of clitic subforms ('Ml -kit 2')
                while q < self.n and (self.t[q] in "?*-" or (
                        self.r[q] in ("IT", "HL") and self.t[q] != ",")):
                    q += 1
                if q == p or not self.t[p:q].strip("?*- "):
                    break
                chunks.append(self.t[p:q].strip())
                p = q
                content_end = q
                r = self.next_nonspace(p)
                if chunks[-1].rstrip().endswith("~"):
                    p = r                       # variant list continues
                    continue
                if r < self.n and self.t[r] == "~":
                    p = r + 1
                    continue
                break
            roman_form = False
            if not chunks:
                # misprints set the subform in roman instead of italic
                # ('GtVT külv 2, …'): accept a lowercase roman word before a
                # count, flagged with a warning
                m3 = re.match(r"\s*([?]?[a-zõäöüšž][a-zõäöüšž*.-]*"
                              r"(?:\s*~\s*[?]?[a-zõäöüšž][a-zõäöüšž*.-]*)*)"
                              r"[\s*?]+(?=\d)", self.t[p:])
                if m3 and all(self.r[p + i] == "REG"
                              for i in range(m3.start(1), m3.end(1))):
                    chunks.append(m3.group(1))
                    fstart = p + m3.start(1)
                    content_end = fstart + len(m3.group(1))
                    p += m3.end()
                    roman_form = True
            form_raw = self.t[fstart:p].strip() or None if chunks else None
            forms = []
            uncertain = orig = False
            for c in chunks:
                for v in (x.strip() for x in c.split("~")):
                    if not v:
                        continue
                    uncertain |= v.startswith("?")
                    orig |= "*" in v
                    forms.append(v.lstrip("?").rstrip("*").strip())

            def emit(count):
                e["source_counts"].append({
                    "source": src,
                    "form": " ~ ".join(forms) if forms else None,
                    "forms": forms, "form_raw": form_raw,
                    "count": count,
                    "uncertain": uncertain, "original_spelling": orig,
                })

            # optional markers spilling into roman type, then the count
            m = re.match(r"[\s*?]*" + COUNT_RE, self.t[p:])
            if m and self.count_boundary_ok(p + m.end()):
                if "*" in self.t[p:p + m.start(1)]:
                    orig = True
                if "?" in self.t[p:p + m.start(1)]:
                    uncertain = True
                if roman_form:
                    self.warn(f"statistics subform in roman type ({src})")
                emit(int(m.group(1).replace(" ", "")))
                p += m.end()
                k = self.next_nonspace(p)
                if k < self.n and self.t[k] in ",.;":
                    p = k + 1
                self.pos = p
                continue

            # count-less list: numerals and pronouns are printed as a plain
            # source list ('kaheksakümmend | Rs, St, GtVT, …' / 'Ml nee ~ need')
            countless_ok = (
                all(sc["count"] is None for sc in e["source_counts"])
                and not e["examples"]
                and (not form_raw or (len(form_raw) <= 35
                                      and not any(c in form_raw for c in ".(")
                                      and not form_raw[0].isupper())))
            if countless_ok:
                k = self.next_nonspace(p)
                if k < self.n and self.t[k] == ",":
                    emit(None)
                    self.pos = k + 1
                    continue
                if k >= self.n or self.line_of(k) > self.line_of(
                        max(content_end - 1, 0)):
                    emit(None)                  # last item ends its line
                    self.pos = p
                    return
                if not chunks and self.match_source(k):
                    emit(None)                  # 'Ml Rs penk 1' misprint
                    self.warn(f"count missing for source {src}")
                    self.pos = k
                    continue
            self.pos = save                     # not a statistics item -> examples
            return

    # -- examples + vt ka ----------------------------------------------------
    def at_vt_ka(self):
        return (self.t.startswith("vt ka", self.pos)
                and self.r[self.pos] == "REG")

    def parse_examples(self):
        e = self.e
        while True:
            self.skip_ws()
            if self.pos >= self.n or self.at_vt_ka():
                return
            got = self.match_source(self.pos)
            if got:
                src, p = got
                self.pos = p
            elif len(self.t) - self.pos >= 3:
                src = None                    # label omitted in print ('tore')
                self.warn("example without source label")
            else:
                rest = self.t[self.pos:].strip()
                if rest:
                    e["unparsed"] = rest
                    self.warn(f"unparsed trailing text ({len(rest)} chars)")
                self.pos = self.n
                return
            text_parts, refs, hls = [], [], []
            hl_cur = []
            closed = False
            # very rare PDF typo: reference group missing its '(' — detect a
            # roman-type verse reference that closes with ')'
            bare_ref = re.compile(
                r"([1-5]?[A-ZÕÄÖÜŠŽ][a-zõäöüšž]{0,3}\d+(?::\d+[ab]?)?"
                r"(?:,\s*[^)]{1,40})?(?:\s(?:pk\s)?algus)?)\)")
            while self.pos < self.n and not closed:
                ch, role = self.t[self.pos], self.r[self.pos]
                if (role == "REG" and ch != "(" and (ch.isdigit() or ch.isupper())
                        and (m := bare_ref.match(self.t, self.pos))
                        and all(r == "REG" for r in self.r[m.start():m.end()])):
                    refs.extend(x.strip() for x in m.group(1).split(",") if x.strip())
                    self.pos = m.end()
                    q = self.pos
                    while q < self.n and self.t[q] in " ;":
                        q += 1
                    if (q >= self.n or self.match_source(q)
                            or (self.t.startswith("vt ka", q) and self.r[q] == "REG")):
                        self.pos = q
                        closed = True
                    else:
                        text_parts.append(f"({m.group(1)})")
                    continue
                if ch == "(" and role in ("REG", "HW", "G"):
                    # ref parens are roman type; rarely misprinted as bold
                    close = self.t.find(")", self.pos)
                    if close == -1:
                        text_parts.append(ch); self.pos += 1; continue
                    group = self.t[self.pos + 1:close]
                    refs.extend(x.strip() for x in group.split(",") if x.strip())
                    self.pos = close + 1
                    # after a reference group: '; SRC' | 'vt ka' | end -> close
                    q = self.pos
                    while q < self.n and self.t[q] in " ;":
                        q += 1
                    if (q >= self.n or self.match_source(q)
                            or (self.t.startswith("vt ka", q) and self.r[q] == "REG")):
                        self.pos = q
                        closed = True
                    else:
                        text_parts.append(f"({group})")   # mid-example reference
                    continue
                if role == "HL":
                    hl_cur.append(ch)
                elif hl_cur:
                    hls.append("".join(hl_cur).strip()); hl_cur = []
                text_parts.append(ch)
                self.pos += 1
            if hl_cur:
                hls.append("".join(hl_cur).strip())
            text = re.sub(r"\s+", " ", "".join(text_parts)).strip()
            if not closed:
                # salvage a reference whose ')' is missing in print
                m2 = re.search(r"\(\s*([1-5]?[A-ZÕÄÖÜŠŽ][^()]{0,40})$", text)
                if m2:
                    refs.extend(x.strip() for x in m2.group(1).split(",")
                                if x.strip())
                    text = text[:m2.start()].rstrip()
                    self.warn(f"reference missing closing paren ({src})")
            e["examples"].append({
                "source": src, "text": text,
                "highlighted": [h for h in hls if h],
                "references": refs,
            })
            if not refs:
                self.warn(f"example without reference ({src})")

    def parse_vt_ka(self):
        e = self.e
        self.skip_ws()
        if not self.at_vt_ka():
            return
        raw = self.t[self.pos:].strip()
        body_t = self.t[self.pos + 5:]
        body_r = self.r[self.pos + 5:]
        parts = body_t.split(";")
        offs, groups = 0, []
        for part in parts:
            pr = body_r[offs:offs + len(part)]
            groups.append(parse_words_list(part, pr))
            offs += len(part) + 1
        fmt = lambda ws: [w["form"] + (str(w["homonym"]) if w["homonym"] else "")
                          for w in ws]
        e["see_also"] = {
            "raw": raw,
            "same_form": fmt(groups[0]) if groups else [],
            "other_form": fmt(groups[1]) if len(groups) > 1 else [],
        }
        self.pos = self.n

    def run(self):
        e = self.e
        self.parse_header()
        if e["reference_only"]:
            self.skip_ws()
            if self.pos < self.n:
                rest = self.t[self.pos:].strip()
                if rest:
                    e["unparsed"] = rest
                    self.warn("text after reference entry")
            return
        self.parse_source_counts()
        self.parse_examples()
        self.parse_vt_ka()
        # numerals/pronouns: the dictionary deliberately prints a plain source
        # list without any frequencies — not a defect
        if (e["total_count"] is None and e["source_counts"]
                and all(sc["count"] is None for sc in e["source_counts"])):
            e["counts_omitted"] = True
            e["warnings"] = [w for w in e["warnings"]
                             if w != "no total count found after headword"]


# ----------------------------------------------------------------------------
# segmentation + driver
# ----------------------------------------------------------------------------

def first_role(spans):
    for role, text in spans:
        if text.strip():
            return role
    return None


def starts_entry(spans):
    """Does this line open a new article? Bold (HW) or — for foreign-material
    headwords — bold-italic (HL) start; must begin with a letter (a misprinted
    bold '(' before a verse reference must not split an entry)."""
    fr = first_role(spans)
    lead = next((t.lstrip()[:1] for _, t in spans if t.strip()), "")
    if fr == "HW":
        # letters, clitic particles ('-gi/-ki') and one misprint where the
        # first headword landed in the running head (', verav vt värav')
        return lead.isalpha() or lead in "-,"
    if fr == "HL":
        # HL also starts wrapped example lines; a header line has no example
        # text (IT) and shows a gloss, a count or a 'vt' reference
        if any(r == "IT" and t.strip() for r, t in spans) or not lead.isalpha():
            return False
        txt = "".join(t for _, t in spans)
        return ("<" in txt or txt.rstrip()[-1:].isdigit()
                or re.search(r"\bvt\b", txt) is not None)
    return False


def last_bold_open(spans):
    """True if a line is an unfinished bold header (wrapped headword list).
    Not for 'vt' lines: their target may legitimately end in '-'
    ('see- vt selle-')."""
    tail = None
    for role, text in spans:
        if text.strip():
            tail = (role, text)
    if not tail:
        return False
    if re.search(r"\bvt\b", "".join(t for _, t in spans)):
        return False
    role, text = tail
    return role in ("HW", "HL") and text.rstrip().endswith((",", "-"))


def slugify(text):
    s = unicodedata.normalize("NFKD", text.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "entry"


def new_entry(letter):
    return {
        "id": None, "letter": letter,
        "headword": None, "headwords": [], "homonym": None,
        "gloss": None, "gloss_rich": None,
        "reference_only": False, "see": None,
        "total_count": None, "counts_omitted": False, "source_counts": [],
        "examples": [], "see_also": None,
        "pages": [], "raw_lines": [], "warnings": [],
    }


def main():
    letter = None
    groups = []          # (letter, [line-record, ...])
    cur = None
    prev_spans = None

    with IN_PATH.open(encoding="utf-8") as f:
        for raw in f:
            rec = json.loads(raw)
            if "letter" in rec:
                letter = rec["letter"]
                cur, prev_spans = None, None
                continue
            spans = [tuple(s) for s in rec["s"]]
            starts = starts_entry(spans) and not (
                prev_spans is not None and last_bold_open(prev_spans))
            if starts or cur is None:
                cur = {"letter": letter, "lines": [], "pages": []}
                groups.append(cur)
            cur["lines"].append(spans)
            cur["pages"].append(rec["p"])
            prev_spans = spans

    entries = []
    ids = Counter()
    total_soft = 0
    for g in groups:
        e = new_entry(g["letter"])
        e["pages"] = sorted(set(g["pages"]))
        e["raw_lines"] = ["".join(t for _, t in spans) for spans in g["lines"]]
        text, roles, line_no, softs = build_stream(g["lines"])
        total_soft += softs
        EntryParser(text, roles, line_no, e).run()
        base = slugify(e["headword"] or e["raw_lines"][0][:20])
        if e["homonym"]:
            base += f"-{e['homonym']}"
        ids[base] += 1
        e["id"] = base if ids[base] == 1 else f"{base}--{ids[base]}"
        entries.append(e)

    JSON_DIR.mkdir(exist_ok=True)
    by_letter = {}
    for e in entries:
        by_letter.setdefault(e["letter"], []).append(e)
    for letter, es in by_letter.items():
        path = JSON_DIR / f"entries_{letter}.json"
        path.write_text(json.dumps(
            {"letter": letter, "entry_count": len(es), "entries": es},
            ensure_ascii=False, indent=1), encoding="utf-8")

    (JSON_DIR / "all_entries.json").write_text(
        json.dumps({"entry_count": len(entries), "entries": entries},
                   ensure_ascii=False, indent=1), encoding="utf-8")

    warn_entries = [e for e in entries if e["warnings"]]
    manifest = {
        "title": "Eesti vanema piiblitõlke sõnastik 1600–1739",
        "publisher": "Eesti Keele Instituut / EKSA 2025",
        "editors": ["Inge Käsi", "Maeve Leivo", "Ahti Lohk", "Anu Pedaja-Ansen",
                    "Heiki Reila", "Kristiina Ross", "Annika Viht"],
        "source_pdf": "APTSK_ALL.pdf",
        "dictionary_pages": [21, 926],
        "extracted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "extraction_scripts": ["aptsk_01_extract_spans.py",
                               "aptsk_02_parse_entries.py",
                               "aptsk_03_validate.py"],
        "author_of_extraction": "Madis Jürviste, Claude Fable 5",
        "text_source_abbreviations": SOURCE_LEGEND,
        "subform_markers": {
            "?": "lemma uncertain — only inflected forms attested",
            "*": "subform given in original orthography (spelling untrustworthy)",
        },
        "letters": {l: len(es) for l, es in by_letter.items()},
        "entry_count": len(entries),
        "reference_entry_count": sum(e["reference_only"] for e in entries),
        "entries_with_warnings": len(warn_entries),
        "soft_hyphens_removed": total_soft,
        "files": sorted(f"entries_{l}.json" for l in by_letter) + ["all_entries.json"],
        "schema_notes": "See APTSK-scripts/PLAN.md — field-by-field schema; "
                        "'raw_lines' preserves every printed line verbatim.",
    }
    (JSON_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"entries          : {len(entries)}")
    print(f"reference entries: {manifest['reference_entry_count']}")
    print(f"with warnings    : {len(warn_entries)}")
    for w in warn_entries[:10]:
        print("  !", w["id"], w["warnings"][:2])
    print(f"letters          : { {l: len(es) for l, es in by_letter.items()} }")
    print(f"written to {JSON_DIR}/")


if __name__ == "__main__":
    main()

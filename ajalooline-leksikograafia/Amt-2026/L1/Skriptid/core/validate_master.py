# Created: 2026-07-13 11-58-38
# Author: Madis Jürviste
# Co-Authored-By: Claude Fable 5
"""Standing READ-ONLY validator for AMT-Master_annotated.json (DQ task T12).

Checks (see DQ-review 2026-07-12, items K2 + A-block):
  schema              identical 30-key schema in identical order on every record
  master-id-unique    Amt-Master-ID unique (and no leading/trailing whitespace)
  id-unique-prefix    `id` unique, `am-` uuid7 shape; edition -id prefixes match
  csc                 Cross-source count == number of sources with a real -et
  placeholder-id      `-et` placeholder (--- / ??? / NULL) implies `-id` == []
  hygiene-strip       no leading/trailing whitespace on any string value
  hygiene-nbsp        no U+00A0 anywhere
  hygiene-nfc         every string value is NFC-normalized
  hygiene-dspace      no double spaces inside values
  multiplicity        only `[xN]` multiplicity markers ([N] / [Nx] are wrong)
  comment-order       Comment-2 filled => Comment-1 filled; same for 3 vs 2
  amt-cat             Amt-Cat within the six sanctioned values
  collation           record order follows Estonian collation of Amt-Master-ID
  dup-tokens          no duplicate `;`-tokens inside one -et/-de field
                      (-et compared case-insensitively per decision D3)

Never writes anything. Exit code 0 iff zero ERRORs (warnings do not fail).

Usage:
    uv run python scripts/validate_master.py [path/to/master.json]
"""
import json
import os
import re
import sys
import unicodedata
from collections import Counter

# ---------------------------------------------------------------------------
# Severity config: flip a check to "warning" while its DQ decision is parked
# (Section 2/3 of DQ-tasklist_AMT-Master_2026-07-12.md). As of 2026-07-13 all
# relevant decisions (D2, D3, D8, ...) are taken, so everything is an error.
# ---------------------------------------------------------------------------
SEVERITY = {
    "schema":           "error",
    "master-id-unique": "error",
    "id-unique-prefix": "error",
    "csc":              "error",
    "placeholder-id":   "error",
    "hygiene-strip":    "error",
    "hygiene-nbsp":     "error",
    "hygiene-nfc":      "error",
    "hygiene-dspace":   "error",
    "multiplicity":     "error",
    "comment-order":    "error",
    "amt-cat":          "error",
    "collation":        "error",
    "dup-tokens":       "error",
}

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "Katus-ALUSANDMED", "json-all",
                            "AMT-Master_annotated.json")

SOURCES = ["Stahl-1637", "Gutslaff-1648", "Göseken-1660",
           "Vestring-17XX", "Helle-1732", "Hupel-1780-est-ger"]

# Same placeholder set as recompute_crosssource.py: not a real attestation.
PLACE = {"---", "???", "NULL", "", " "}

# Canonical 30-key schema in canonical order.
CANONICAL_KEYS = [
    "Amt-Master-ID", "id", "Amt-Cat", "Sem-Cat", "Teema", "Sugu",
    "DEF_en", "DEF_et",
    "Stahl-1637-et", "Stahl-1637-de", "Stahl-1637-id",
    "Gutslaff-1648-et", "Gutslaff-1648-de", "Gutslaff-1648-id",
    "Göseken-1660-et", "Göseken-1660-de", "Göseken-1660-id",
    "Vestring-17XX-et", "Vestring-17XX-de", "Vestring-17XX-id",
    "Helle-1732-et", "Helle-1732-de", "Helle-1732-id",
    "Hupel-1780-est-ger-et", "Hupel-1780-est-ger-de", "Hupel-1780-est-ger-id",
    "Cross-source count", "Comment-1", "Comment-2", "Comment-3",
]

# Per-source id prefixes (katus_lib.PREFIX).
ID_PREFIX = {
    "Stahl-1637": "st", "Gutslaff-1648": "gu", "Göseken-1660": "go",
    "Vestring-17XX": "ve", "Helle-1732": "he", "Hupel-1780-est-ger": "hu",
}
UUID_RE = re.compile(r"^[a-z]{2}-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}"
                     r"-[0-9a-f]{4}-[0-9a-f]{12}$")

AMT_CAT_ALLOWED = {"K1", "K2", "K3", "K1, K2", "K1, K3", "K2, K3"}

# Bad multiplicity markers: [3] or [2x]; the sanctioned form is [x3] / [x2].
BAD_MULT_RE = re.compile(r"\[\d+x?\]")

# ---------------------------------------------------------------------------
# Estonian collation — copied verbatim from convert_amt_master.py so the
# validator agrees with the script that produced the record order.
# ---------------------------------------------------------------------------
ET_ALPHABET = "abcdefghijklmnopqrsšzžtuvwõäöüxy"
PRE_LETTER = {" ": 0, ",": 1, "-": 2}
LETTER_RANK = {ch: i + 10 for i, ch in enumerate(ET_ALPHABET)}


def collation_key(s):
    """Estonian-collation sort key for a (casefolded) string."""
    key = []
    for ch in s.lower():
        if ch in PRE_LETTER:
            key.append(PRE_LETTER[ch])
        elif ch in LETTER_RANK:
            key.append(LETTER_RANK[ch])
        else:
            key.append(1000 + ord(ch))
    return key


# ---------------------------------------------------------------------------
# Violation collection
# ---------------------------------------------------------------------------
VIOLATIONS = {name: [] for name in SEVERITY}
MAX_PRINT = 25          # detail lines printed per check


def hit(check, msg):
    VIOLATIONS[check].append(msg)


def iter_strings(rec):
    """Yield (field-label, string value) for every string in the record,
    including the items of the -id list fields."""
    for k, v in rec.items():
        if isinstance(v, str):
            yield k, v
        elif isinstance(v, list):
            for j, item in enumerate(v):
                if isinstance(item, str):
                    yield f"{k}[{j}]", item


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_schema(recs):
    for i, r in enumerate(recs):
        if list(r) != CANONICAL_KEYS:
            rid = r.get("Amt-Master-ID", f"#(index {i})")
            missing = [k for k in CANONICAL_KEYS if k not in r]
            extra = [k for k in r if k not in CANONICAL_KEYS]
            what = (f"missing={missing} extra={extra}" if missing or extra
                    else "key order differs")
            hit("schema", f"{rid!r}: {what}")


def check_master_id_unique(recs):
    counts = Counter(r.get("Amt-Master-ID") for r in recs)
    for mid, n in counts.items():
        if n > 1:
            hit("master-id-unique", f"{mid!r} occurs {n} times")
    for r in recs:
        mid = r.get("Amt-Master-ID", "")
        if isinstance(mid, str) and mid != mid.strip():
            hit("master-id-unique", f"{mid!r}: leading/trailing whitespace in key")


def check_ids(recs):
    counts = Counter(r.get("id") for r in recs)
    for rid, n in counts.items():
        if n > 1:
            hit("id-unique-prefix", f"id {rid!r} occurs {n} times")
    for r in recs:
        mid = r.get("Amt-Master-ID", "?")
        rid = r.get("id", "")
        if not (isinstance(rid, str) and rid.startswith("am-")
                and UUID_RE.match(rid)):
            hit("id-unique-prefix", f"{mid!r}: malformed id {rid!r}")
        for src, pref in ID_PREFIX.items():
            ids = r.get(f"{src}-id", [])
            if not isinstance(ids, list):
                hit("id-unique-prefix", f"{mid!r}: {src}-id is not a list")
                continue
            for eid in ids:
                if not (isinstance(eid, str) and eid.startswith(pref + "-")
                        and UUID_RE.match(eid)):
                    hit("id-unique-prefix",
                        f"{mid!r}: {src}-id contains {eid!r} "
                        f"(expected prefix {pref}-)")


def check_csc(recs):
    """Rule from recompute_crosssource.py: a source counts iff <Source>-et is
    a string outside PLACE. Value accepted as str or int (D8 pending/applied)."""
    for r in recs:
        mid = r.get("Amt-Master-ID", "?")
        n = sum(1 for s in SOURCES
                if isinstance(r.get(f"{s}-et"), str)
                and r.get(f"{s}-et") not in PLACE)
        raw = r.get("Cross-source count")
        try:
            val = int(str(raw))
        except (TypeError, ValueError):
            hit("csc", f"{mid!r}: non-numeric Cross-source count {raw!r}")
            continue
        if val != n:
            hit("csc", f"{mid!r}: Cross-source count {raw!r} != recomputed {n}")


def check_placeholder_id(recs):
    for r in recs:
        mid = r.get("Amt-Master-ID", "?")
        for s in SOURCES:
            et = r.get(f"{s}-et")
            ids = r.get(f"{s}-id")
            if isinstance(et, str) and et in PLACE and ids:
                hit("placeholder-id",
                    f"{mid!r}: {s}-et is placeholder {et!r} but {s}-id "
                    f"has {len(ids)} link(s)")


def check_hygiene(recs):
    for r in recs:
        mid = r.get("Amt-Master-ID", "?")
        for label, v in iter_strings(r):
            if v != v.strip():
                hit("hygiene-strip", f"{mid!r} {label}: {v!r}")
            if " " in v:
                hit("hygiene-nbsp", f"{mid!r} {label}: {v!r}")
            if unicodedata.normalize("NFC", v) != v:
                hit("hygiene-nfc", f"{mid!r} {label}: {v!r}")
            if "  " in v:
                hit("hygiene-dspace", f"{mid!r} {label}: {v!r}")


def check_multiplicity(recs):
    for r in recs:
        mid = r.get("Amt-Master-ID", "?")
        for label, v in iter_strings(r):
            for m in BAD_MULT_RE.findall(v):
                hit("multiplicity", f"{mid!r} {label}: bad marker {m!r} in {v!r}")


def check_comment_order(recs):
    for r in recs:
        mid = r.get("Amt-Master-ID", "?")
        c1, c2, c3 = (r.get("Comment-1"), r.get("Comment-2"), r.get("Comment-3"))
        if c2 != "NULL" and c1 == "NULL":
            hit("comment-order", f"{mid!r}: Comment-2 filled while Comment-1 NULL")
        if c3 != "NULL" and c2 == "NULL":
            hit("comment-order", f"{mid!r}: Comment-3 filled while Comment-2 NULL")


def check_amt_cat(recs):
    for r in recs:
        mid = r.get("Amt-Master-ID", "?")
        cat = r.get("Amt-Cat")
        if cat not in AMT_CAT_ALLOWED:
            hit("amt-cat", f"{mid!r}: Amt-Cat {cat!r}")


def check_collation(recs):
    prev_key, prev_mid = None, None
    for r in recs:
        mid = r.get("Amt-Master-ID", "")
        key = collation_key(mid if isinstance(mid, str) else "")
        if prev_key is not None and key < prev_key:
            hit("collation", f"{mid!r} sorts before its predecessor {prev_mid!r}")
        prev_key, prev_mid = key, mid


def check_dup_tokens(recs):
    """No duplicate `;`-tokens within one edition -et/-de field. Estonian
    word-form (-et) fields compare case-insensitively (decision D3: case-only
    variants are duplicates); -de fields compare exactly."""
    for r in recs:
        mid = r.get("Amt-Master-ID", "?")
        for s in SOURCES:
            for suffix in ("-et", "-de"):
                field = f"{s}{suffix}"
                v = r.get(field)
                if not isinstance(v, str) or v in PLACE or ";" not in v:
                    continue
                tokens = [t.strip() for t in v.split(";") if t.strip()]
                norm = [t.casefold() if suffix == "-et" else t for t in tokens]
                dupes = [t for t, n in Counter(norm).items() if n > 1]
                if dupes:
                    hit("dup-tokens",
                        f"{mid!r} {field}: duplicate token(s) {dupes} in {v!r}")


# ---------------------------------------------------------------------------
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    with open(path, encoding="utf-8") as f:          # read-only, never writes
        data = json.load(f)
    if list(data) != ["AMT-Master"]:
        print(f"FATAL: unexpected root key(s) {list(data)}")
        sys.exit(2)
    recs = data["AMT-Master"]
    print(f"Validating {path}  ({len(recs)} records)\n")

    check_schema(recs)
    check_master_id_unique(recs)
    check_ids(recs)
    check_csc(recs)
    check_placeholder_id(recs)
    check_hygiene(recs)
    check_multiplicity(recs)
    check_comment_order(recs)
    check_amt_cat(recs)
    check_collation(recs)
    check_dup_tokens(recs)

    n_err = n_warn = 0
    for check in SEVERITY:
        msgs = VIOLATIONS[check]
        sev = SEVERITY[check]
        tag = "OK  " if not msgs else ("ERR " if sev == "error" else "WARN")
        print(f"[{tag}] {check:18} {len(msgs)} violation(s)")
        for m in msgs[:MAX_PRINT]:
            print(f"         {m}")
        if len(msgs) > MAX_PRINT:
            print(f"         ... and {len(msgs) - MAX_PRINT} more")
        if msgs:
            if sev == "error":
                n_err += len(msgs)
            else:
                n_warn += len(msgs)

    print(f"\nTotal: {n_err} error(s), {n_warn} warning(s)")
    sys.exit(0 if n_err == 0 else 1)


if __name__ == "__main__":
    main()

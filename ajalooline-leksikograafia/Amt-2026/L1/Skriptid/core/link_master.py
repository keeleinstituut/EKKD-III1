# Created: 2026-07-07 13-19-11
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
# Co-Authored-By: Claude Fable 5
"""Link AMT-Master source cells to edition entry ids.

For every Master row and every source, the Estonian form recorded in
`<Source>-et` is matched against that edition's entries (headword-et, and any
mwu-et). The matched edition `id`s are written into a new `<Source>-id` field
placed right after `<Source>-de`.

Matching is tiered for transparency / confidence:
  t1  exact (after light normalisation: drop (), {}, [], ~, punctuation)
  t2  + collapse spaces/hyphens
  t3  + degemination (collapse doubled letters)
  t4  + historical-orthography fold (w->v, ck->kk, sz->ss)
t1-t3 (and unambiguous t4) are written automatically; ambiguous matches,
t4 matches, and unmatched attested cells are emitted to a review report.

Run with --write to modify the Master; default is a dry-run report.
"""
import difflib
import json
import re
import sys

JD = "Katus-ALUSANDMED/json-all"
MASTER = f"{JD}/AMT-Master_annotated.json"
REPORT = "scripts/linkage_report.json"

# Master source-column prefix  ->  edition file / source tag
SOURCES = [
    "Stahl-1637", "Gutslaff-1648", "Göseken-1660",
    "Vestring-17XX", "Helle-1732", "Hupel-1780-est-ger",
]
PLACE = {"---", "???", "null", "nan", "", " "}


def norm(s):
    s = s.strip().lower()
    s = re.sub(r"\([^)]*\)", "", s)          # (r.) (d.) dialect/usage notes
    s = s.replace("{", "").replace("}", "")  # Göseken reconstruction braces
    s = re.sub(r"\[x\d+\]", "", s)           # [x2] occurrence-count notes
    s = re.sub(r"\[[^\]]*\]", " ", s) if re.search(r"\[[^\]]*\b(vt|sub|lk)\b", s) else s
    s = s.replace("[", "").replace("]", "")  # keep bracketed content otherwise
    s = s.replace("~", "")
    s = re.sub(r"\bx\d+\b", "", s)
    s = s.strip().strip("\"'.,;:!?")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def collapse(s):
    return re.sub(r"[\s\-]", "", s)


def degem(s):
    return re.sub(r"(.)\1+", r"\1", s)


def fold(s):
    return s.replace("ck", "kk").replace("w", "v").replace("sz", "ss")


# tier name -> key function applied to a normalised form
TIERS = [
    ("t1", lambda s: s),
    ("t2", lambda s: collapse(s)),
    ("t3", lambda s: degem(collapse(s))),
    ("t4", lambda s: fold(collapse(s))),
]


def split_forms(cell):
    return [p for p in re.split(r"[;,/]| od\.| oder ", cell)]


def load(tag):
    with open(f"{JD}/{tag}.json", encoding="utf-8") as f:
        return json.load(f)[tag]


def build_indexes(tag):
    """One dict per tier: tier-key -> set(entry ids)."""
    idx = {t: {} for t, _ in TIERS}
    for e in load(tag):
        forms = [("hw", e["headword-et"])] + [("mwu", m["mwu-et"]) for m in e["mwu"]]
        for _kind, k in forms:
            if not k or k.strip().lower() in PLACE:
                continue
            nk = norm(k)
            if not nk:
                continue
            for tname, fn in TIERS:
                idx[tname].setdefault(fn(nk), set()).add(e["id"])
    return idx


def match_form(form_norm, idx):
    """Return (tier, set_of_ids) for the first tier that matches, else (None,set())."""
    for tname, fn in TIERS:
        key = fn(form_norm)
        if key in idx[tname]:
            return tname, set(idx[tname][key])
    return None, set()


def main(write):
    with open(MASTER, encoding="utf-8") as f:
        master = json.load(f)
    entries = master["AMT-Master"]
    indexes = {tag: build_indexes(tag) for tag in SOURCES}

    report = {tag: {"matched": 0, "by_tier": {"t1": 0, "t2": 0, "t3": 0, "t4": 0},
                    "ambiguous": [], "t4_matches": [], "unmatched": [],
                    "attested": 0} for tag in SOURCES}

    for row in entries:
        for tag in SOURCES:
            et = row.get(f"{tag}-et", "NULL")
            id_field = f"{tag}-id"
            if et.strip().lower() in PLACE:
                row[id_field] = []
                continue
            subs = [norm(s) for s in split_forms(et)]
            subs = [s for s in subs if s and s not in PLACE]
            if not subs:
                row[id_field] = []
                continue
            rep = report[tag]
            rep["attested"] += 1
            cell_ids, tiers_used, ambiguous, has_match = [], set(), False, False
            for s in subs:
                tier, ids = match_form(s, indexes[tag])
                if not ids:
                    continue
                has_match = True
                tiers_used.add(tier)
                if len(ids) > 1:
                    ambiguous = True
                # write t1-t3 always; t4 only if unambiguous
                if tier in ("t1", "t2", "t3") or len(ids) == 1:
                    cell_ids.extend(sorted(ids))
            cell_ids = sorted(set(cell_ids))
            row[id_field] = cell_ids
            if has_match:
                rep["matched"] += 1
                worst = max(tiers_used, key=lambda t: ["t1", "t2", "t3", "t4"].index(t))
                rep["by_tier"][worst] += 1
                if ambiguous:
                    rep["ambiguous"].append({"et": et, "ids": cell_ids})
                if "t4" in tiers_used:
                    rep["t4_matches"].append({"et": et, "ids": cell_ids})
            else:
                # fuzzy suggestion for the review queue (NOT auto-linked)
                t1 = indexes[tag]["t1"]
                cand = difflib.get_close_matches(subs[0], list(t1), n=1, cutoff=0.75)
                sugg = None
                if cand:
                    sugg = {"edition_form": cand[0],
                            "ids": sorted(t1[cand[0]]),
                            "score": round(difflib.SequenceMatcher(
                                None, subs[0], cand[0]).ratio(), 3)}
                rep["unmatched"].append({"et": et, "suggestion": sugg})

    # console summary
    print(f"{'source':22} {'attest':>6} {'match':>6} {'unmat':>6}  "
          f"{'t1':>4}{'t2':>4}{'t3':>4}{'t4':>4}  {'ambig':>5}")
    for tag in SOURCES:
        r = report[tag]
        bt = r["by_tier"]
        print(f"{tag:22} {r['attested']:6} {r['matched']:6} "
              f"{len(r['unmatched']):6}  {bt['t1']:4}{bt['t2']:4}{bt['t3']:4}"
              f"{bt['t4']:4}  {len(r['ambiguous']):5}")

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nreport -> {REPORT}")

    if write:
        # reorder keys so <Source>-id sits right after <Source>-de
        reordered = []
        for row in entries:
            new = {}
            for k, v in row.items():
                if k.endswith("-id") and k[:-3] in SOURCES:
                    continue  # skip; inserted after the matching -de
                new[k] = v
                if k.endswith("-de"):
                    base = k[:-3]
                    if base in SOURCES:
                        new[f"{base}-id"] = row[f"{base}-id"]
            reordered.append(new)
        master["AMT-Master"] = reordered
        with open(MASTER, "w", encoding="utf-8") as f:
            json.dump(master, f, ensure_ascii=False, indent=2)
        print(f"WROTE links into {MASTER}")
    else:
        print("(dry-run; pass --write to modify the Master)")


if __name__ == "__main__":
    main("--write" in sys.argv)

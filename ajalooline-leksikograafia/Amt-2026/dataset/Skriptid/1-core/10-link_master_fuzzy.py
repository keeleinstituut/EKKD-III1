# Created: 2026-07-07 13-19-11
# Author: Madis Jürviste
# Co-Authored-By: Claude Opus 4.8
# Co-Authored-By: Claude Fable 5
"""Second linkage pass: recover the unmatched tail with German-gloss confirmation.

For each Master cell still unlinked (`<Source>-id` empty but `<Source>-et`
attested), find fuzzy Estonian candidates in the edition, but accept one ONLY
if the German glosses also correspond (`<Source>-de` vs the candidate's
`equiv-de`). This rejects the dangerous look-alikes that pure string similarity
catches: agent-noun vs verb (armatseja/armatsema), and different words that
differ by one letter (Naljakas/paljakas, aia mees/laiwa mees).

Run dry by default; `--write` appends the confirmed ids into the Master.
"""
import difflib
import json
import re
import sys

import importlib
lm = importlib.import_module("09-link_master")  # reuse norm(), split_forms(), SOURCES, PLACE, load()

JD = lm.JD
MASTER = lm.MASTER
FUZZY_REPORT = "scripts/linkage_fuzzy_report.json"

EE_CUTOFF = 0.84
ARTICLES = {"der", "die", "das", "den", "dem", "des", "ein", "eine", "einen",
            "einem", "einer", "eines", "zu", "zur", "zum"}
# generic head-nouns of person/people compounds: a lone shared one is NOT
# enough to confirm a lexeme (naise mees vs naiseta mees both gloss "...Kerl").
GENERIC_HEADS = {"leute", "kerl", "mann", "manns", "weib", "weibs", "mensch",
                 "frau", "person", "ding", "sache", "art"}


def norm_de(s):
    """Normalise a German gloss to a set of content tokens + a flat string."""
    if not s or s.strip().lower() in lm.PLACE:
        return set(), ""
    s = s.lower().replace("ß", "ss")
    s = re.sub(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}", " ", s)   # drop note groups
    s = re.sub(r"[^a-zäöüé\s]", " ", s)
    toks = [t for t in s.split() if t and t not in ARTICLES]
    return set(toks), " ".join(toks)


def german_ok(master_de, edition_de, ee_score):
    """Confirm two glosses describe the same lexeme (high precision)."""
    mt, ms = norm_de(master_de)
    et, es = norm_de(edition_de)
    if not mt or not et:
        return False                      # can't confirm -> don't link
    if ms == es:
        return True
    # substantial overlap of content tokens
    shared = mt & et
    union = mt | et
    if union and len(shared) / len(union) >= 0.5:
        return True
    # shorter full gloss contained in the longer one
    short, long = sorted((ms, es), key=len)
    if len(short) >= 5 and short in long:
        return True
    # near-identical Estonian AND a shared *distinctive* (non-generic) word
    distinctive = {t for t in shared if len(t) >= 5 and t not in GENERIC_HEADS}
    if ee_score >= 0.90 and distinctive:
        return True
    return False


def main(write):
    with open(MASTER, encoding="utf-8") as f:
        master = json.load(f)
    entries = master["AMT-Master"]

    # per-source: id->entry, and list of (norm_headword/mwu, id, equiv-de)
    ed = {}
    for tag in lm.SOURCES:
        items, keys = [], []
        by_id = {}
        for e in lm.load(tag):
            by_id[e["id"]] = e
            forms = [e["headword-et"]] + [m["mwu-et"] for m in e["mwu"]]
            for k in forms:
                if k and k.strip().lower() not in lm.PLACE:
                    nk = lm.norm(k)
                    if nk:
                        items.append((nk, e["id"], e["equiv-de"]))
                        keys.append(nk)
        ed[tag] = {"items": items, "keys": keys, "by_id": by_id}

    accepted = {tag: [] for tag in lm.SOURCES}
    for row in entries:
        for tag in lm.SOURCES:
            if row.get(f"{tag}-id"):
                continue                                   # already linked
            et = row.get(f"{tag}-et", "NULL")
            if et.strip().lower() in lm.PLACE:
                continue
            subs = [lm.norm(s) for s in lm.split_forms(et)]
            subs = [s for s in subs if s and s not in lm.PLACE]
            if not subs:
                continue
            mde = row.get(f"{tag}-de", "NULL")
            found = []
            for s in subs:
                for cand in difflib.get_close_matches(
                        s, ed[tag]["keys"], n=6, cutoff=EE_CUTOFF):
                    score = round(difflib.SequenceMatcher(None, s, cand).ratio(), 3)
                    for nk, eid, ede in ed[tag]["items"]:
                        if nk == cand and german_ok(mde, ede, score):
                            found.append((eid, cand, ede, score))
            if found:
                ids = sorted({f[0] for f in found})
                row[f"{tag}-id"] = ids
                best = max(found, key=lambda x: x[3])
                accepted[tag].append({
                    "master_et": et, "master_de": mde,
                    "edition_et": best[1], "edition_de": best[2],
                    "score": best[3], "ids": ids})

    print(f"{'source':22} fuzzy-linked (German-confirmed)")
    total = 0
    for tag in lm.SOURCES:
        print(f"  {tag:22} {len(accepted[tag])}")
        total += len(accepted[tag])
    print(f"  {'TOTAL':22} {total}")
    print("\nAccepted pairs (ee_master -> ee_edition | de_master ~ de_edition):")
    for tag in lm.SOURCES:
        for a in accepted[tag]:
            print(f"  [{tag[:5]}] {a['master_et'][:26]:26} -> {a['edition_et'][:22]:22}"
                  f" | {a['master_de'][:24]:24} ~ {a['edition_de'][:24]}")

    with open(FUZZY_REPORT, "w", encoding="utf-8") as f:
        json.dump(accepted, f, ensure_ascii=False, indent=2)

    if write:
        with open(MASTER, "w", encoding="utf-8") as f:
            json.dump(master, f, ensure_ascii=False, indent=2)
        print(f"\nWROTE {total} confirmed fuzzy links into {MASTER}")
    else:
        print("\n(dry-run; pass --write to modify the Master)")


if __name__ == "__main__":
    main("--write" in sys.argv)

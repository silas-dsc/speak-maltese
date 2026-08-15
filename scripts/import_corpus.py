#!/usr/bin/env python3
"""Import Maltese sentence pairs from an openly-licensed corpus, levelled by CEFR.

Source: **Tatoeba** (`mlt`↔`eng`), CC BY 2.0 FR. Chosen over the alternatives on
three grounds that all matter for a learner:

* **Register.** Tatoeba is everyday sentences. Bible translations and the National
  Library's digitised books are liturgical and 19th-century respectively — excellent
  Maltese, wrong Maltese to learn *speaking* from, and they would teach forms no one
  says today.
* **Licence.** Tatoeba is CC BY, so sentences can be redistributed with attribution.
  The 1984 Maltese Bible is under active copyright (Malta Bible Society), and NLA
  holdings are a mix with no blanket reuse grant.
* **Alignment.** Sentences arrive already paired with English, so a card has both
  sides without machine translation — the failure that made the "2000 most common
  Maltese words" list unusable.

Levelling is by *your* deck, not by an abstract scale: a sentence is A1 if it is
short and every word is already tier-1/2 vocabulary, and drifts upward as it gets
longer and leans on words you have not met. That is a proxy for CEFR, not a
substitute — it is deliberately conservative and the output is quarantined for
review rather than taught directly.

    python scripts/import_corpus.py                 # fetch, level, write TSV
    python scripts/import_corpus.py --max-level A2  # only the easy end
    python scripts/import_corpus.py --report        # what it would add, no write
"""

from __future__ import annotations

import argparse
import bz2
import csv
import io
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import curriculum, text  # noqa: E402
from backend.config import DATA_DIR  # noqa: E402

BASE = "https://downloads.tatoeba.org/exports/per_language"
OUT = DATA_DIR / "corpus_phrases.tsv"
CACHE = DATA_DIR / ".corpus_cache"

ATTRIBUTION = ("Tatoeba (tatoeba.org), CC BY 2.0 FR — sentence ids retained "
               "for attribution")

LEVELS = ["A1", "A2", "B1", "B2", "C1"]


def _fetch(url: str, name: str) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / name
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    print(f"  ↓ {url}")
    import httpx  # already a dependency, and carries its own CA bundle

    r = httpx.get(url, timeout=300, follow_redirects=True,
                  headers={"user-agent": "speak-maltese/1.0 (personal study)"})
    r.raise_for_status()
    raw = bz2.decompress(r.content).decode("utf-8")
    cached.write_text(raw, encoding="utf-8")
    return raw


def load_pairs() -> list[tuple[str, str, str]]:
    """(maltese, english, sentence_id) for every aligned pair."""
    mlt = _fetch(f"{BASE}/mlt/mlt_sentences.tsv.bz2", "mlt_sentences.tsv")
    links = _fetch(f"{BASE}/mlt/mlt-eng_links.tsv.bz2", "mlt_eng_links.tsv")
    eng = _fetch(f"{BASE}/eng/eng_sentences.tsv.bz2", "eng_sentences.tsv")

    def index(raw: str) -> dict[str, str]:
        out = {}
        for row in csv.reader(io.StringIO(raw), delimiter="\t"):
            if len(row) >= 3:
                out[row[0]] = row[2]
        return out

    mt_by_id, en_by_id = index(mlt), index(eng)
    pairs, seen = [], set()
    for row in csv.reader(io.StringIO(links), delimiter="\t"):
        if len(row) < 2:
            continue
        a, b = row[0], row[1]
        if a in mt_by_id and b in en_by_id:
            mt = mt_by_id[a]
            if mt in seen:
                continue
            seen.add(mt)
            pairs.append((mt, en_by_id[b], a))
    return pairs


# ── Filtering and levelling ────────────────────────────────────────────────

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
BAD = re.compile(r"[<>{}|@#*_/\\]|\d{4}|https?://")

# Tatoeba is crowd-written and contains obscenity and violence — real Maltese, but
# not what a learner should be handed unprompted. Checked on the folded form so
# spelling variants do not slip through.
COARSE = {
    "zobb", "zibel", "ħara", "hara", "kelb ta", "mmur nitqabad",
    "stronz", "kaxxa nfern", "ghoxx", "għoxx", "liba", "fuck", "shit",
    "damn", "bastard", "bitch", "sex", "sexual", "kill", "murder", "rape",
    "suicide", "drug", "drugs", "whore",
}


def clean(mt: str, en: str) -> bool:
    joined = " " + text.fold(mt) + " " + en.lower() + " "
    return not any(f" {w} " in joined or w in text.fold(mt).split() for w in COARSE)


def usable(mt: str, en: str) -> bool:
    if not mt or not en or BAD.search(mt) or BAD.search(en):
        return False
    n = len(WORD.findall(mt))
    if not 2 <= n <= 14:
        return False
    if not mt.strip()[-1] in ".!?":
        return False
    # A sentence dense with capitalised words is usually about named people or
    # places, which teaches a name rather than the language.
    caps = sum(1 for w in mt.split()[1:] if w[:1].isupper())
    return caps <= 1


def deck_tiers() -> dict[str, int]:
    tiers: dict[str, int] = {}
    for r in (curriculum._read_tsv(curriculum.VOCAB_TSV)
              + curriculum._read_tsv(curriculum.PHRASES_TSV)):
        tier = int(r.get("tier") or 3)
        for w in re.split(r"[\s\-']+", r["mt"]):
            f = text.fold(w)
            if len(f) >= 2:
                tiers[f] = min(tier, tiers.get(f, 9))
    return tiers


def level_of(mt: str, tiers: dict[str, int]) -> tuple[str, float]:
    """Level a sentence by how much of it the learner already has.

    Length and unknown-word share are the two things that actually make a sentence
    hard to say, and both are measurable here. Grammar complexity is not, which is
    the main reason this is a proxy rather than a CEFR judgement.
    """
    words = [text.fold(w) for w in WORD.findall(mt)]
    words = [w for w in words if len(w) >= 2]
    if not words:
        return "C1", 0.0
    known = [w for w in words if w in tiers]
    share = len(known) / len(words)
    easy = sum(1 for w in known if tiers[w] <= 2) / len(words)
    n = len(words)

    if n <= 6 and share >= 0.9 and easy >= 0.7:
        return "A1", share
    if n <= 8 and share >= 0.75:
        return "A2", share
    if n <= 11 and share >= 0.55:
        return "B1", share
    if n <= 14 and share >= 0.35:
        return "B2", share
    return "C1", share


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-level", default="C1", choices=LEVELS)
    ap.add_argument("--report", action="store_true", help="measure only, write nothing")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print("Tatoeba mlt↔eng — CC BY 2.0 FR")
    pairs = load_pairs()
    print(f"  {len(pairs)} aligned pairs")

    tiers = deck_tiers()
    keep, by_level = [], Counter()
    cutoff = LEVELS.index(args.max_level)
    for mt, en, sid in pairs:
        if not usable(mt, en) or not clean(mt, en):
            continue
        lvl, share = level_of(mt, tiers)
        by_level[lvl] += 1
        if LEVELS.index(lvl) <= cutoff:
            keep.append({"id": f"tt{sid}", "mt": mt.strip(), "en": en.strip(),
                         "level": lvl, "known": f"{share:.2f}"})

    print(f"  {sum(by_level.values())} usable after filtering\n")
    print("  by level:")
    for lvl in LEVELS:
        print(f"    {lvl}  {by_level[lvl]:>4}")

    new_words = set()
    for row in keep:
        for w in WORD.findall(row["mt"]):
            f = text.fold(w)
            if len(f) >= 2 and f not in tiers:
                new_words.add(f)
    print(f"\n  {len(keep)} selected · {len(new_words)} word types not already in the deck")

    if args.report:
        print("\n  (report only — nothing written)")
        for row in keep[:8]:
            print(f"    [{row['level']}] {row['mt']}  ||  {row['en']}")
        return 0

    if args.limit:
        keep = keep[:args.limit]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        fh.write(f"# Auto-imported from {ATTRIBUTION}\n")
        fh.write("# UNVERIFIED: levels are a proxy from deck overlap and length, not a\n")
        fh.write("# CEFR judgement, and the Maltese is corpus text nobody here has checked.\n")
        w = csv.DictWriter(fh, delimiter="\t",
                           fieldnames=["id", "mt", "en", "level", "known"])
        w.writeheader()
        w.writerows(keep)
    print(f"\n✓ wrote {len(keep)} rows to {OUT.relative_to(DATA_DIR.parent)}")
    print("  Review before promoting any of it into core_vocab.tsv / phrases.tsv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

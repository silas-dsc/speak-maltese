#!/usr/bin/env python3
"""How much Maltese do the scripted dialogues actually put in front of you?

Counts distinct word *types* across everything a dialogue exposes — the lines the
app speaks, the answers it accepts, and its corrections — and measures that against
the curated deck. Comparison is on the folded form, so `Noqgħod` and `noqgħod` are
one word and a missing diacritic never counts as a miss.

    python scripts/coverage.py            # summary
    python scripts/coverage.py --missing  # deck words no dialogue has reached yet
    python scripts/coverage.py --missing --tier 1
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import curriculum, dialogue, text  # noqa: E402

WORD_SPLIT = re.compile(r"[\s\-']+")


def words_of(s: str) -> set[str]:
    out = set()
    for raw in WORD_SPLIT.split(s or ""):
        w = text.fold(raw)
        if len(w) >= 2:
            out.add(w)
    return out


def dialogue_words() -> tuple[Counter, int]:
    """Every Maltese word a learner meets in the scripted conversations."""
    counts: Counter = Counter()
    lines = 0
    for d in dialogue.all_dialogues():
        for n in d.get("nodes", {}).values():
            spoken = [n["say_mt"]]
            for key in ("correct", "close", "wrong"):
                if n.get(key, {}).get("mt"):
                    spoken.append(n[key]["mt"])
            spoken += [a["mt"] for a in n.get("accept", []) if not a.get("open")]
            for line in spoken:
                lines += 1
                counts.update(words_of(line))
    return counts, lines


def deck_words() -> dict[str, dict]:
    """Deck headwords keyed by folded form, with their tier."""
    out: dict[str, dict] = {}
    rows = (curriculum._read_tsv(curriculum.VOCAB_TSV)
            + curriculum._read_tsv(curriculum.PHRASES_TSV))
    for r in rows:
        for w in words_of(r["mt"]):
            out.setdefault(w, {"tier": int(r.get("tier") or 3),
                               "mt": r["mt"], "en": r["en"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missing", action="store_true")
    ap.add_argument("--tier", type=int, default=None)
    args = ap.parse_args()

    used, lines = dialogue_words()
    deck = deck_words()
    covered = {w for w in deck if w in used}  # noqa
    scenes = len(dialogue.all_dialogues())
    nodes = sum(len(d["nodes"]) for d in dialogue.all_dialogues())

    print(f"{scenes} scenes · {nodes} turns · {lines} spoken lines")
    print(f"{len(used)} distinct Maltese words appear in the dialogues\n")

    print(f"Deck coverage: {len(covered)}/{len(deck)} words  ({len(covered)/len(deck):.0%})")
    for tier in (1, 2, 3):
        t = {w for w, m in deck.items() if m["tier"] == tier}
        if not t:
            continue
        hit = len(t & covered)
        bar = "█" * round(20 * hit / len(t)) + "·" * (20 - round(20 * hit / len(t)))
        print(f"  tier {tier}  {bar}  {hit:>3}/{len(t):<3} ({hit/len(t):.0%})")

    beyond = len(set(used) - set(deck))
    print(f"\n{beyond} words appear in dialogue but are not deck headwords "
          f"(inflected forms, names, function words).")

    if args.missing:
        gaps = sorted(w for w in deck
                      if w not in used
                      and (args.tier is None or deck[w]["tier"] == args.tier))
        print(f"\nNot yet reached ({len(gaps)}):")
        for w in gaps:
            m = deck[w]
            print(f"  t{m['tier']}  {m['mt']:<28} {m['en']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

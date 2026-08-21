#!/usr/bin/env python3
"""Prompts for deliberate mistakes, across both classes and several kinds.

The first twenty of these were all one kind — a doubled consonant said single — and all one
class: attempts that *should* be marked close enough. A threshold cannot be fitted from one
class. It gives a lower bound and no upper bound, which is why the deployed one came from
"keep 93% of the honest recordings" rather than from the boundary itself, and why the only
wrong-side examples the grader has ever seen are reversed, hissing or quiet — none of which
is a wrong *Maltese attempt*.

So: two classes and several kinds.

  accept — a beginner's near-miss, which should be credited as close enough:
      geminate    a doubled consonant said once
      ghajn       `għ` sounded as a hard g, where it is silent and lengthens the vowel
      hkbira      `ħ` flattened to an English h
      ie          `ie` shortened to a plain i
      zeta        `ż` said as the `z` of `zokkor`, which in Maltese is ts
      xin         `x` said as s rather than sh

  reject — not the line that was asked for, and should be marked wrong:
      dropped     a whole word left out
      other       a different line from the deck said instead

`q` is deliberately absent. Its GOP is -5.37 on recordings that are *correct*, so the model
cannot hear a glottal stop at all and an error there measures nothing.

Every deck line is used, not only the ones with a pronunciation guide. Requiring a guide
sounded right and caps the whole set at 25 source sentences — `_guide` covers the original
recording script, not the deck, so 25 of 407 lines have one. Audio is the real aid anyway:
the recorder plays the correct line and then the mistake, and for five of the six accept
kinds the mistake is a *different phoneme* that text-to-speech renders audibly. Geminates
are the exception, where the rendering differs by one to three percent of duration, and
there the written instruction carries it.

    python scripts/make_error_prompts.py --accept 120 --reject 80
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import curriculum, text as mtext  # noqa: E402

DOUBLE = re.compile(r"([bcdfgjklmnpqrstvwxzċġħż])\1")


def geminate(flat: str) -> list[str]:
    out = []
    for m in DOUBLE.finditer(flat):
        i = m.start()
        out.append(flat[:i] + flat[i + 1:])
    return out


def ghajn(flat: str) -> list[str]:
    # `għ` is silent and lengthens the vowel beside it. An English speaker reads the g.
    return [flat.replace("għ", "g", 1)] if "għ" in flat else []


def hkbira(flat: str) -> list[str]:
    return [flat.replace("ħ", "h", 1)] if "ħ" in flat else []


def ie(flat: str) -> list[str]:
    return [flat.replace("ie", "i", 1)] if "ie" in flat else []


def zeta(flat: str) -> list[str]:
    return [flat.replace("ż", "z", 1)] if "ż" in flat else []


def xin(flat: str) -> list[str]:
    return [flat.replace("x", "s", 1)] if "x" in flat else []


def dropped(flat: str) -> list[str]:
    """A whole word left out — not a near-miss, a different sentence."""
    words = flat.split()
    if len(words) < 3:
        return []
    # Never the first word: the prompt has to stay recognisable as an attempt at this line.
    return [" ".join(words[:i] + words[i + 1:]) for i in range(1, len(words))]


ACCEPT_KINDS = {"geminate": geminate, "ghajn": ghajn, "hkbira": hkbira,
                "ie": ie, "zeta": zeta, "xin": xin}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept", type=int, default=120)
    ap.add_argument("--reject", type=int, default=80)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "error_prompts_200.tsv")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    from compare_stt import _guide

    en, guide = _guide()
    key = lambda x: mtext.normalise(x).lower().strip()  # noqa: E731
    guided = {key(k): v for k, v in guide.items() if v}
    glosses = {key(k): v for k, v in en.items()}

    lines = [r["mt"] for r in curriculum._read_tsv(curriculum.PHRASES_TSV)]
    lines += [r["ex_mt"] for r in curriculum._read_tsv(curriculum.VOCAB_TSV) if r.get("ex_mt")]
    flats = []
    seen = set()
    for ln in lines:
        f = key(ln)
        if f and f not in seen:
            seen.add(f)
            flats.append(f)
    if not flats:
        print("no deck lines found", file=sys.stderr)
        return 2
    with_guide = sum(1 for f in flats if f in guided)
    print(f"{len(flats)} deck lines, {with_guide} with a pronunciation guide\n")

    rng = random.Random(args.seed)
    rows = []

    # Accept side: round-robin over the kinds so no single kind dominates, which is what
    # made the first twenty unable to say anything about the others.
    pools = {k: [(f, v) for f in flats for v in fn(f) if v != f]
             for k, fn in ACCEPT_KINDS.items()}
    for pool in pools.values():
        rng.shuffle(pool)
    kinds = [k for k, v in pools.items() if v]
    i = 0
    while len([r for r in rows if r["class"] == "accept"]) < args.accept and kinds:
        k = kinds[i % len(kinds)]
        i += 1
        if not pools[k]:
            kinds = [x for x in kinds if pools[x]]
            continue
        intended, say = pools[k].pop()
        rows.append({"say": say, "intended": intended, "class": "accept", "kind": k,
                     "detail": k, "guide": guided.get(intended, ""),
                     "meaning": glosses.get(intended, "")})

    # Reject side: half a word dropped, half a different line entirely.
    drop_pool = [(f, v) for f in flats for v in dropped(f)]
    rng.shuffle(drop_pool)
    want_drop = args.reject // 2
    for intended, say in drop_pool[:want_drop]:
        rows.append({"say": say, "intended": intended, "class": "reject", "kind": "dropped",
                     "detail": "a word left out", "guide": guided.get(intended, ""),
                     "meaning": glosses.get(intended, "")})
    others = flats[:]
    rng.shuffle(others)
    for n in range(args.reject - want_drop):
        intended = others[n % len(others)]
        say = others[(n + 7) % len(others)]
        if say == intended:
            continue
        rows.append({"say": say, "intended": intended, "class": "reject", "kind": "other",
                     "detail": "a different line from the deck",
                     "guide": guided.get(say, ""), "meaning": glosses.get(say, "")})

    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, delimiter="\t",
                           fieldnames=["say", "intended", "class", "kind", "detail",
                                       "guide", "meaning"])
        w.writeheader()
        w.writerows(rows)

    by: dict[tuple[str, str], int] = {}
    for r in rows:
        by[(r["class"], r["kind"])] = by.get((r["class"], r["kind"]), 0) + 1
    print(f"{len(rows)} prompts → {args.out}")
    for (cls, kind), n in sorted(by.items()):
        print(f"  {cls:7} {kind:9} {n:>4}")
    print(f"\n  record them with:\n"
          f"    python scripts/compare_stt.py --record-errors {args.out} --input :3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

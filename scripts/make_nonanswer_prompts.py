#!/usr/bin/env python3
"""Prompts for the non-answers the app actually has to refuse.

The floor came down to 0.15 on the strength of 15 English and filler utterances rendered by
a text-to-speech voice, and that is a proxy in the place where a proxy most flatters the
result. A synthesised "I don't know" is cleanly articulated non-Maltese at full level. A
learner who is stuck is quiet, hesitant and fragmentary, and lands far closer to the
decision boundary — which is exactly where the change needs testing.

So these are recorded rather than synthesised. Each prompt names the deck line the app asked
for and what to do instead of answering it:

    silence   say nothing at all
    filler    um, er, a hesitation sound and nothing else
    english   answer in English, the honest thing a stuck learner does
    partial   start the Maltese line and stop after a syllable or two
    offtopic  say something unrelated, in English

Every one is class `reject`: the app must refuse it as the line it was asked for. `partial`
is the interesting one and the hardest — it *is* the beginning of the right answer, so it
sits between a near-miss and a non-answer, and where the grader puts it is worth knowing.

    python scripts/make_nonanswer_prompts.py --per-kind 8
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import curriculum, text as mtext  # noqa: E402

# What to do instead of answering. Written as an instruction to a person, because there is
# no line to imitate — which is the whole difference from the mispronunciation prompts.
KINDS = {
    "silence": "say NOTHING — stay quiet for a couple of seconds, then stop",
    "filler": "just a hesitation — “ummm” or “er”, no words",
    "english": "answer in ENGLISH — “I don't know”, “no idea”, whatever is honest",
    "partial": "start the Maltese line and STOP after a syllable or two",
    "offtopic": "say something unrelated, in English — “hello”, “what's for lunch”",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-kind", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "nonanswer_prompts.tsv")
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    lines = [r["mt"] for r in curriculum._read_tsv(curriculum.PHRASES_TSV)]
    lines += [r["ex_mt"] for r in curriculum._read_tsv(curriculum.VOCAB_TSV) if r.get("ex_mt")]
    key = lambda x: mtext.normalise(x).lower().strip()  # noqa: E731
    flats, seen = [], set()
    for ln in lines:
        f = key(ln)
        if f and f not in seen:
            seen.add(f)
            flats.append(f)
    if not flats:
        print("no deck lines found", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    pool = flats[:]
    rng.shuffle(pool)
    rows = []
    i = 0
    for kind, how in KINDS.items():
        for _ in range(args.per_kind):
            intended = pool[i % len(pool)]
            i += 1
            # `say` is an instruction, not a line: nothing is rendered for it, and the
            # recorder plays only the line that was asked for.
            rows.append({"say": how, "intended": intended, "class": "reject",
                         "kind": kind, "detail": how})

    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, delimiter="\t",
                           fieldnames=["say", "intended", "class", "kind", "detail"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} prompts → {args.out}")
    for kind in KINDS:
        print(f"  {kind:9} {sum(1 for r in rows if r['kind'] == kind):>3}")
    print(f"\n  record them with:\n"
          f"    python scripts/compare_stt.py --record-errors {args.out} --input :2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

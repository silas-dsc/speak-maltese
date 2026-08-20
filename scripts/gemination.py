#!/usr/bin/env python3
"""Does the model hear a doubled consonant?

Maltese distinguishes `kollox` from `kolox` by consonant length alone, and the README
records the failure plainly: `kolox` scores 1.02 against audio of `kollox` and is accepted.
That is the named blocker for the degemination work, and the aggregate app score barely
moves either way — so the aggregate cannot be the measurement. This is.

No new audio is needed. Every recording of a line containing a geminate already contains
the evidence: score the true spelling and its degeminated twin against the *same* audio and
see which the model prefers. A model that hears length prefers the truth; one that does not
is indifferent, and indifference is what lets a learner drop the doubling for free.

    python scripts/gemination.py --models frontend/stt
    python scripts/gemination.py --models data/distill/gem/onnx    # after 6d

Reported per pair, and as the only number that matters: how often the truth wins.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import text as mtext  # noqa: E402
from backend.config import DATA_DIR  # noqa: E402

CLIPS = DATA_DIR / "eval_clips"

# Doubled consonants only. A doubled vowel is a different thing in Maltese orthography and
# is not what the degemination work is about.
DOUBLE = re.compile(r"([bcdfgjklmnpqrstvwxzċġħż])\1")


def degeminate(flat: str) -> list[tuple[str, str]]:
    """Every single-geminate variant of a line, with the consonant that was halved.

    One at a time rather than all at once: a line with two geminates would otherwise be
    scored against a variant differing in two places, and a win would not say which
    length the model heard."""
    out = []
    for m in DOUBLE.finditer(flat):
        i = m.start()
        out.append((flat[:i] + flat[i + 1:], m.group(1)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="frontend/stt")
    ap.add_argument("--clips-dir", type=Path, default=CLIPS)
    args = ap.parse_args()

    from constrained_ctc import confidence, encode, load, rank_score
    from make_negatives import read_clip

    logprobs_for, vocab, blank, space = load(args.models)
    manifest = args.clips_dir / "manifest.tsv"
    with manifest.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t")
                if (r.get("text") or "").strip() and r["file"].startswith("me_")]

    print(f"{args.models}\n")
    print(f"  {'clip':12} {'doubled':>7} {'truth':>8} {'halved':>8} {'margin':>8}"
          f" {'rank':>7}  line")
    wins = total = rank_wins = 0
    for row in rows:
        flat = mtext.normalise(row["text"]).lower().strip()
        variants = degeminate(flat)
        if not variants:
            continue
        wave = read_clip(args.clips_dir / row["file"])
        if wave is None:
            continue
        post = logprobs_for(wave)
        true_ids = encode(flat, vocab, space)
        if not true_ids or len(true_ids) > len(post):
            continue
        c_true = confidence(post, true_ids, blank)
        r_true = rank_score(post, true_ids, blank)
        for short, letter in variants:
            ids = encode(short, vocab, space)
            if not ids or len(ids) > len(post):
                continue
            c_short = confidence(post, ids, blank)
            r_short = rank_score(post, ids, blank)
            total += 1
            won = c_true > c_short
            wins += won
            # The same comparison the app actually makes: the duration prior charges the
            # shorter spelling for being short, which is the bias that favours it here.
            rank_won = r_true > r_short
            rank_wins += rank_won
            print(f"  {row['file']:12} {letter!r:>7} {c_true:8.3f} {c_short:8.3f} "
                  f"{c_true - c_short:+8.3f} {'ok' if rank_won else 'LOST':>7}  "
                  f"{'' if won else '← halved wins  '}{flat}")
    if not total:
        print("  no recordings of lines containing a doubled consonant")
        return 1
    print(f"\n  on confidence alone the true spelling wins {wins}/{total} "
          f"({wins / total * 100:.0f}%)")
    print(f"  on rank, as the app compares it:                {rank_wins}/{total} "
          f"({rank_wins / total * 100:.0f}%)")
    print("\n  50% is a coin toss. Below it the model prefers the halved spelling, which is"
          "\n  the short-sequence bias again: dropping a doubled letter removes an"
          "\n  obligatory emission and makes the sequence easier to align. The duration"
          "\n  prior charges a hypothesis for being shorter than its frames, so the rank"
          "\n  column is the one that says whether the app is fooled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def score_errors(model: str, clips_dir: Path) -> int:
    """Does the grader catch a doubled consonant that was deliberately dropped?

    This is the question the honest recordings cannot answer. Each clip here is a
    mispronunciation, labelled with the line the speaker was *asked* for, so the grader is
    being asked exactly what the app asks: is this that line? Every accept is a learner
    told they were right when they were not.

    Reported against the whole deck rather than a sampled field, so the number does not
    move with a draw: the runner-up is the best any other line manages on the same audio."""
    from constrained_ctc import confidence, encode, load, rank_score

    from backend import dialogue
    from make_negatives import read_clip

    manifest = clips_dir / "errors" / "manifest.tsv"
    if not manifest.exists():
        print(f"no deliberate errors at {manifest}. Write prompts and record them:\n"
              f"  python scripts/gemination.py --models {model} "
              f"--write-prompts data/error_prompts.tsv\n"
              f"  python scripts/compare_stt.py --record-errors data/error_prompts.tsv "
              f"--input :3", file=sys.stderr)
        return 1
    with manifest.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    logprobs_for, vocab, blank, space = load(model)
    deck = [mtext.normalise(x).lower().strip() for x in dialogue.accepted_lines()]
    deck = [x for x in deck if x]

    print(f"{model} · {len(rows)} deliberate mispronunciations\n")
    print(f"  {'clip':12} {'conf':>7} {'floor':>6} {'field':>6}  said")
    caught = 0
    for row in rows:
        wave = read_clip(clips_dir / "errors" / row["file"])
        if wave is None:
            continue
        post = logprobs_for(wave)
        want = mtext.normalise(row["text"]).lower().strip()
        ids = encode(want, vocab, space)
        if not ids or len(ids) > len(post):
            continue
        conf = confidence(post, ids, blank)
        rank = rank_score(post, ids, blank)
        best_rival = -float("inf")
        for line in deck:
            if line == want:
                continue
            rid = encode(line, vocab, space)
            if rid and len(rid) <= len(post):
                best_rival = max(best_rival, rank_score(post, rid, blank))
        # The app's two conditions, reported apart: a floor that is too low and a field
        # that is too weak want different fixes, and one label would hide which.
        passes_floor = conf >= 0.35
        wins_field = rank > best_rival + 0.02
        accepted = passes_floor and wins_field
        caught += not accepted
        print(f"  {row['file']:12} {conf:7.3f} {'ok' if passes_floor else 'no':>6} "
              f"{'ok' if wins_field else 'no':>6}  "
              f"{'CAUGHT' if not accepted else 'passed as correct'}  {row['said']}")
    n = len([r for r in rows if (clips_dir / 'errors' / r['file']).exists()])
    if not n:
        print("  no audio found for any prompt")
        return 1
    print(f"\n  caught {caught}/{n} ({caught / n * 100:.0f}%)")
    print("  Every one not caught is a learner told they were right. This is the number")
    print("  degemination in the shard has to move; the aggregate app score will not show it.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="frontend/stt")
    ap.add_argument("--prompts", action="store_true",
                    help="list deliberate-error prompts worth recording, worst first")
    ap.add_argument("--write-prompts", type=Path, default=None,
                    help="write those prompts to a TSV the recorder can read")
    ap.add_argument("--errors", action="store_true",
                    help="score the deliberate mispronunciations instead of the honest clips")
    ap.add_argument("--want", type=int, default=20,
                    help="how many prompts to write; the marginal ones are the "
                         "informative ones, so this takes the worst N")
    ap.add_argument("--clips-dir", type=Path, default=CLIPS)
    args = ap.parse_args()

    if args.errors:
        return score_errors(args.models, args.clips_dir)

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
    prompts: list[tuple[float, str, str, str, str]] = []
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
            # Ranked by the margin the app actually decides on, so the prompts that come
            # first are the ones where a deliberate error would be most informative.
            prompts.append((r_true - r_short, row["file"], letter, flat, short))
            print(f"  {row['file']:12} {letter!r:>7} {c_true:8.3f} {c_short:8.3f} "
                  f"{c_true - c_short:+8.3f} {'ok' if rank_won else 'LOST':>7}  "
                  f"{'' if won else '← halved wins  '}{flat}")
    if not total:
        print("  no recordings of lines containing a doubled consonant")
        return 1
    if args.write_prompts:
        # Ordered by the margin the app decides on: the pairs it already gets wrong are
        # worth recording first, and the near-misses after them, because a prompt the
        # model handles comfortably cannot tell us anything when it is said wrong.
        # Prefer prompts the reader can actually say. Whoever records these is a learner,
        # not a Maltese speaker — the app exists for exactly that person — so a prompt with
        # no pronunciation guide is unusable however informative its margin would have
        # been. Guided first, worst margin first within that.
        from compare_stt import _guide

        _en, guide = _guide()
        guided = {mtext.normalise(k).lower().strip() for k, v in guide.items() if v}
        chosen = sorted(prompts, key=lambda r: (r[3] not in guided, r[0]))[:args.want]
        no_guide = sum(1 for r in chosen if r[3] not in guided)
        if no_guide:
            print(f"  note: {no_guide} of these have no pronunciation guide — a learner "
                  f"cannot say them from the spelling alone")
        with args.write_prompts.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, delimiter="\t",
                               fieldnames=["say", "intended", "halved", "margin", "heard_in"])
            w.writeheader()
            for margin, clip, letter, flat, short in chosen:
                w.writerow({"say": short, "intended": flat, "halved": letter,
                            "margin": f"{margin:.4f}", "heard_in": clip})
        wrong = sum(1 for m, *_ in chosen if m < 0)
        print(f"\n  {len(chosen)} prompts → {args.write_prompts} "
              f"({wrong} the model gets wrong today)")
        print(f"  record them with:\n"
              f"    python scripts/compare_stt.py --record-errors {args.write_prompts} "
              f"--input :3")
        return 0

    if args.prompts:
        print("\n  Deliberate errors worth recording, the ones the model gets wrong first.")
        print("  Every clip in the set is an honest attempt, so nothing here measures")
        print("  whether a learner who drops the doubling is *caught* — only whether the")
        print("  model could tell if asked. Say the halved spelling on purpose and the")
        print("  gap closes.\n")
        for margin, clip, letter, flat, short in sorted(prompts):
            state = "wrong now" if margin < 0 else "right now"
            print(f"  {state}  margin {margin:+.3f}  say {short!r}")
            print(f"                              for  {flat!r}  (halved {letter!r}, "
                  f"heard in {clip})")
        return 0

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

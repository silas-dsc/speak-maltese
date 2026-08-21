#!/usr/bin/env python3
"""Cross-validate the near-miss verdict, because its thresholds were fitted on the test set.

`GOP_IGNORE` and `GOP_MIN` were both chosen on the same 75 recordings the verdict was then
reported against: the ignore-set is the graphemes whose median GOP is worst on those clips,
and the threshold is the value that keeps 93% of them. Numbers obtained that way are an
upper bound on what a new speaker would see, and the gap between the two is not knowable
by looking harder at the same clips.

So: refit both on K-1 folds and report the held-out fold. Repeated over the folds this
gives an honest estimate, and the spread across folds says whether the thresholds are a
property of Maltese or of this speaker's recordings.

    python scripts/gop_cv.py --folds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.config import DATA_DIR  # noqa: E402

CACHE = DATA_DIR / "eval_clips" / "gop_cache.json"


def fit(rows: list[dict], ignore_cut: float, keep: float) -> tuple[set[str], float]:
    """The ignore-set and threshold, from these clips only."""
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for t, g in zip(r["tokens"], r["gop"]):
            by[t].append(g)
    ignore = {t for t, v in by.items() if float(np.median(v)) <= ignore_cut}
    scores = sorted(score(r, ignore) for r in rows)
    scores = [s for s in scores if not np.isnan(s)]
    if not scores:
        return ignore, -np.inf
    # The threshold that keeps `keep` of these clips, which is how GOP_MIN was chosen.
    at = max(0, int(len(scores) * (1.0 - keep)) - 1)
    return ignore, scores[at]


def score(row: dict, ignore: set[str]) -> float:
    kept = [g for t, g in zip(row["tokens"], row["gop"]) if t not in ignore and t != "␣"]
    return float(np.mean(kept)) if kept else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ignore-cut", type=float, default=-1.0)
    ap.add_argument("--keep", type=float, default=0.93)
    ap.add_argument("--cache", type=Path, default=CACHE)
    args = ap.parse_args()

    if not args.cache.exists():
        print(f"no cache at {args.cache} — run scripts/gop.py --models frontend/stt",
              file=sys.stderr)
        return 2
    rows = json.loads(args.cache.read_text())
    learner = [r for r in rows if r["kind"] == "learner"]
    negs = [r for r in rows if r["kind"] != "learner"]
    if len(learner) < args.folds:
        print("not enough clips to fold", file=sys.stderr)
        return 2

    # Deterministic folds by clip name, so a rerun says the same thing.
    order = sorted(learner, key=lambda r: r["clip"])
    folds = [order[i::args.folds] for i in range(args.folds)]

    print(f"{args.folds}-fold, refitting the ignore-set and threshold on each split\n")
    print(f"  {'fold':>5} {'n':>4} {'thr':>7} {'ignored':>8} {'held-out kept':>14} "
          f"{'reversed':>9} {'hiss':>6}")
    kept_all, rev_all, thr_all, ign_sizes = [], [], [], []
    for i, held in enumerate(folds):
        train = [r for j, f in enumerate(folds) if j != i for r in f]
        ignore, thr = fit(train, args.ignore_cut, args.keep)
        ok = [score(r, ignore) for r in held]
        kept = sum(1 for s in ok if not np.isnan(s) and s >= thr) / max(1, len(ok))
        rev = [score(r, ignore) for r in negs if r["kind"] == "reversed"]
        rev_rate = sum(1 for s in rev if not np.isnan(s) and s >= thr) / max(1, len(rev))
        hiss = [score(r, ignore) for r in negs if r["kind"] == "hiss"]
        hiss_rate = sum(1 for s in hiss if not np.isnan(s) and s >= thr) / max(1, len(hiss))
        kept_all.append(kept); rev_all.append(rev_rate); thr_all.append(thr)
        ign_sizes.append(len(ignore))
        print(f"  {i + 1:>5} {len(held):>4} {thr:>7.2f} {len(ignore):>8} "
              f"{kept * 100:>13.0f}% {rev_rate * 100:>8.0f}% {hiss_rate * 100:>5.0f}%")

    print(f"\n  held-out kept:  {np.mean(kept_all) * 100:.0f}% "
          f"(spread {min(kept_all) * 100:.0f}-{max(kept_all) * 100:.0f}%)")
    print(f"  reversed let in: {np.mean(rev_all) * 100:.1f}% "
          f"(spread {min(rev_all) * 100:.0f}-{max(rev_all) * 100:.0f}%)")
    print(f"  threshold:      {np.mean(thr_all):.2f} "
          f"(spread {min(thr_all):.2f} to {max(thr_all):.2f})")
    print(f"  ignored tokens: {np.mean(ign_sizes):.1f} of them, varying by fold")
    print("\n  A threshold that moves a lot between folds is a property of these clips")
    print("  rather than of Maltese, and the held-out figure is what a new speaker sees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

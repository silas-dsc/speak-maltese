#!/usr/bin/env python3
"""Per-token Goodness of Pronunciation, from the posteriors the app already computes.

The grader asks one question — *is this the line it was asked for* — and answers it with a
single number for the whole utterance. That is why three of the learner's recordings are
refused outright: one sound the model cannot do sinks a sentence that was otherwise right,
and `--probe-loss` showed the loss cannot be forgiven at the utterance level without
admitting half of all backwards speech.

Goodness of Pronunciation asks the question per sound instead. Classically it needs forced
alignment to know which frames belong to which phone; the segmentation-free form
(arXiv 2507.16838) drops that by taking the alignment CTC already implies. For each target
token, the forward-backward pass gives an occupancy over frames, and the score is the
posterior-weighted margin between the token and whatever the model would rather have said:

    GOP(i) = sum_t gamma_i(t) * (log p(y_i | t) - max_v log p(v | t)) / sum_t gamma_i(t)

Zero means "at every frame this token owns, it was also the model's first choice". More
negative means the model wanted something else there. No lexicon, no aligner, one extra
forward-backward over posteriors that have already been computed.

Two things this makes possible that an utterance score cannot:

  * Saying *which* sound was wrong, rather than refusing the whole attempt.
  * Telling apart a sound the learner got wrong from a sound the model cannot hear. Those
    look identical in an utterance score and want opposite treatment — and this model is
    known to be unable to hear gh and q, which is not the learner's fault.

    python scripts/gop.py --models frontend/stt      # score, cache, and profile
    python scripts/gop.py --report                   # read the cache back
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import text as mtext  # noqa: E402
from backend.config import DATA_DIR  # noqa: E402

CLIPS = DATA_DIR / "eval_clips"
CACHE = CLIPS / "gop_cache.json"
NEG_INF = -1e30


def _logsumexp2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Stable log(exp(a) + exp(b)) elementwise, with -inf held at -inf."""
    hi = np.maximum(a, b)
    lo = np.minimum(a, b)
    out = hi + np.log1p(np.exp(np.clip(lo - hi, -700, 0)))
    return np.where(hi <= NEG_INF / 2, NEG_INF, out)


def occupancy(post: np.ndarray, ids: list[int], blank: int) -> np.ndarray:
    """Per-target-token frame occupancy, from the CTC forward-backward.

    Returns an (L, T) array whose row i says how much of each frame target token i owns.
    This is the alignment CTC already implies, summed over every path rather than committed
    to the single best one — which is the point: a learner's timing is exactly what a forced
    alignment gets wrong, and a soft occupancy does not have to choose.
    """
    T = len(post)
    L = len(ids)
    # The standard extended sequence: a blank between every token, and at both ends.
    ext = np.full(2 * L + 1, blank, dtype=np.int64)
    ext[1::2] = ids
    S = len(ext)

    emit = post[:, ext]                                   # (T, S)
    # A token may be skipped into only if it differs from the one two back — the usual
    # CTC rule that forbids collapsing a real repeat into one emission.
    can_skip = np.zeros(S, dtype=bool)
    can_skip[2:] = (ext[2:] != ext[:-2]) & (ext[2:] != blank)

    alpha = np.full((T, S), NEG_INF)
    alpha[0, 0] = emit[0, 0]
    if S > 1:
        alpha[0, 1] = emit[0, 1]
    for t in range(1, T):
        prev = alpha[t - 1]
        same = prev
        one = np.concatenate(([NEG_INF], prev[:-1]))
        two = np.concatenate(([NEG_INF, NEG_INF], prev[:-2]))
        two = np.where(can_skip, two, NEG_INF)
        alpha[t] = _logsumexp2(_logsumexp2(same, one), two) + emit[t]

    beta = np.full((T, S), NEG_INF)
    beta[T - 1, S - 1] = 0.0
    if S > 1:
        beta[T - 1, S - 2] = 0.0
    for t in range(T - 2, -1, -1):
        nxt = beta[t + 1] + emit[t + 1]
        same = nxt
        one = np.concatenate((nxt[1:], [NEG_INF]))
        skip_from = np.concatenate((can_skip[2:], [False, False]))
        two = np.concatenate((nxt[2:], [NEG_INF, NEG_INF]))
        two = np.where(skip_from, two, NEG_INF)
        beta[t] = _logsumexp2(_logsumexp2(same, one), two)

    total = _logsumexp2(alpha[T - 1, S - 1], alpha[T - 1, S - 2] if S > 1 else NEG_INF)
    if total <= NEG_INF / 2 or not np.isfinite(total):
        return np.zeros((L, T))
    gamma = np.exp(np.clip(alpha + beta - total, -700, 0))     # (T, S)
    return gamma[:, 1::2].T                                    # (L, T), tokens only


def token_gop(post: np.ndarray, ids: list[int], blank: int) -> np.ndarray:
    """GOP per target token: the occupancy-weighted margin against the model's own choice."""
    gam = occupancy(post, ids, blank)
    best = post.max(axis=1)                                    # (T,)
    out = np.zeros(len(ids))
    for i, tok in enumerate(ids):
        w = gam[i]
        mass = w.sum()
        if mass <= 1e-9:
            out[i] = float(post[:, tok].max() - best.max())     # never aligned anywhere
            continue
        out[i] = float((w * (post[:, tok] - best)).sum() / mass)
    return out


# Duplicated in `frontend/nanostt.js` on purpose — the frontend cannot import Python —
# and pinned together by `tests/test_client_gop.py`, the same arrangement the duration
# constants use.

# Graphemes whose score says nothing about the learner: `q` because the model cannot hear
# a glottal stop at all, and `g`/`h`/`'` because `għ` is silent, so there is no sound to
# find and the model is right to fail. Charging a learner for these charges them for the
# model's blind spots. Measured on the 75 recordings that are correct.
GOP_IGNORE = frozenset({"'", "d", "g", "h", "j", "q", "r", "ż"})

# Below this the sounds are not the ones the line asks for. Set where it holds 93% of the
# learner's own recordings while refusing every time-reversed clip — the negative the
# confidence floor is worst at, admitting 67% of them.
GOP_MIN = -2.29


def gop_score(post: np.ndarray, ids: list[int], toks: list[str], blank: int) -> float:
    """One number for the attempt: the mean over tokens the model can be trusted on.

    NaN when the line is made entirely of ignored tokens. That is not a verdict and callers
    must not read it as one — "no opinion" and "wrong" want opposite treatment."""
    g = token_gop(post, ids, blank)
    kept = [x for tok, x in zip(toks, g) if tok not in GOP_IGNORE]
    return float(np.mean(kept)) if kept else float("nan")


def worst_sound(post: np.ndarray, ids: list[int], toks: list[str],
                blank: int) -> dict | None:
    """The worst-scoring token worth telling the learner about, skipping blind spots."""
    g = token_gop(post, ids, blank)
    pairs = [(x, i) for i, (tok, x) in enumerate(zip(toks, g)) if tok not in GOP_IGNORE]
    if not pairs:
        return None
    x, i = min(pairs)
    return {"token": toks[i], "index": i, "gop": float(x)}


def profile(rows: list[dict]) -> dict[str, dict]:
    """Per-grapheme GOP over *correct* recordings.

    A token that scores badly on speech known to be right is not a token the learner is
    getting wrong — it is a token the model cannot hear. Separating those is the whole
    reason for scoring per token, and it cannot be done from an utterance score at all."""
    by: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["kind"] != "learner":
            continue
        for tok, score in zip(row["tokens"], row["gop"]):
            by[tok].append(score)
    out = {}
    for tok, vals in by.items():
        v = np.array(vals)
        out[tok] = {"n": len(v), "mean": float(v.mean()),
                    "median": float(np.median(v)), "p10": float(np.percentile(v, 10))}
    return out


def collect(model: str, clips_dir: Path) -> list[dict]:
    from constrained_ctc import load
    from make_negatives import read_clip, read_negatives

    from gop_encode import encode_tokens

    logprobs_for, vocab, blank, space = load(model)
    manifest = clips_dir / "manifest.tsv"
    with manifest.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t")
                if (r.get("text") or "").strip() and r["file"].startswith("me_")]

    records = []

    def one(wave, flat: str, kind: str, name: str) -> None:
        post = logprobs_for(wave)
        ids, toks = encode_tokens(flat, vocab, space)
        if not ids or len(ids) > len(post):
            return
        g = token_gop(post, ids, blank)
        records.append({"clip": name, "kind": kind, "text": flat,
                        "frames": int(len(post)), "tokens": toks,
                        "gop": [float(x) for x in g]})

    for i, row in enumerate(rows, 1):
        wave = read_clip(clips_dir / row["file"])
        if wave is None:
            continue
        one(wave, mtext.normalise(row["text"]).lower().strip(), "learner", row["file"])
        print(f"  {i}/{len(rows)} {row['file']}", flush=True)

    # Negatives are scored against the line their source clip was reading, so a rule built
    # on these numbers is measured against the same thing the grader faces.
    src = {r["file"]: mtext.normalise(r["text"]).lower().strip() for r in rows}
    for neg in read_negatives():
        wave = read_clip(clips_dir / "negatives" / neg["file"])
        if wave is None:
            continue
        origin = neg.get("source", "-")
        flat = src.get(origin) or next(iter(src.values()))
        one(wave, flat, neg["kind"], neg["file"])
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="model dir; omit to read the cache")
    ap.add_argument("--clips-dir", type=Path, default=CLIPS)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.models:
        rows = collect(args.models, args.clips_dir)
        if not rows:
            print("nothing scored", file=sys.stderr)
            return 2
        args.cache.write_text(json.dumps(rows), encoding="utf-8")
        print(f"\n{len(rows)} clips scored → {args.cache}")
    else:
        if not args.cache.exists():
            print(f"no cache at {args.cache} — run once with --models", file=sys.stderr)
            return 2
        rows = json.loads(args.cache.read_text(encoding="utf-8"))

    prof = profile(rows)
    print(f"\nper-grapheme GOP on {sum(1 for r in rows if r['kind']=='learner')} correct "
          f"recordings — 0 is perfect, more negative means the model wanted something else")
    print(f"  {'tok':>5} {'n':>5} {'median':>9} {'p10':>9}")
    for tok, d in sorted(prof.items(), key=lambda kv: kv[1]["median"]):
        print(f"  {tok!r:>5} {d['n']:>5} {d['median']:>9.3f} {d['p10']:>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

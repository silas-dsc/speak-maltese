#!/usr/bin/env python3
"""What length should a hypothesis of this many tokens have claimed?

`constrained_ctc.rank_score` charges a hypothesis for claiming a length the audio
cannot support, and the charge is `-½z²` on a straight line fitted through frames
against tokens. Three constants carry that line and a fourth, `DUR_WEIGHT`, decides how
loudly it gets to speak. All four were fitted once, reported in the README, and have
never been re-measured. This re-measures them, and — more to the point — measures the
thing the ranking actually consumes, which is not the fit.

    python scripts/fit_duration.py                    # fit and report
    python scripts/fit_duration.py --frames speech     # ...against trimmed audio
    python scripts/fit_duration.py --corpus data/distill   # from the real corpus
    python scripts/fit_duration.py --json fit.json     # write the constants out

Two sources. By default it reads `data/audio_cache`, which holds `edge-tts` renders of
deck lines — the text is known exactly, so frames and tokens correspond. `--corpus`
instead reads a `distill_stt.py teacher` shard, which is what the published fit used
and the only source that can settle whether the published numbers reproduce.

**Read the differential table, not the fit.** A prior that describes the corpus well is
not the same as a prior that ranks well, and it is easy to improve the first while
breaking the second. `rank_score` compares hypotheses against one recording, so `frames`
is held fixed and only `tokens` varies: every hypothesis takes whatever common penalty
the audio's length implies, and that part cancels. What survives is the *difference*
between the charge on the true line and the charge on a rival, and the rival that matters
is the shortest thing in the field — `Bonġu!`, five tokens, which is what every one of
the five documented ranking failures lost to. So the last table asks the only question
worth asking of a change here: does it still charge five tokens more than the truth, by
enough to reverse those five, and not by so much that it becomes λ = 0.3 under another
name.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import text as mtext  # noqa: E402
from backend.config import AUDIO_CACHE, CFG, DATA_DIR  # noqa: E402
from audio_io import read_audio  # noqa: E402
from constrained_ctc import (  # noqa: E402
    DUR_INTERCEPT, DUR_SLOPE, DUR_SD, DUR_WEIGHT, SPEECH_DROP_DB,
    frame_energy, speech_span,
)

VOCAB = ROOT / "frontend" / "stt" / "vocab.txt"
SR = 16000
HOP = 160

# The (voice, rate) pairs `distill_stt.py` renders, so the same clips are found here.
VARIANTS = [
    (CFG.azure_voice, 0.95),
    (CFG.azure_voice_alt, 0.95),
    (CFG.azure_voice, 1.10),
    (CFG.azure_voice_alt, 0.85),
]

# The rival every documented ranking failure lost to: the shortest line in the field.
SHORT_RIVAL_TOKENS = 5

# What the five documented failures were beaten by, in confidence units. A change has to
# clear this to fix them: 0.679 against 0.815, 0.573 against 0.728, 0.432 against 0.571,
# 0.243 against 0.377, 0.494 against 0.633 — the widest of which is 0.155.
NEEDED_DIFFERENTIAL = 0.155


def vocabulary() -> dict[str, int]:
    out = {}
    for line in VOCAB.read_text(encoding="utf-8").splitlines():
        if " " in line:
            tok, idx = line.rsplit(" ", 1)
            out[tok] = int(idx)
    return out


def encode(flat: str, vocab: dict[str, int]) -> list[int]:
    """Characters to ids, spaces to `▁`, exactly as `nanostt.encodeTarget` does."""
    return [vocab["▁" if ch == " " else ch] for ch in flat
            if ("▁" if ch == " " else ch) in vocab]


# ── Sources ────────────────────────────────────────────────────────────────

def load_wave(path: Path) -> np.ndarray | None:
    """Any cached render to mono 16kHz float, by whichever decoder is installed.

    See `audio_io.read_audio`: either `soundfile` or the `faster_whisper` already in
    `requirements.txt` will do, so fitting needs no install of its own."""
    return read_audio(path, SR)


def from_cache(drop_db: float, limit: int | None = None) -> list[dict]:
    """Deck lines that have been rendered, paired with the render's length."""
    from backend import tts
    from prebuild_audio import lines_for

    vocab = vocabulary()
    rows, seen = [], set()
    for line in lines_for("all"):
        flat = mtext.normalise(line).lower().strip()
        if not flat or flat in seen:
            continue
        seen.add(flat)
        ids = encode(flat, vocab)
        if not ids:
            continue
        for voice, rate in VARIANTS:
            path = AUDIO_CACHE / f"{tts._cache_key(line, voice, rate, 'edge')}.mp3"
            if not path.exists():
                continue
            wave = load_wave(path)
            if wave is None or wave.size < 400:
                continue
            energy = frame_energy(wave)
            start, end = speech_span(energy, drop_db)
            rows.append({"tokens": len(ids),
                         "total": len(energy) // 2,
                         "speech": max(0, end // 2 - start // 2)})
            if limit and len(rows) >= limit:
                return rows
    return rows


def from_corpus(work: Path, limit: int | None = None) -> list[dict]:
    """A `distill_stt.py teacher` shard — the source the published fit used.

    Only the passes that carry a text label are usable: FLEURS chunks are three-second
    cuts of Wikipedia prose with a guessed transcript, so their frames and tokens are
    only loosely related, which is exactly why the published fit set them aside. The
    stored `frames` is already the student's output frame count, and it is the whole
    pass, so there is no speech span to be had here."""
    vocab = vocabulary()
    rows = []
    for meta_path in sorted(work.glob("index_*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for item in meta["items"]:
            flat = (item.get("text") or "").strip()
            if not flat or (item.get("source") or "").startswith("accent"):
                continue
            ids = encode(flat, vocab)
            if not ids:
                continue
            rows.append({"tokens": len(ids), "total": int(item["frames"]),
                         "speech": int(item["frames"])})
            if limit and len(rows) >= limit:
                return rows
    return rows


# ── Fitting ────────────────────────────────────────────────────────────────

def fit_line(tokens: np.ndarray, frames: np.ndarray) -> tuple[float, float, float]:
    """Ordinary least squares, and the residual spread it leaves behind."""
    design = np.c_[np.ones_like(tokens), tokens]
    coef, *_ = np.linalg.lstsq(design, frames, rcond=None)
    resid = frames - design @ coef
    return float(coef[0]), float(coef[1]), float(resid.std(ddof=2))


def fit_sd(tokens: np.ndarray, resid: np.ndarray) -> tuple[float, float]:
    """`sd ≈ s0 + s1 × tokens`, through the absolute residuals.

    A half-normal has mean `sd·√(2/π)`, so scaling |resid| by the reciprocal turns a fit
    of the absolute residual into a fit of the standard deviation."""
    design = np.c_[np.ones_like(tokens), tokens]
    coef, *_ = np.linalg.lstsq(design, np.abs(resid) * np.sqrt(np.pi / 2), rcond=None)
    return float(coef[0]), float(coef[1])


def prior(tokens, frames, intercept, slope, sd) -> np.ndarray:
    return -0.5 * ((frames - (intercept + slope * tokens)) / np.maximum(1e-6, sd)) ** 2


def calibration(tokens, frames, intercept, slope, sd) -> tuple[float, float]:
    """A z that is doing its job has sd 1 and almost nothing past 3."""
    z = (frames - (intercept + slope * tokens)) / np.maximum(1e-6, sd)
    return float(z.std()), float((np.abs(z) > 3).mean())


def differential(tokens, frames, intercept, slope, sd_true, sd_rival) -> np.ndarray:
    """What the prior charges the short rival over the truth, in confidence units."""
    return DUR_WEIGHT * (prior(tokens, frames, intercept, slope, sd_true)
                         - prior(SHORT_RIVAL_TOKENS, frames, intercept, slope, sd_rival))


# ── Report ─────────────────────────────────────────────────────────────────

BINS = (0, 6, 10, 14, 18, 24, 30, 40, 60, 10_000)


def report(rows: list[dict], which: str, lo: int, hi: int) -> dict:
    tokens = np.array([r["tokens"] for r in rows], dtype=float)
    frames = np.array([r[which] for r in rows], dtype=float)
    keep = (tokens >= lo) & (tokens <= hi)
    tokens, frames = tokens[keep], frames[keep]
    if len(tokens) < 30:
        print(f"only {len(tokens)} rows in {lo}-{hi} tokens — not enough to fit",
              file=sys.stderr)
        return {}

    print(f"{len(tokens)} passes · {which} frames · tokens {lo}-{hi}\n")
    intercept, slope, sd = fit_line(tokens, frames)
    resid = frames - (intercept + slope * tokens)
    s0, s1 = fit_sd(tokens, resid)
    print(f"  refit      frames ~ {intercept:7.2f} + {slope:.4f} * tokens   sd {sd:.2f}")
    print(f"  deployed   frames ~ {DUR_INTERCEPT:7.2f} + {DUR_SLOPE:.4f} * tokens   "
          f"sd {DUR_SD:.2f}")
    print(f"  sd(tokens) ~ {s0:.3f} + {s1:.4f} * tokens\n")

    print("  residual spread by length — one constant cannot describe this")
    print(f"  {'tokens':>10} {'n':>6} {'mean tok':>9} {'resid sd':>9}")
    for a, b in zip(BINS, BINS[1:]):
        m = (tokens >= a) & (tokens < b)
        if m.sum() < 15:
            continue
        label = f"{a}-{b}" if b < 10_000 else f"{a}+"
        print(f"  {label:>10} {m.sum():6d} {tokens[m].mean():9.1f} {resid[m].std():9.2f}")

    configs = [
        ("deployed", DUR_INTERCEPT, DUR_SLOPE, DUR_SD, DUR_SD),
        ("refit, one sd", intercept, slope, sd, sd),
        ("refit, sd(tokens)", intercept, slope,
         s0 + s1 * tokens, s0 + s1 * SHORT_RIVAL_TOKENS),
    ]

    print(f"\n  calibration — z wants sd 1.0 and a thin tail")
    print(f"  {'config':22} {'z sd':>7} {'|z|>3':>8}")
    for name, a, b, sd_t, _ in configs:
        zsd, tail = calibration(tokens, frames, a, b, sd_t)
        print(f"  {name:22} {zsd:7.3f} {tail * 100:7.1f}%")

    print(f"\n  what {SHORT_RIVAL_TOKENS} tokens is charged over the truth, in "
          f"confidence units")
    print(f"  (needs >= {NEEDED_DIFFERENTIAL:.3f} to reverse the documented failures)")
    print(f"  {'config':22} {'median':>9} {'10th pct':>9} {'reverses':>9}")
    out = {}
    for name, a, b, sd_t, sd_r in configs:
        g = differential(tokens, frames, a, b, sd_t, sd_r)
        ok = float((g >= NEEDED_DIFFERENTIAL).mean())
        print(f"  {name:22} {np.median(g):+9.3f} {np.percentile(g, 10):+9.3f} "
              f"{ok * 100:8.1f}%")
        out[name] = {"median": float(np.median(g)), "reverses": ok}

    return {"n": int(len(tokens)), "frames": which,
            "intercept": intercept, "slope": slope, "sd": sd,
            "sd_intercept": s0, "sd_slope": s1, "differential": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", choices=["total", "speech", "both"], default="both",
                    help="count every frame, or only the frames that are speech")
    ap.add_argument("--corpus", type=Path, default=None,
                    help="a distill_stt.py shard directory (default: the audio cache)")
    ap.add_argument("--drop-db", type=float, default=SPEECH_DROP_DB,
                    help="how far under the loudest frame still counts as speech")
    ap.add_argument("--min-tokens", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=62,
                    help="the deployed field spans 8-62 tokens; outside it the line "
                         "bends and the fit stops meaning anything")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", type=Path, default=None, help="write the fit out")
    args = ap.parse_args()

    if args.corpus:
        rows = from_corpus(args.corpus, args.limit)
        if args.frames != "total":
            print("a shard stores whole passes, so only --frames total is available "
                  "from --corpus", file=sys.stderr)
        wants = ["total"]
    else:
        rows = from_cache(args.drop_db, args.limit)
        wants = ["total", "speech"] if args.frames == "both" else [args.frames]

    if not rows:
        print("nothing to fit — render some audio with scripts/prebuild_audio.py",
              file=sys.stderr)
        return 2

    pad = np.array([r["total"] - r["speech"] for r in rows], dtype=float)
    if pad.any():
        print(f"padding trimmed: mean {pad.mean():.1f} frames, sd {pad.std():.1f}, "
              f"range {int(pad.min())}-{int(pad.max())}")
        print("(a near-constant pad is absorbed by the intercept and leaves the "
              "residual alone)\n")

    out = {}
    for which in wants:
        out[which] = report(rows, which, args.min_tokens, args.max_tokens)
        print()

    if args.json:
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"→ {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

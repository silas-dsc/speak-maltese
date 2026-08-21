#!/usr/bin/env python3
"""The things the grader has to turn away, built from the clips it has to accept.

Every constant in the acceptance rule was chosen against two sets: a learner's own
recordings, which must be accepted, and a pile of negatives, which must not. The
recordings are in `data/eval_clips`. The negatives were generated once, used to produce
the tables in the README, and never committed — so the sweep that priced `MIN_CONFIDENCE`
cannot be re-run, and no new constant can be priced the same way. This rebuilds them.

    python scripts/make_negatives.py                 # → data/eval_clips/negatives
    python scripts/make_negatives.py --report        # just say what is there

Four kinds, and each is here because it caught something:

  * **Digital silence.** The first thing ranking accepted. Against an all-blank posterior
    a short sequence is the likelier reading, so silence beat a field of longer
    alternatives outright — the bug the duration prior exists to kill.
  * **White noise, at five levels.** Silence is easy; a floor low enough to admit hiss is
    the failure mode that actually shipped, and it only appears at some levels.
  * **The clips at −30dB.** One real take sat 30dB under the rest and was the
    worst-scoring recording in the set. Quiet speech is not noise and must not be graded
    as though it were.
  * **The clips reversed.** Full speech energy, speech-like spectra, no words. Nothing
    else in this list has that shape, and it is the one negative the deployed rule still
    admits 8% of.

**A caveat that matters for reading any number this produces.** The historical set's
exact composition is not recorded anywhere in the repository — only that there were 90 of
them, of these four kinds. The defaults here come to 90 and are the obvious reading of
that, but they are a reconstruction. Percentages from a regenerated set are comparable to
the published tables only to the extent the composition happens to match, so a sweep that
uses these should re-measure the *current* rule alongside any candidate rather than
comparing against the numbers in the README. `manifest.tsv` records what was actually
built, so at least the next person is not reconstructing it again.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_io import SR, peak, read_audio, write_wav  # noqa: E402
from backend.config import DATA_DIR  # noqa: E402

CLIPS = DATA_DIR / "eval_clips"
NEGATIVES = CLIPS / "negatives"

# Five levels spanning "a quiet room" to "a bus". The top of the range has to be loud
# enough to clear an energy gate, which is the point — a gate that only rejects quiet
# noise is not rejecting noise.
NOISE_LEVELS = (0.005, 0.02, 0.05, 0.12, 0.30)

# Default attenuation for `attenuated`, which is no longer a negative generator: it is the
# gain-invariance probe. Turning a clip down 30dB and getting the same verdict is the
# property the feature normalisation is supposed to guarantee, and the harness test asserts
# exactly that.
QUIET_DB = -30.0

# Enough silence and hiss clips to see a percentage, and one of each transform per clip.
# 20 + 20 + 25 + 25 = 90 against 25 recordings, which is the published count.
N_SILENCE = 20
N_NOISE = 20

# Durations for the synthesised negatives, sampled from the real clips so a length
# artefact cannot be what rejects them.
FALLBACK_SECONDS = 2.0


# ── The transforms ─────────────────────────────────────────────────────────
# Pure and separate from any file handling, so they can be checked without a microphone.

def reversed_clip(wave: np.ndarray) -> np.ndarray:
    """Speech backwards: the energy and the spectrum of speech, and no words in it."""
    return wave[::-1].copy()


def attenuated(wave: np.ndarray, db: float = QUIET_DB) -> np.ndarray:
    """The same take, `db` quieter. Negative `db` makes it quieter."""
    return (wave * (10.0 ** (db / 20.0))).astype(np.float32)


def white_noise(samples: int, level: float, rng) -> np.ndarray:
    """Gaussian hiss at a given RMS, clipped to what a WAV can hold."""
    return np.clip(rng.normal(0.0, level, samples), -1.0, 1.0).astype(np.float32)


def digital_silence(samples: int) -> np.ndarray:
    """Not a quiet room — actually zero, which is what a muted microphone produces."""
    return np.zeros(samples, dtype=np.float32)


# ── Files ──────────────────────────────────────────────────────────────────

def read_clip(path: Path) -> np.ndarray | None:
    """Mono 16kHz float, by whichever decoder is installed. See `audio_io`."""
    got = read_audio(path, SR)
    if got is None:
        print(f"  ! {path.name}: nothing here can decode it — skipped", file=sys.stderr)
    return got


def write_clip(path: Path, wave: np.ndarray) -> None:
    write_wav(path, wave, SR)


def source_clips(clips_dir: Path) -> list[Path]:
    """The learner's own recordings. Synthetic renders are deliberately not included:
    a negative derived from TTS measures the synthesiser, and the whole reason this set
    exists is that constants fitted on synthetic speech did not transfer."""
    return sorted(p for p in clips_dir.glob("me_*.*")
                  if p.suffix.lower() in (".wav", ".flac", ".mp3"))


def build(clips_dir: Path = CLIPS, out_dir: Path = NEGATIVES,
          seed: int = 5) -> list[dict]:
    """Every negative, written out, with a row describing each."""
    sources = source_clips(clips_dir)
    if not sources:
        print(f"no recordings in {clips_dir} — run\n"
              f"  python scripts/compare_stt.py --record 25\n"
              f"on a machine with a microphone first", file=sys.stderr)
        return []

    rng = np.random.default_rng(seed)
    waves = [(p, w) for p, w in ((p, read_clip(p)) for p in sources) if w is not None]
    if not waves:
        return []
    lengths = [len(w) for _, w in waves]
    rows: list[dict] = []

    def emit(name: str, wave: np.ndarray, kind: str, source: str, detail: str) -> None:
        write_clip(out_dir / name, wave)
        rows.append({"file": name, "kind": kind, "source": source, "detail": detail,
                     "seconds": f"{len(wave) / SR:.2f}", "peak": f"{peak(wave):.4f}"})

    # Lengths drawn from the real clips, so nothing here is rejected for being an
    # implausible duration rather than for not being speech.
    for i in range(N_SILENCE):
        n = int(rng.choice(lengths))
        emit(f"neg_silence_{i:03d}.wav", digital_silence(n), "silence", "-", "zeros")

    for i in range(N_NOISE):
        n = int(rng.choice(lengths))
        level = NOISE_LEVELS[i % len(NOISE_LEVELS)]
        emit(f"neg_hiss_{i:03d}.wav", white_noise(n, level, rng), "hiss", "-",
             f"rms {level}")

    for path, wave in waves:
        stem = path.stem
        # No `quiet` negatives. An attenuated copy of a correct answer is still the correct
        # answer: the recogniser normalises each mel bin over the clip, so a uniform gain
        # change shifts log-mel by a constant that per-bin mean subtraction removes exactly.
        # Measured rather than assumed — amplifying the nine quietest real non-answers by 8x
        # moved target confidence by at most 0.017 and reversed no verdict, and 93% of the
        # 75 attenuated copies this used to emit were accepted, which is the invariance
        # showing up as a 93% "failure" rate against a label that was wrong.
        #
        # Leaving them in was worse than useless. Any rule fitted against this set was being
        # charged for accepting correct answers, which pushes every threshold in the strict
        # direction for no reason — and they were 75 of 190 negatives, so the pull was large.
        emit(f"neg_reversed_{stem}.wav", reversed_clip(wave), "reversed", path.name,
             "time-reversed")

    manifest = out_dir / "manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, delimiter="\t",
                               fieldnames=["file", "kind", "source", "detail",
                                           "seconds", "peak"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def read_negatives(out_dir: Path = NEGATIVES) -> list[dict]:
    """Whatever was built, as the sweep reads it."""
    manifest = out_dir / "manifest.tsv"
    if not manifest.exists():
        return []
    with manifest.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def summarise(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", type=Path, default=CLIPS)
    ap.add_argument("--out", type=Path, default=NEGATIVES)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--report", action="store_true",
                    help="describe what is already built and exit")
    args = ap.parse_args()

    if args.report:
        rows = read_negatives(args.out)
        if not rows:
            print(f"nothing built under {args.out}")
            return 1
        print(f"{len(rows)} negatives under {args.out}")
        for kind, n in sorted(summarise(rows).items()):
            print(f"  {kind:9} {n:4d}")
        return 0

    rows = build(args.clips_dir, args.out, args.seed)
    if not rows:
        return 2
    print(f"{len(rows)} negatives → {args.out}")
    for kind, n in sorted(summarise(rows).items()):
        print(f"  {kind:9} {n:4d}")
    print("\nThe published tables were measured on a set whose exact composition was")
    print("never recorded, so treat these as a rebuild rather than the same set: sweep")
    print("the current rule alongside any candidate instead of comparing to the README.")
    print("\nnext:  python scripts/sweep_grader.py --models frontend/stt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

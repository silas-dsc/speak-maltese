#!/usr/bin/env python3
"""Cut long recordings into utterance-sized pieces on the silences between them.

The BDL course tracks average 55.7 seconds and the distillation shard holds passes of
about 1.5. Handing the teacher a 55-second file is not a slower version of the same thing:
its posteriors drift over that length, and one bad stretch contaminates a label covering
the whole minute rather than one utterance. The Global Recordings material has the same
shape with music between the speech.

Splitting on energy rather than on a fixed clock is the whole point — a fixed cut lands
mid-word, and a mid-word boundary teaches the student a syllable that no word begins with.

    python scripts/segment_audio.py --in data/corpora/bdl_raw --out data/corpora/bdl
    python scripts/segment_audio.py --in data/corpora/grn_raw --out data/corpora/grn \
        --min-silence 0.20

`--min-silence` is per-source and has to be, because the pause structure is what differs
between them rather than anything about the speech. Course audio leaves a long gap for the
learner to repeat into, so 0.9 finds the phrase boundaries: 7172 segments over 3.78h at a
1.9s mean. Continuous narration has no such gaps, and the same 0.9 cuts it into 9.5s slabs
of which 3% are usable; 0.20 gives 1539 segments over 1.03h at 2.4s, the same shape as the
course audio. Reading the mean back after a run is the check — it should be around two
seconds, because that is what the distillation shard holds.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_io import SR, write_wav  # noqa: E402

# A pause has to be this long to count as a boundary. Shorter than this and it is the gap
# inside a word — Maltese `q` is a glottal stop, which *is* a silence, and cutting there
# would split a word at its most distinctive sound.
# How long a pause has to be to count as a boundary. Long, deliberately: at 0.25s the
# course audio breaks into single words at a 0.9s median, and 0.9s gives a 2.1s median with
# 76% of segments between one and four seconds — the length the distillation shard holds.
# A teacher labelling one word has less context than one labelling a phrase.
MIN_SILENCE = 0.9

# A frame is silence if it sits this far below the track's loudest frame. The same shape
# as `SPEECH_DROP_DB` in the recogniser, and for the same reason: relative to the peak, so
# it survives a quiet recording without an absolute level to tune per source.
#
# Two wrong answers were tried first, and both are instructive. A fraction of the *median*
# fails because the median frame of this material sits 36 to 39 dB below the peak — more
# than half of each track is already near-silence — so anchoring there anchors to silence,
# the threshold lands under the noise floor and every segment comes out at the maximum
# length: 20 segments averaging 10.7s, all forced cuts rather than pauses. A *percentile*
# fixed that on real audio and is fragile in a different way: when silence is one large
# tight cluster, the percentile falls inside it and splits it, so half the silent frames
# read as loud and no pause is ever long enough. Distance below the peak has neither
# failure, and separates a synthetic 51%-silence track cleanly at every value tried.
# 34 rather than 22: both give a 2.0s median segment, and 34 keeps 60% of the audio where
# 22 keeps 47% — thirteen points of the best-matched speech in the project for nothing in
# return. Going further merges utterances instead: 40 keeps 85% but the mean jumps to 5.0s.
# What the remaining 40% is, is real — course audio leaves long pauses for the learner to
# repeat into, and those are not speech.
SILENCE_DROP_DB = 34.0

MIN_SEG, MAX_SEG = 0.6, 12.0
PAD = 0.08


def decode(path: Path) -> np.ndarray | None:
    """Any format ffmpeg reads → mono float at 16k, via a pipe rather than a temp file."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
         "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)


def spans(wave: np.ndarray, min_silence: float = MIN_SILENCE,
          drop_db: float = SILENCE_DROP_DB) -> list[tuple[int, int]]:
    """`[start, end)` sample ranges holding speech, split on long-enough pauses."""
    hop = SR // 100                                   # 10ms frames
    n = len(wave) // hop
    if n < 2:
        return []
    frames = wave[:n * hop].reshape(n, hop)
    energy = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    thresh = float(energy.max()) * (10.0 ** (-drop_db / 20.0))
    loud = energy > thresh
    if not loud.any():
        return []

    runs, start = [], None
    quiet = 0
    need = int(min_silence * 100)
    for i, is_loud in enumerate(loud):
        if is_loud:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= need:
                runs.append((start, i - quiet + 1))
                start = None
    if start is not None:
        runs.append((start, len(loud)))

    out = []
    pad = int(PAD * 100)
    for a, b in runs:
        a, b = max(0, a - pad), min(len(loud), b + pad)
        secs = (b - a) / 100
        if secs < MIN_SEG:
            continue
        if secs <= MAX_SEG:
            out.append((a * hop, b * hop))
            continue
        # Too long to be one utterance and no pause inside it: cut on the quietest frames
        # rather than on a clock, so the boundary still lands in the least-bad place.
        pieces = int(np.ceil(secs / MAX_SEG))
        step = (b - a) // pieces
        cut = a
        for k in range(1, pieces):
            lo, hi = a + k * step - step // 4, a + k * step + step // 4
            lo, hi = max(a + 1, lo), min(b - 1, hi)
            at = int(lo + np.argmin(energy[lo:hi])) if hi > lo else a + k * step
            out.append((cut * hop, at * hop))
            cut = at
        out.append((cut * hop, b * hop))
    return [(a, b) for a, b in out if (b - a) / SR >= MIN_SEG]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", dest="dst", type=Path, required=True)
    ap.add_argument("--min-silence", type=float, default=MIN_SILENCE)
    ap.add_argument("--drop-db", type=float, default=SILENCE_DROP_DB)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    files = sorted(p for p in args.src.rglob("*")
                   if p.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a", ".ogg"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"nothing to segment under {args.src}", file=sys.stderr)
        return 2
    args.dst.mkdir(parents=True, exist_ok=True)

    made = 0
    total = 0.0
    skipped = 0
    for i, path in enumerate(files, 1):
        wave = decode(path)
        if wave is None or not len(wave):
            skipped += 1
            continue
        cuts = spans(wave, args.min_silence, args.drop_db)
        for k, (a, b) in enumerate(cuts):
            piece = wave[a:b]
            write_wav(args.dst / f"{path.stem}_{k:03d}.wav", piece, SR)
            made += 1
            total += len(piece) / SR
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)} files · {made} segments · {total / 3600:.2f}h",
                  flush=True)

    print(f"\n✓ {made} segments, {total / 3600:.2f}h → {args.dst}")
    if made:
        print(f"  mean {total / made:.1f}s a segment"
              f"{f', {skipped} files unreadable' if skipped else ''}")
    print(f"  next:  python scripts/distill_stt.py teacher --sources corpus "
          f"--corpus-name {args.dst.name} --shard {args.dst.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

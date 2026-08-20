#!/usr/bin/env python3
"""Read and write clips without adding a dependency to do it.

The tooling here already has two ways to decode audio, depending on what happened to be
installed for something else: `soundfile`, which is a small wheel, and
`faster_whisper.audio.decode_audio`, which is already in `requirements.txt` because the
recogniser fallback needs it. Either is enough, and asking for a third would mean anyone
running the sweep has to install something before the negatives will build.

Writing needs neither. A 16-bit mono WAV is a header and some samples, and `wave` is in
the standard library — so nothing that only writes clips depends on anything at all.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 16000


def read_audio(path: Path, sample_rate: int = SR) -> np.ndarray | None:
    """Mono float32 at `sample_rate`, or `None` if nothing here can read it.

    Tried in order of cost: `soundfile` if it is present, then `faster_whisper`, which is
    heavier but already required. Both are imported lazily so this module imports on
    numpy alone — `tests/test_scripts.py` imports every script, and CI installs neither."""
    try:
        import soundfile as sf

        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        wave_out = data.mean(axis=1)
    except Exception:                        # noqa: BLE001 — fall through to the other
        try:
            from faster_whisper.audio import decode_audio

            wave_out = np.asarray(decode_audio(str(path), sampling_rate=sample_rate),
                                  dtype=np.float32)
            rate = sample_rate
        except Exception:                    # noqa: BLE001 — genuinely unreadable
            return None

    if rate != sample_rate:
        n = int(round(len(wave_out) * sample_rate / rate))
        if n < 2:
            return None
        wave_out = np.interp(np.arange(n) * rate / sample_rate,
                             np.arange(len(wave_out)), wave_out).astype(np.float32)
    return np.ascontiguousarray(wave_out, dtype=np.float32)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = SR) -> None:
    """16-bit mono PCM, via the standard library.

    Clipped before conversion rather than allowed to wrap: an overflowing float becomes a
    loud click at the opposite polarity, which in a set of negatives would read as a
    transient the grader ought to reject rather than as the level it was asked for."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())


def peak(samples: np.ndarray) -> float:
    return float(np.max(np.abs(samples))) if len(samples) else 0.0

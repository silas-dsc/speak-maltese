#!/usr/bin/env python3
"""Render the deck in voices that mispronounce Maltese, as training data.

The recogniser's remaining weakness is not size and not the decoder — it is that every
Maltese model available is trained on native speakers, and the app is for learners. On 25
recordings in a learner's voice the 315M teacher marks 20% of correct answers correct.
There is no L2 Maltese corpus to fix that with; I looked.

What there is: 322 `edge-tts` voices, of which two are Maltese. An English voice reading
`Mingħajr zokkor` mispronounces it roughly the way an English speaker does — `għ` attempted
as a consonant, `q` as a k, no gemination — and the text label is known regardless. That is
the domain gap, synthesised, in unlimited quantity.

It is a proxy and should be read as one. A grapheme-to-phoneme system reading foreign
orthography produces one specific wrong pronunciation per voice, not the distribution of
human learner errors. Closer than native Maltese; not the real thing.

**Why this bypasses `tts.synthesize`.** That function refuses any voice outside `mt-MT-` and
falls back to the app's own, deliberately: the app must never teach Maltese in an Italian
accent, which is a point the README makes explicitly. The guard is right and stays. This
calls the edge backend directly so the app path is untouched — and because the first probe
of this idea came back with six byte-identical files, which would have been three hours
spent training on duplicated native audio.

    python scripts/render_accents.py --lines 750
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backend.config import DATA_DIR  # noqa: E402

OUT = DATA_DIR / "accents"

# Spread across phonologies a Maltese learner plausibly arrives with: English (the app's
# own reporter is Australian), Italian for the Romance layer, Arabic for the Semitic one,
# then French, German and Spanish for a wider spread of vowel systems.
VOICES = [
    "en-GB-SoniaNeural", "en-US-GuyNeural", "en-AU-NatashaNeural",
    "it-IT-DiegoNeural", "ar-EG-SalmaNeural", "fr-FR-DeniseNeural",
    "de-DE-KatjaNeural", "es-ES-AlvaroNeural",
]


def clip_path(line: str, voice: str) -> Path:
    key = hashlib.sha256(f"accent|{voice}|{line}".encode()).hexdigest()[:32]
    return OUT / f"{key}.mp3"


def corpus(limit: int | None) -> list[str]:
    """Deck lines, minus the evaluation sentences, spread rather than truncated."""
    from distill_stt import corpus as deck_corpus

    lines = deck_corpus()
    if limit and limit < len(lines):
        step = max(1, len(lines) // limit)
        lines = lines[::step][:limit]
    return lines


async def main() -> int:
    from backend import tts

    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=750, help="deck lines per voice")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--voices", default=",".join(VOICES))
    args = ap.parse_args()

    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    lines = corpus(args.lines)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(ln, v) for v in voices for ln in lines if not clip_path(ln, v).exists()]
    total = len(lines) * len(voices)
    print(f"{len(lines)} lines x {len(voices)} voices = {total} clips "
          f"({total - len(jobs)} already rendered, {len(jobs)} to do)")
    if not jobs:
        return 0

    sem = asyncio.Semaphore(args.concurrency)
    done = failed = 0
    lock = asyncio.Lock()
    t0 = time.time()

    async def render(line: str, voice: str) -> None:
        nonlocal done, failed
        async with sem:
            try:
                # The backend directly, not synthesize(): see the note at the top.
                audio = await tts._edge(line, voice, 1.0)
                ok = bool(audio)
                if ok:
                    clip_path(line, voice).write_bytes(audio)
            except Exception as exc:  # noqa: BLE001 — one voice failing is not fatal
                ok, err = False, exc
        async with lock:
            if ok:
                done += 1
            else:
                failed += 1
                if failed <= 5:
                    print(f"  ! {voice} {line[:40]!r}: {err}")
            n = done + failed
            if n % 200 == 0 or n == len(jobs):
                rate = n / max(0.001, time.time() - t0)
                left = (len(jobs) - n) / max(0.001, rate)
                print(f"  {n:>5}/{len(jobs)}  {done} ok  {failed} failed  "
                      f"{rate:.1f}/s  ~{left/60:.0f}min left", flush=True)

    await asyncio.gather(*(render(ln, v) for ln, v in jobs))
    size = sum(p.stat().st_size for p in OUT.glob("*.mp3"))
    print(f"\n{len(list(OUT.glob('*.mp3')))} clips, {size/1e6:.0f}MB in {OUT}")
    return 1 if failed > len(jobs) // 10 else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

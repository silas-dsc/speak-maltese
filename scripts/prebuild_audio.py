#!/usr/bin/env python3
"""Render every Maltese line the app can speak to MP3, ahead of time.

Everything the app says is authored — scripted dialogue lines, deck words and
phrases, and the example sentences — so none of it needs to be synthesised while
someone is waiting. Rendering it up front means:

* a drill turn waits on nothing but speech recognition,
* review cards play the instant they appear,
* and the app works with no network at all once the cache is warm.

The cache is keyed on (provider, voice, rate, text), so changing voice or speaking
rate is a different render — pass `--voice` / `--rate` to build those too. Already
cached lines are skipped, so re-running after editing a dialogue is cheap.

    python scripts/prebuild_audio.py               # everything, default voice
    python scripts/prebuild_audio.py --what drills # just the scripted dialogues
    python scripts/prebuild_audio.py --voice mt-MT-JosephNeural
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import curriculum, dialogue, tts  # noqa: E402
from backend.config import AUDIO_CACHE, CFG  # noqa: E402


def lines_for(what: str) -> list[str]:
    out: list[str] = []
    if what in ("all", "drills"):
        out += dialogue.every_line()
    if what in ("all", "deck"):
        vocab = curriculum._read_tsv(curriculum.VOCAB_TSV)
        phrases = curriculum._read_tsv(curriculum.PHRASES_TSV)
        out += [r["mt"] for r in vocab + phrases]
        out += [r["ex_mt"] for r in vocab if r.get("ex_mt")]
    if what in ("all", "deck"):
        for s in curriculum.load_scenarios():
            if s.get("opener_mt"):
                out.append(s["opener_mt"])
    seen, uniq = set(), []
    for line in out:
        line = (line or "").strip()
        if line and line not in seen:
            seen.add(line)
            uniq.append(line)
    return uniq


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["all", "drills", "deck"], default="all")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--rate", type=float, default=0.95)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    lines = lines_for(args.what)
    voice = args.voice or CFG.azure_voice
    print(f"Rendering {len(lines)} lines · voice={voice} · rate={args.rate}")
    print(f"Cache: {AUDIO_CACHE}\n")

    done = failed = 0
    t0 = time.time()
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def render(i: int, line: str) -> None:
        nonlocal done, failed
        async with sem:
            try:
                await tts.synthesize(line, voice, args.rate)
                ok = True
            except Exception as exc:  # noqa: BLE001
                ok = False
                err = exc
        async with lock:
            if ok:
                done += 1
            else:
                failed += 1
                print(f"  ✗ {line[:52]!r}: {err}")
            if (done + failed) % 25 == 0 or (done + failed) == len(lines):
                print(f"  {done + failed:>4}/{len(lines)}  ({done} ok, {failed} failed)",
                      flush=True)

    await asyncio.gather(*(render(i, ln) for i, ln in enumerate(lines)))

    size = sum(f.stat().st_size for f in AUDIO_CACHE.glob("*") if f.is_file())
    print(f"\n✓ {done} rendered, {failed} failed in {time.time() - t0:.0f}s")
    print(f"  cache now {len(list(AUDIO_CACHE.glob('*')))} files, {size / 1e6:.1f} MB")
    if failed:
        print("  Re-run to retry the failures — cached lines are skipped.")
    return 1 if failed and not done else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

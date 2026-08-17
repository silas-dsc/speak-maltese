#!/usr/bin/env python3
"""Assemble the whole app as static files, for GitHub Pages or any dumb host.

Nothing here is a port — the ports already happened (frontend/text.js,
dialogue.js, srs.js, schedule.js, store.js, nanostt.js, each with a parity test
against the Python it came from). This just pre-renders what the server used to
answer and lays it out so the same `index.html` works with no backend:

    /api/bootstrap        → api/bootstrap.json
    /api/deck             → api/deck.json
    /api/drill/dialogues  → api/dialogues.json  (the whole script, not just the list)
    /api/grammar          → api/grammar.json
    /api/tts?text=…       → audio/<hash>.mp3 + audio/index.json

The audio is the interesting one. The server synthesised on demand and cached by
`sha256(provider|voice|rate|text)`; a static build cannot hash on the fly cheaply
in every browser, so the mapping is written out as a manifest and the player looks
the line up instead. Only lines the app can actually say are included — the deck,
its examples and every scripted line — which is what `prebuild_audio.py` already
renders.

Speech recognition runs on the device (nanostt.js), from a 2.1MB model copied out of
`frontend/stt/`. The build writes `capabilities.stt = []` because there is no server
behind it, not because nothing can listen.

    python scripts/build_static.py                    # → dist/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import curriculum, dialogue  # noqa: E402
from backend.config import AUDIO_CACHE, CFG, FRONTEND_DIR  # noqa: E402

DIST = ROOT / "dist"

# Everything the client loads by name. Anything missing here is a blank page.
SHELL = ("index.html", "style.css", "app.js", "srs.js", "store.js", "schedule.js",
         "splash.js", "nanostt.js", "text.js", "dialogue.js", "session.js", "capture.js", "sw.js",
         "manifest.webmanifest")

# The on-device recogniser. 2.1MB, which is why it can live in the repository and be
# served from the same origin as the page — GitHub refuses single files over 100MB, and
# the 200MB model this replaces had to be fetched from the Hugging Face Hub.
STT_DIR = "stt"


def cache_key(text: str, voice: str, rate: float, provider: str = "edge") -> str:
    return hashlib.sha256(f"{provider}|{voice}|{rate}|{text}".encode()).hexdigest()[:32]


def copy_shell() -> None:
    for name in SHELL:
        src = FRONTEND_DIR / name
        if src.exists():
            shutil.copy(src, DIST / name)
        else:
            print(f"  ! missing {name}")
    if (FRONTEND_DIR / "img").is_dir():
        shutil.copytree(FRONTEND_DIR / "img", DIST / "img", dirs_exist_ok=True)
    # Without this the page loads, the deck seeds, and the mic silently never works.
    if (FRONTEND_DIR / STT_DIR).is_dir():
        shutil.copytree(FRONTEND_DIR / STT_DIR, DIST / STT_DIR, dirs_exist_ok=True)
    else:
        print(f"  ! missing {STT_DIR}/ — the build will have no recogniser")


def write_api(stt_base: str) -> dict:
    api = DIST / "api"
    api.mkdir(parents=True, exist_ok=True)

    cards = curriculum.deck_rows()
    (api / "deck.json").write_text(
        json.dumps({"cards": cards}, ensure_ascii=False), encoding="utf-8")

    (api / "dialogues.json").write_text(
        json.dumps(dialogue.load(), ensure_ascii=False), encoding="utf-8")

    (api / "grammar.json").write_text(
        json.dumps({"markdown": curriculum.grammar_notes()}, ensure_ascii=False),
        encoding="utf-8")

    (api / "bootstrap.json").write_text(json.dumps({
        "capabilities": {
            # No server of its own: synthesis is pre-rendered, and recognition happens
            # on the device unless `stt_base` names somewhere else. `stt` describes what
            # a *server* could do for this build, which is nothing.
            "tts": ["prerendered"],
            "stt": ["remote"] if stt_base else [],
        },
        "static": True,
        # Empty — the normal case — means the device does it, from the 2.1MB model in
        # `stt/`. Setting it hands recognition to a deployment of the FastAPI app
        # instead, which no longer buys a phone anything.
        "stt_base": stt_base.rstrip("/"),
        "defaults": {
            "voice": CFG.azure_voice,
            "rate": 0.95,
            "daily_new": CFG.daily_new_limit,
            "daily_review": CFG.daily_review_limit,
            "target_retention": CFG.target_retention,
        },
    }, ensure_ascii=False), encoding="utf-8")
    return {"cards": len(cards), "scenes": len(dialogue.all_dialogues())}


def wanted_lines() -> list[str]:
    """Every Maltese line the app can utter, exactly as prebuild_audio.py sees it."""
    out = list(dialogue.every_line())
    vocab = curriculum._read_tsv(curriculum.VOCAB_TSV)
    phrases = curriculum._read_tsv(curriculum.PHRASES_TSV)
    out += [r["mt"] for r in vocab + phrases]
    out += [r["ex_mt"] for r in vocab if r.get("ex_mt")]
    seen, uniq = set(), []
    for line in out:
        line = (line or "").strip()
        if line and line not in seen:
            seen.add(line)
            uniq.append(line)
    return uniq


def copy_audio(rate: float) -> dict:
    """Copy the rendered MP3s and write the text→file manifest."""
    out_dir = DIST / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    index, missing, total = {}, [], 0

    for line in wanted_lines():
        name = f"{cache_key(line, CFG.azure_voice, rate)}.mp3"
        src = AUDIO_CACHE / name
        if not src.exists():
            missing.append(line)
            continue
        shutil.copy(src, out_dir / name)
        index[line] = name
        total += src.stat().st_size

    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8")
    return {"files": len(index), "missing": len(missing), "bytes": total,
            "examples": missing[:3]}


def stamp_shell_version() -> str:
    """Give the service worker a cache name that changes when the build does.

    The worker serves the shell stale-while-revalidate, so without this a deploy
    arrived in pieces: a page open across one ran the new `api/dialogues.json`
    against the previous `app.js`. Naming the cache after the build makes the old
    one unreachable, so the next load refetches the shell whole.

    Hashed over what the shell cache actually holds — the JS, the CSS, the HTML,
    the manifests — and not over the commit, so a deploy that changes nothing the
    device would notice leaves its cache alone. The MP3s are excluded: they live in
    a cache of their own that no build should ever invalidate.
    """
    sw = DIST / "sw.js"
    if not sw.exists():
        return ""
    digest = hashlib.sha256()
    for path in sorted(p for p in DIST.rglob("*") if p.is_file()):
        if path == sw or path.suffix == ".mp3":
            continue
        digest.update(path.relative_to(DIST).as_posix().encode())
        digest.update(path.read_bytes())
    build = digest.hexdigest()[:12]
    src = sw.read_text(encoding="utf-8")
    stamped = src.replace("const BUILD = 'dev'", f"const BUILD = '{build}'", 1)
    if stamped == src:
        raise SystemExit("sw.js has no `const BUILD = 'dev'` line to stamp")
    sw.write_text(stamped, encoding="utf-8")
    return build


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rate", type=float, default=0.95)
    ap.add_argument("--stt-base", default="",
                    help="base URL of a deployment that can transcribe, for a build "
                         "that would rather centralise it. Empty — the default — keeps "
                         "recognition on the device, which is now 2.1MB and shipped "
                         "with the page.")
    args = ap.parse_args()

    global DIST
    DIST = args.out or ROOT / "dist"
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    copy_shell()
    counts = write_api(args.stt_base)
    audio = copy_audio(args.rate)

    # Pages serves _-prefixed paths oddly and runs Jekyll unless told not to.
    (DIST / ".nojekyll").write_text("", encoding="utf-8")

    # Last, so the hash covers everything above it.
    build = stamp_shell_version()

    print(f"{DIST.relative_to(ROOT) if DIST.is_relative_to(ROOT) else DIST}")
    print(f"  {counts['cards']} cards · {counts['scenes']} scenes")
    print(f"  shell cache: {build or 'unstamped'}")
    print(f"  {audio['files']} audio files · {audio['bytes'] / 1e6:.0f} MB")
    if audio["missing"]:
        print(f"  ! {audio['missing']} lines have no audio — run "
              f"scripts/prebuild_audio.py first")
        for line in audio["examples"]:
            print(f"      {line}")
    print(f"  recogniser: {args.stt_base or 'on-device only'}")
    return 1 if audio["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

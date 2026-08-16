#!/usr/bin/env python3
"""Assemble the whole app as static files, for GitHub Pages or any dumb host.

Nothing here is a port — the ports already happened (frontend/text.js,
dialogue.js, srs.js, schedule.js, store.js, localstt.js, each with a parity test
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

Speech recognition has no static equivalent: it runs on the device (localstt.js)
or not at all. The build writes `capabilities.stt = []` so the UI says so honestly
rather than offering a mic that posts into the void.

    python scripts/build_static.py                    # → dist/
    python scripts/build_static.py --models-base https://huggingface.co/…/resolve/main/
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
         "splash.js", "localstt.js", "text.js", "dialogue.js", "sw.js",
         "manifest.webmanifest")


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


def write_api(models_base: str) -> dict:
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
            # No server: synthesis is pre-rendered, recognition is on-device only.
            "tts": ["prerendered"],
            "stt": [],
        },
        "static": True,
        "models_base": models_base,
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rate", type=float, default=0.95)
    ap.add_argument("--models-base", default="/models/",
                    help="where localstt.js should fetch the ONNX weights from; "
                         "for Pages this has to be off-site, since GitHub refuses "
                         "files over 100MB")
    args = ap.parse_args()

    global DIST
    DIST = args.out or ROOT / "dist"
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    copy_shell()
    counts = write_api(args.models_base)
    audio = copy_audio(args.rate)

    # Pages serves _-prefixed paths oddly and runs Jekyll unless told not to.
    (DIST / ".nojekyll").write_text("", encoding="utf-8")

    print(f"{DIST.relative_to(ROOT) if DIST.is_relative_to(ROOT) else DIST}")
    print(f"  {counts['cards']} cards · {counts['scenes']} scenes")
    print(f"  {audio['files']} audio files · {audio['bytes'] / 1e6:.0f} MB")
    if audio["missing"]:
        print(f"  ! {audio['missing']} lines have no audio — run "
              f"scripts/prebuild_audio.py first")
        for line in audio["examples"]:
            print(f"      {line}")
    print(f"  models base: {args.models_base}")
    return 1 if audio["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

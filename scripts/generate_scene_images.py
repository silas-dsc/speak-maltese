#!/usr/bin/env python3
"""Generate one illustration per scripted scene, locally.

Locally, because the alternative is stock photography with licences to audit — and
this repo is public. Images are produced by whatever image model Ollama is serving
(`x/flux2-klein` by default), so nothing is copied from the web and the provenance
is recorded in `frontend/img/CREDITS.md` next to the files.

They are deliberately small: a scene header is decoration, not content, so each is
downscaled and written as WebP at a few tens of kilobytes rather than a megabyte of
PNG. The whole set should stay under ~1 MB so cloning stays cheap.

    python scripts/generate_scene_images.py           # only missing ones
    python scripts/generate_scene_images.py --force   # redo everything
    python scripts/generate_scene_images.py --only cafe
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from backend import dialogue  # noqa: E402
from backend.config import ROOT  # noqa: E402

OUT_DIR = ROOT / "frontend" / "img"
OLLAMA = "http://localhost:11434/api/generate"
MODEL = "x/flux2-klein:latest"

# One shared style so fifteen images read as one set rather than fifteen moods.
STYLE = ("flat vector editorial illustration, warm Mediterranean palette of "
         "limestone cream, terracotta and sea blue, soft flat shapes, minimal "
         "detail, no text, no lettering, no people's faces in close-up, "
         "calm and friendly, wide banner composition")

SCENES = {
    "greet": "two neighbours greeting each other on a Maltese street with a limestone balcony above",
    "cafe": "a small Maltese cafe terrace, coffee cups on a round table, pastizzi on a plate",
    "directions": "a narrow Valletta street corner with a signpost and steps, someone pointing the way",
    "stuck": "an open phrasebook and a speech bubble with a question mark, on a wooden table",
    "market": "a Maltese open air market stall piled with tomatoes, oranges and grapes",
    "weather": "a sunny Maltese bay with fishing boats, bright sky and a few clouds",
    "restaurant": "a seaside Maltese restaurant table set for two, fresh fish and a carafe of wine",
    "doctor": "a calm clinic waiting room with a plant and a window, gentle light",
    "smalltalk": "two friends chatting on a bench overlooking the Grand Harbour",
    "work": "a bright small office desk with a laptop, notebook and coffee, window onto rooftops",
    "family": "a family kitchen table with several chairs and a bowl of fruit, homely",
    "where": "a living room with keys on a table, books on a shelf, a chair and a window",
    "likes": "a plate of Maltese food seen from above, fish, bread and salad, appetising",
    "plans": "a calendar and cinema tickets on a table beside a phone, evening light",
    "meeting": "a doorway with a welcome mat, warm light spilling out, someone waving hello",
    "home": "a Maltese townhouse interior, open door onto a bright kitchen, tiled floor",
    "routine": "an alarm clock on a bedside table beside a window at sunrise, calm",
    "shop": "a small Maltese grocery shop counter with shelves of milk, eggs and bread",
    "colours": "two folded shirts on a table, one blue and one green, tape measure beside them",
    "people": "a framed family photograph on a sideboard beside a vase of flowers",
    "feelings": "a quiet balcony with one chair and a cup of tea, soft evening light",
    "town": "a Maltese village square with a church dome, a school and a signpost",
    "phone": "an old telephone on a hall table with a notepad and pencil beside it",
    "pharmacy": "a small pharmacy counter with a green cross sign, shelves of boxes behind",
    "bus": "a bright yellow Maltese bus at a stop on a coastal road, blue sky",
    "lost": "a folded paper map and a street sign at a confusing junction of narrow lanes",
    "learning": "an open notebook with handwriting, a pencil and a dictionary on a desk",
    "booking": "a restaurant table laid with a reserved card, candle and folded napkins",
    "clothes": "a small clothes shop rail with shirts in green, yellow and black, shoes below",
    "keys": "a doorstep with a bunch of keys lying on the stone step, warm evening light",
    "hobbies": "a football, a guitar and swimming goggles on a sunlit terrace floor",
    "relatives": "an older couple sitting together on a balcony, seen from behind, warm light",
    "jobs": "a hospital, a school and an office building along a Maltese street",
    "opinions": "two coffee cups on a table with a newspaper between them, mid-conversation",
    "outing": "a rocky Maltese swimming spot with clear turquoise water and a towel on the rock",
}


def generate(prompt: str, timeout: int = 900) -> bytes:
    r = httpx.post(OLLAMA, timeout=timeout,
                   json={"model": MODEL, "prompt": prompt, "stream": False})
    r.raise_for_status()
    data = r.json()
    b64 = data.get("image") or (data.get("images") or [None])[0]
    if not b64:
        raise RuntimeError(f"no image in response (keys: {list(data)})")
    return base64.b64decode(b64)


def to_webp(png: bytes, width: int = 640) -> bytes:
    from PIL import Image

    im = Image.open(io.BytesIO(png)).convert("RGB")
    # Crop to a 2:1 banner from the centre, then downscale.
    w, h = im.size
    target_h = min(h, w // 2)
    top = (h - target_h) // 2
    im = im.crop((0, top, w, top + target_h))
    im = im.resize((width, width // 2), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=78, method=6)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = [d["id"] for d in dialogue.all_dialogues()]
    if args.only:
        ids = [i for i in ids if i == args.only]

    made, skipped, failed = [], 0, []
    for i, sid in enumerate(ids, 1):
        dest = OUT_DIR / f"scene-{sid}.webp"
        if dest.exists() and not args.force:
            skipped += 1
            continue
        subject = SCENES.get(sid)
        if not subject:
            print(f"  ? no prompt for scene {sid!r} — skipping")
            continue
        prompt = f"{subject}. {STYLE}"
        t0 = time.time()
        print(f"  [{i}/{len(ids)}] {sid} …", end="", flush=True)
        try:
            webp = to_webp(generate(prompt), args.width)
            dest.write_bytes(webp)
            made.append(sid)
            print(f" {len(webp)/1024:.0f} KB in {time.time()-t0:.0f}s")
        except Exception as exc:  # noqa: BLE001
            failed.append(sid)
            print(f" FAILED: {exc}")

    total = sum(f.stat().st_size for f in OUT_DIR.glob("*.webp"))
    (OUT_DIR / "CREDITS.md").write_text(
        "# Scene illustrations\n\n"
        f"Generated locally with `{MODEL}` via Ollama by "
        "`scripts/generate_scene_images.py`. No third-party imagery is used, so there\n"
        "is nothing here copied from the web.\n\n"
        "Check the model's own licence before using these commercially — that is a\n"
        "property of the generator, not of this repository.\n\n"
        "Regenerate with `python scripts/generate_scene_images.py --force`.\n",
        encoding="utf-8")

    print(f"\n✓ {len(made)} generated, {skipped} already present, {len(failed)} failed")
    print(f"  {len(list(OUT_DIR.glob('*.webp')))} images · {total/1024:.0f} KB total")
    if failed:
        print("  failed: " + ", ".join(failed))
    return 1 if failed and not made else 0


if __name__ == "__main__":
    raise SystemExit(main())

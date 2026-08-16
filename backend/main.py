"""FastAPI app: static frontend + JSON/audio API."""

from __future__ import annotations

import logging
import threading

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import CFG, FRONTEND_DIR
from . import curriculum, dialogue, srs, stt, text, tts

logging.basicConfig(
    level=logging.DEBUG if CFG.debug else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("speak-maltese")

app = FastAPI(title="Speak Maltese", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    log.info("deck: %d cards", len(curriculum.deck_rows()))
    log.info("TTS: %s | STT: %s", tts.available() or "none", stt.available() or "none")

    # Warm the local recogniser off the request path: loading it lazily puts the
    # whole model load — several seconds — onto the first thing the learner says.
    # `preload` warms whichever local models are in the chain, so the test is
    # whether *any* of them is active. Testing for faster_whisper alone meant a
    # wav2vec2-only install, which is the fast default, never warmed anything.
    if {"wav2vec2", "faster_whisper"} & set(stt.available()):
        threading.Thread(target=stt.preload, name="stt-preload", daemon=True).start()

    # Scripted dialogue speaks from a fixed, finite script, so all of it can be
    # synthesised up front. Once warm, a drill turn waits on nothing but the
    # recogniser — which is the whole point of that mode.
    threading.Thread(target=_prewarm_dialogue_audio,
                     name="tts-prewarm", daemon=True).start()


def _prewarm_dialogue_audio() -> None:
    import asyncio

    lines = dialogue.every_line()
    voice, rate = CFG.azure_voice, 0.95

    async def run() -> int:
        done = 0
        for line in lines:
            try:
                await tts.synthesize(line, voice, rate)
                done += 1
            except Exception:  # noqa: BLE001 — a cold cache is not fatal
                pass
        return done

    try:
        done = asyncio.run(run())
        log.info("dialogue audio cached: %d/%d lines", done, len(lines))
    except Exception:  # noqa: BLE001
        log.exception("dialogue audio prewarm failed")


# ── Bootstrap ──────────────────────────────────────────────────────────────

@app.get("/api/bootstrap")
def bootstrap() -> dict:
    """What the client cannot work out for itself: which speech providers exist,
    and the server's idea of the defaults. Progress is not here — it lives in the
    browser, so this response is identical for every visitor and safe to cache."""
    caps = CFG.capabilities()
    caps["tts"] = tts.available()
    caps["stt"] = stt.available()
    return {
        "capabilities": caps,
        "defaults": {
            "voice": CFG.azure_voice,
            "rate": 0.95,
            "daily_new": CFG.daily_new_limit,
            "daily_review": CFG.daily_review_limit,
            "target_retention": CFG.target_retention,
        },
    }


@app.get("/api/health")
def health() -> dict:
    """Is the recogniser actually loaded, or is the first utterance going to pay
    for it? On a cold container the model load is tens of seconds, and a learner
    holding the mic button with nothing happening reads as broken. The client
    polls this and holds the door shut until it says ready."""
    return {
        "ready": stt.is_warm() or not stt.needs_warmup(),
        "warming": stt.needs_warmup() and not stt.is_warm(),
        "stt": stt.available(),
        "tts": tts.available(),
    }


@app.get("/api/deck")
def deck() -> dict:
    """The whole curated deck, for the client to seed its own database from.

    Cards are content and ship with the app; the schedule built on top of them is
    the learner's and never leaves their device."""
    return {"cards": curriculum.deck_rows()}


@app.get("/api/grammar")
def grammar() -> dict:
    return {"markdown": curriculum.grammar_notes()}


# ── Scripted conversation (no model in the loop) ───────────────────────────

@app.get("/api/drill/dialogues")
def drill_list() -> dict:
    return {"dialogues": [
        {k: d[k] for k in ("id", "name", "name_en", "level")}
        for d in dialogue.all_dialogues()
    ]}


@app.post("/api/drill/start")
def drill_start(payload: dict = Body(...)) -> dict:
    node = dialogue.start(payload.get("dialogue") or "")
    if not node:
        raise HTTPException(404, "unknown dialogue")
    return node


@app.post("/api/drill/answer")
def drill_answer(payload: dict = Body(...)) -> dict:
    result = dialogue.evaluate(
        payload.get("dialogue") or "",
        payload.get("node") or "",
        payload.get("said") or "",
        int(payload.get("attempts") or 0),
    )
    if result.get("error"):
        raise HTTPException(404, result["error"])
    # A phrase produced correctly under time pressure is exactly what should be
    # scheduled — but the schedule lives in the browser now, so the response just
    # carries `matched_mt`/`matched_en` and the client does the bookkeeping.
    return result


# ── Speech ─────────────────────────────────────────────────────────────────

@app.post("/api/stt")
async def speech_to_text(audio: UploadFile = File(...),
                         target: str = Form(default="")) -> dict:
    raw = await audio.read()
    try:
        result = await stt.transcribe(raw, audio.content_type or "audio/webm")
    except stt.STTError as exc:
        raise HTTPException(503, str(exc)) from exc
    if target:
        result["assessment"] = _assess(result["text"], target)
    return result


@app.get("/api/tts")
async def text_to_speech(text_: str = Query(..., alias="text"),
                         rate: float = Query(1.0, ge=0.5, le=1.5),
                         voice: str | None = None) -> Response:
    try:
        audio, mime = await tts.synthesize(text_, voice, rate)
    except tts.TTSError as exc:
        raise HTTPException(503, str(exc)) from exc
    return Response(
        content=audio, media_type=mime,
        headers={"Cache-Control": "public, max-age=604800"},
    )


@app.post("/api/attempt")
def attempt(payload: dict = Body(...)) -> dict:
    """Grade a spoken or typed attempt against a target sentence."""
    said = (payload.get("said") or "").strip()
    target = (payload.get("target") or "").strip()
    if not target:
        raise HTTPException(400, "target is required")
    return _assess(said, target)


def _assess(said: str, target: str) -> dict:
    s = text.score(said, target)
    return {
        "said": text.normalise(said),
        "target": text.normalise(target),
        "score": s,
        "grade": _auto_grade(s),
        "verdict": ("perfect" if s >= 0.95 else "close" if s >= 0.75
                    else "partial" if s >= 0.5 else "off"),
        "diff": text.diff_words(said, target),
    }


def _auto_grade(score: float) -> int:
    if score >= 0.95:
        return srs.EASY
    if score >= 0.78:
        return srs.GOOD
    if score >= 0.55:
        return srs.HARD
    return srs.AGAIN


# ── Spaced repetition ──────────────────────────────────────────────────────
#
# There is no server-side scheduler any more. `/api/queue`, `/api/review`,
# `/api/stats` and `/api/cards` all read and wrote one SQLite file, which meant a
# single shared review history for everyone who opened the app and nothing left
# after a container restart. The FSRS implementation now runs in the browser
# against IndexedDB (frontend/srs.js, schedule.js, store.js); the server keeps
# only what it alone can do — the decks, the scripted dialogue, recognition and
# synthesis. Grading a single utterance is still here because it is pure text
# comparison over the Maltese rules in backend/text.py, and it is stateless.


# ── Static frontend ────────────────────────────────────────────────────────

@app.middleware("http")
async def cache_headers(request: Request, call_next):
    """Keep the shell fresh and the assets cheap.

    StaticFiles alone let a browser hold on to an old app.js or style.css
    indefinitely — which showed up as new CSS simply not applying after an edit.
    The shell now always revalidates (cheap: a 304 when unchanged), while content
    that is addressed by name and never edited in place is cached hard.
    """
    response = await call_next(request)
    path = request.url.path
    if path in ("/", "/index.html", "/style.css", "/sw.js", "/manifest.webmanifest") \
            or (path.endswith(".js") and path.count("/") == 1):
        response.headers["Cache-Control"] = "no-cache"
    elif path.startswith("/img/"):
        response.headers.setdefault("Cache-Control", "public, max-age=604800")
    return response



@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.exception_handler(404)
async def spa_fallback(request: Request, exc):  # noqa: ANN001
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "not found"}, status_code=404)
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")

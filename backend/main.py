"""FastAPI app: static frontend + JSON/audio API."""

from __future__ import annotations

import hashlib
import logging
import threading

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import CFG, FRONTEND_DIR
from . import curriculum, dialogue, games, phonetics, srs, stt, text, tts

logging.basicConfig(
    level=logging.DEBUG if CFG.debug else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("speak-maltese")

app = FastAPI(title="Speak Maltese", version="1.0.0")

# The static build sends utterances here to be recognised, from another origin, and
# a browser refuses that unless this says otherwise. Named origins rather than `*`:
# these endpoints hold no learner state, but a wildcard on a POST that accepts audio
# invites every page on the internet to use this as a free transcription service.
_ORIGINS = [o.strip() for o in CFG.cors_origins.split(",") if o.strip()]
if _ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ORIGINS,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["content-type"],
    )


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


# ── Mini-games ─────────────────────────────────────────────────────────────

@app.get("/api/games")
def games_payload() -> dict:
    """The mini-games. Derived here rather than in the browser for the same reason the
    static build derives them: one implementation, and the client only marks."""
    return games.all_games()


# ── Scripted conversation (no model in the loop) ───────────────────────────

@app.get("/api/drill/dialogues")
def drill_list() -> dict:
    # `steps` is how many turns the scene is, which the client shows as "2/4" in the
    # conversation's header — the one bit of "how far in am I" that survived the
    # scene picker being replaced by its own screen.
    return {"dialogues": [
        {**{k: d[k] for k in ("id", "name", "name_en", "level")},
         "steps": len(d.get("nodes") or {})}
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
    """Grade one spoken or typed attempt.

    The orthographic score alone was the only grading path in the app that ignored
    phonetics, and it was costing the learner marks for the recogniser's habits
    rather than their own. Speech recognisers put word boundaries where they like:
    `Birra kiesħa` comes back as `birrakisħa`, `Ninsa kollox` as `nin sa kollox`,
    `ix-xarabank` as `ix-xara bank`. `text.score` is word-aligned, so a join or a
    split wrecks it, while the phonetic key ignores spacing entirely — which is why
    the scripted-dialogue matcher has always blended the two and this did not.

    Same blend as `dialogue._best_match`, so the two halves of the app now agree
    about what counts as saying it right. Measured over 334 real transcripts from
    the 4-bit recogniser: answers graded Good or better go from 96.1% to 100%, and
    the highest score an unrelated sentence reaches actually falls, from 0.879 to
    0.824.
    """
    phon = phonetics.similarity(said, target, soft=True)
    s = max(phon, 0.6 * phon + 0.4 * text.score(said, target))
    s = round(s, 4)
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



# The ONNX weights for on-device recognition, when they have been built. They are
# large and immutable, so they cache hard and are simply absent if nobody ran
# scripts/export_onnx_web.py — the app falls back to server-side recognition.
_MODELS_DIR = FRONTEND_DIR.parent / "web" / "models"
if _MODELS_DIR.is_dir():
    app.mount("/models", StaticFiles(directory=_MODELS_DIR), name="models")


# The files the worker precaches as the shell, which is what its cache is named
# after. Ordinary content, not a manifest to keep in step: anything the client
# imports is a *.js here, and tests/test_api.py checks the two lists agree.
_SHELL_GLOBS = ("*.js", "index.html", "style.css", "manifest.webmanifest")


@app.get("/sw.js")
def service_worker() -> Response:
    """The service worker, with its shell cache named after this build.

    scripts/build_static.py does this for the Pages build; served from the repo the
    placeholder would stay `dev` forever, so the worker's bytes would never change,
    no new worker would install, and an edited app.js would keep being served from
    the old cache — a reload late, every time. That is the bug the naming exists to
    remove, and it is worse here, where the files change while you watch.

    Hashed per request rather than at startup: in development the point is to notice
    an edit made a second ago, and it is a dozen small files.
    """
    digest = hashlib.sha256()
    files = sorted(p for g in _SHELL_GLOBS for p in FRONTEND_DIR.glob(g)
                   if p.name != "sw.js")
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    src = (FRONTEND_DIR / "sw.js").read_text(encoding="utf-8")
    src = src.replace("const BUILD = 'dev'", f"const BUILD = '{digest.hexdigest()[:12]}'", 1)
    return Response(src, media_type="text/javascript",
                    headers={"Cache-Control": "no-cache"})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.exception_handler(404)
async def spa_fallback(request: Request, exc):  # noqa: ANN001
    if request.url.path.startswith("/api/"):
        # Keep the reason an endpoint gave. This handler catches deliberate 404s as
        # well as missing routes, and flattening them to "not found" threw away the
        # only thing the client could act on: the drill knows to abandon a saved
        # conversation whose node has gone, and could not tell that from a typo in
        # a URL.
        return JSONResponse({"detail": getattr(exc, "detail", None) or "not found"},
                            status_code=404)
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")

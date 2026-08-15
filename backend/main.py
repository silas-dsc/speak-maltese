"""FastAPI app: static frontend + JSON/audio API."""

from __future__ import annotations

import logging
import threading
import uuid

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import CFG, FRONTEND_DIR
from . import curriculum, db, dialogue, srs, stt, text, tts, tutor

logging.basicConfig(
    level=logging.DEBUG if CFG.debug else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("speak-maltese")

app = FastAPI(title="Speak Maltese", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    db.init()
    seeded = curriculum.seed()
    log.info("decks loaded: %s", seeded)
    log.info("TTS: %s | STT: %s | tutor: %s",
             tts.available() or "none", stt.available() or "none",
             CFG.capabilities()["tutor_provider"] or "none")

    # Warm the local recogniser off the request path. The Maltese fine-tune is a
    # whisper-large, so loading it lazily would put several seconds onto the first
    # thing the learner says. In a thread so the UI is up immediately.
    if "faster_whisper" in stt.available():
        threading.Thread(target=stt.preload, name="stt-preload", daemon=True).start()

    # Scripted dialogue speaks from a fixed, finite script, so all of it can be
    # synthesised up front. Once warm, a drill turn waits on nothing but the
    # recogniser — which is the whole point of that mode.
    threading.Thread(target=_prewarm_dialogue_audio,
                     name="tts-prewarm", daemon=True).start()


def _prewarm_dialogue_audio() -> None:
    import asyncio

    lines = dialogue.every_line()
    voice = db.get_setting("voice", CFG.azure_voice)
    rate = float(db.get_setting("rate", 0.95))

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
    caps = CFG.capabilities()
    caps["tts"] = tts.available()
    caps["stt"] = stt.available()
    return {
        "capabilities": caps,
        "scenarios": curriculum.load_scenarios(),
        "counts": db.counts(),
        "profile": {k: v for k, v in curriculum.learner_profile().items()
                    if k in ("level", "learned_count")},
        "settings": {
            "voice": db.get_setting("voice", CFG.azure_voice),
            "rate": db.get_setting("rate", 0.95),
            "show_english": db.get_setting("show_english", True),
            "autoplay": db.get_setting("autoplay", True),
            "daily_new": CFG.daily_new_limit,
        },
        "sessions": db.sessions(10),
    }


@app.post("/api/settings")
def save_settings(payload: dict = Body(...)) -> dict:
    for key in ("voice", "rate", "show_english", "autoplay"):
        if key in payload:
            db.set_setting(key, payload[key])
    return {"ok": True}


@app.get("/api/grammar")
def grammar() -> dict:
    return {"markdown": curriculum.grammar_notes()}


# ── Conversation ───────────────────────────────────────────────────────────

@app.post("/api/session")
def new_session(payload: dict = Body(default={})) -> dict:
    sid = uuid.uuid4().hex[:12]
    scenario_id = payload.get("scenario")
    scenarios = {s["id"]: s for s in curriculum.load_scenarios()}
    scenario = scenarios.get(scenario_id or "", scenarios.get("intro", {}))
    opener = {
        "reply_mt": scenario.get("opener_mt", "Bonġu! Kif int?"),
        "reply_en": scenario.get("opener_en", "Good morning! How are you?"),
        "correction": {"needed": False},
        "gloss": [], "new_vocab": [], "difficulty_signal": "ok",
    }
    db.add_turn(sid, scenario_id, "tutor", opener["reply_mt"], opener["reply_en"], opener)
    return {"session_id": sid, "scenario": scenario, "opener": opener}


@app.post("/api/chat")
async def chat(payload: dict = Body(...)) -> dict:
    user_text = (payload.get("text") or "").strip()
    if not user_text:
        raise HTTPException(400, "text is required")
    session_id = payload.get("session_id") or uuid.uuid4().hex[:12]
    scenario_id = payload.get("scenario")
    try:
        data = await tutor.respond(user_text, session_id, scenario_id)
    except tutor.TutorUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("tutor failed")
        raise HTTPException(502, f"tutor error: {exc}") from exc
    data["session_id"] = session_id
    data["counts"] = db.counts()
    return data


@app.get("/api/history")
def get_history(session_id: str = Query(...), limit: int = 60) -> dict:
    return {"turns": db.history(session_id, limit)}


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
    )
    if result.get("error"):
        raise HTTPException(404, result["error"])
    # A scripted turn still feeds the spaced-repetition system: a phrase you
    # produced correctly under time pressure is exactly what should be scheduled.
    if result["verdict"] == "correct" and result.get("matched_mt"):
        _schedule_from_drill(result["matched_mt"], result.get("matched_en") or "")
    return result


def _schedule_from_drill(mt: str, en: str) -> None:
    try:
        ids = curriculum.register_new_vocab([{"mt": mt, "en": en}], "drill")
        for cid in ids:
            st = db.state_of(cid)
            if st.state == "new":
                db.save_state(cid, srs.review(st, srs.GOOD, CFG.target_retention))
                db.log_review(cid, srs.GOOD, "produce", None, None, None)
    except Exception:  # noqa: BLE001 — never let bookkeeping break a turn
        log.exception("could not schedule drill phrase")


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

@app.get("/api/queue")
def queue(limit: int = 20, topics: str | None = None,
          include_new: bool = True, max_tier: int | None = None) -> dict:
    topic_list = [t for t in (topics or "").split(",") if t] or None
    cards = curriculum.build_queue(limit, topic_list, include_new, max_tier)
    for c in cards:
        st = db.state_of(c["id"])
        c["intervals"] = {str(k): v for k, v in srs.preview(st, CFG.target_retention).items()}
    return {"cards": cards, "counts": db.counts()}


@app.post("/api/review")
def review(payload: dict = Body(...)) -> dict:
    card_id = payload.get("card_id")
    if not card_id:
        raise HTTPException(400, "card_id is required")
    card = db.get_card(card_id)
    if not card:
        raise HTTPException(404, "unknown card")

    said = payload.get("said")
    mode = payload.get("mode", "recognise")
    assessment = None
    grade = payload.get("grade")

    if said and mode in ("produce", "repeat"):
        assessment = _assess(said, card["mt"])
        if grade is None:
            grade = assessment["grade"]
    if grade is None:
        raise HTTPException(400, "grade or said is required")

    st = db.state_of(card_id)
    st = srs.review(st, int(grade), CFG.target_retention)
    db.save_state(card_id, st)
    db.log_review(card_id, int(grade), mode,
                  assessment["score"] if assessment else None,
                  said, payload.get("elapsed_ms"))
    return {
        "card_id": card_id,
        "state": st.state,
        "due": db.iso(st.due),
        "stability_days": round(st.stability, 2),
        "difficulty": round(st.difficulty, 2),
        "assessment": assessment,
        "counts": db.counts(),
    }


@app.get("/api/stats")
def stats() -> dict:
    return db.stats()


@app.get("/api/cards")
def cards(q: str | None = None, topic: str | None = None, limit: int = 60) -> dict:
    sql = """SELECT c.*, s.state, s.due, s.stability, s.lapses
             FROM cards c JOIN card_state s ON s.card_id=c.id WHERE 1=1"""
    args: list = []
    if q:
        sql += " AND (c.mt LIKE ? OR c.en LIKE ?)"
        args += [f"%{q}%", f"%{q}%"]
    if topic:
        sql += " AND c.topic = ?"
        args.append(topic)
    sql += " ORDER BY c.tier, c.id LIMIT ?"
    args.append(limit)
    with db.db() as conn:
        rows = [dict(r) for r in conn.execute(sql, args).fetchall()]
    return {"cards": rows}


@app.post("/api/cards/suspend")
def suspend(payload: dict = Body(...)) -> dict:
    with db.db() as conn:
        conn.execute("UPDATE cards SET suspended=? WHERE id=?",
                     (1 if payload.get("suspended", True) else 0, payload["card_id"]))
    return {"ok": True}


@app.post("/api/schedule-correction")
def schedule_correction(payload: dict = Body(...)) -> dict:
    """Turn a correction from the conversation into a scheduled card."""
    mt = (payload.get("mt") or "").strip()
    en = (payload.get("en") or "").strip()
    if not mt:
        raise HTTPException(400, "mt is required")
    ids = curriculum.register_new_vocab(
        [{"mt": mt, "en": en or mt, "note": payload.get("why")}],
        payload.get("scenario"),
    )
    return {"card_ids": ids}


# ── Static frontend ────────────────────────────────────────────────────────

@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.exception_handler(404)
async def spa_fallback(request: Request, exc):  # noqa: ANN001
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "not found"}, status_code=404)
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")

"""SQLite persistence: cards, review log, conversation history, settings."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from .config import CFG, DB_PATH
from . import srs

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,          -- vocab | phrase | sentence
    mt            TEXT NOT NULL,
    en            TEXT NOT NULL,
    pos           TEXT,
    tier          INTEGER DEFAULT 3,
    topic         TEXT,
    example_mt    TEXT,
    example_en    TEXT,
    note          TEXT,
    literal       TEXT,
    source        TEXT DEFAULT 'core',    -- core | tutor | import
    suspended     INTEGER DEFAULT 0,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS card_state (
    card_id       TEXT PRIMARY KEY REFERENCES cards(id) ON DELETE CASCADE,
    stability     REAL DEFAULT 0,
    difficulty    REAL DEFAULT 0,
    reps          INTEGER DEFAULT 0,
    lapses        INTEGER DEFAULT 0,
    state         TEXT DEFAULT 'new',
    step          INTEGER DEFAULT 0,
    due           TEXT,
    last_review   TEXT,
    -- production (speaking) is tracked separately from recognition
    prod_reps     INTEGER DEFAULT 0,
    prod_correct  INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_state_due ON card_state(due);

CREATE TABLE IF NOT EXISTS reviews (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id       TEXT NOT NULL,
    grade         INTEGER NOT NULL,
    mode          TEXT NOT NULL,          -- recognise | produce | listen | conversation
    score         REAL,
    said          TEXT,
    elapsed_ms    INTEGER,
    reviewed_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reviews_at ON reviews(reviewed_at);

CREATE TABLE IF NOT EXISTS turns (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    scenario      TEXT,
    role          TEXT NOT NULL,          -- user | tutor
    mt            TEXT,
    en            TEXT,
    payload       TEXT,                   -- full JSON of a tutor turn
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, id);

CREATE TABLE IF NOT EXISTS errors (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    kind          TEXT NOT NULL,          -- grammar | vocab | spelling | word-order | pronunciation
    learner       TEXT,
    corrected     TEXT,
    why           TEXT,
    card_id       TEXT,
    resolved      INTEGER DEFAULT 0,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key           TEXT PRIMARY KEY,
    value         TEXT
);
"""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, detect_types=0, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db():
    conn = _conn()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    with db() as conn:
        conn.executescript(SCHEMA)


def iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def parse(s: str | None) -> datetime | None:
    if not s:
        return None
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ── Cards ──────────────────────────────────────────────────────────────────

def upsert_cards(rows: Iterable[dict]) -> int:
    n = 0
    stamp = iso(srs.now())
    with db() as conn:
        for r in rows:
            conn.execute(
                """INSERT INTO cards (id, kind, mt, en, pos, tier, topic, example_mt,
                                      example_en, note, literal, source, created_at)
                   VALUES (:id,:kind,:mt,:en,:pos,:tier,:topic,:example_mt,
                           :example_en,:note,:literal,:source,:created_at)
                   ON CONFLICT(id) DO UPDATE SET
                     mt=excluded.mt, en=excluded.en, pos=excluded.pos, tier=excluded.tier,
                     topic=excluded.topic, example_mt=excluded.example_mt,
                     example_en=excluded.example_en, note=excluded.note,
                     literal=excluded.literal""",
                {
                    "id": r["id"], "kind": r.get("kind", "vocab"), "mt": r["mt"],
                    "en": r["en"], "pos": r.get("pos"), "tier": int(r.get("tier") or 3),
                    "topic": r.get("topic"), "example_mt": r.get("example_mt"),
                    "example_en": r.get("example_en"), "note": r.get("note"),
                    "literal": r.get("literal"), "source": r.get("source", "core"),
                    "created_at": stamp,
                },
            )
            conn.execute(
                "INSERT OR IGNORE INTO card_state (card_id) VALUES (?)", (r["id"],)
            )
            n += 1
    return n


def get_card(card_id: str) -> dict | None:
    with db() as conn:
        row = conn.execute(
            """SELECT c.*, s.stability, s.difficulty, s.reps, s.lapses, s.state,
                      s.step, s.due, s.last_review, s.prod_reps, s.prod_correct
               FROM cards c JOIN card_state s ON s.card_id = c.id
               WHERE c.id = ?""", (card_id,)
        ).fetchone()
    return dict(row) if row else None


def state_of(card_id: str) -> srs.CardState:
    with db() as conn:
        row = conn.execute("SELECT * FROM card_state WHERE card_id=?", (card_id,)).fetchone()
    if not row:
        return srs.CardState()
    return srs.CardState(
        stability=row["stability"] or 0.0,
        difficulty=row["difficulty"] or 0.0,
        reps=row["reps"] or 0,
        lapses=row["lapses"] or 0,
        state=row["state"] or "new",
        step=row["step"] or 0,
        due=parse(row["due"]),
        last_review=parse(row["last_review"]),
    )


def save_state(card_id: str, st: srs.CardState) -> None:
    with db() as conn:
        conn.execute(
            """UPDATE card_state SET stability=?, difficulty=?, reps=?, lapses=?,
                                     state=?, step=?, due=?, last_review=?
               WHERE card_id=?""",
            (st.stability, st.difficulty, st.reps, st.lapses, st.state, st.step,
             iso(st.due), iso(st.last_review), card_id),
        )


def log_review(card_id: str, grade: int, mode: str, score: float | None,
               said: str | None, elapsed_ms: int | None) -> None:
    with db() as conn:
        conn.execute(
            """INSERT INTO reviews (card_id, grade, mode, score, said, elapsed_ms, reviewed_at)
               VALUES (?,?,?,?,?,?,?)""",
            (card_id, grade, mode, score, said, elapsed_ms, iso(srs.now())),
        )
        if mode == "produce":
            conn.execute(
                """UPDATE card_state SET prod_reps = prod_reps + 1,
                       prod_correct = prod_correct + ?
                   WHERE card_id = ?""",
                (1 if grade >= srs.GOOD else 0, card_id),
            )


def due_cards(limit: int, topics: list[str] | None = None) -> list[dict]:
    now_iso = iso(srs.now())
    q = """SELECT c.*, s.stability, s.difficulty, s.reps, s.lapses, s.state, s.due,
                  s.prod_reps, s.prod_correct
           FROM cards c JOIN card_state s ON s.card_id = c.id
           WHERE c.suspended = 0 AND s.state != 'new' AND s.due <= ?"""
    args: list[Any] = [now_iso]
    if topics:
        q += " AND c.topic IN (%s)" % ",".join("?" * len(topics))
        args += topics
    q += " ORDER BY s.due ASC LIMIT ?"
    args.append(limit)
    with db() as conn:
        return [dict(r) for r in conn.execute(q, args).fetchall()]


def new_cards(limit: int, topics: list[str] | None = None,
              max_tier: int | None = None) -> list[dict]:
    """New cards, easiest tier first — frequency-ordered introduction."""
    q = """SELECT c.*, s.state FROM cards c JOIN card_state s ON s.card_id = c.id
           WHERE c.suspended = 0 AND s.state = 'new'"""
    args: list[Any] = []
    if topics:
        q += " AND c.topic IN (%s)" % ",".join("?" * len(topics))
        args += topics
    if max_tier:
        q += " AND c.tier <= ?"
        args.append(max_tier)
    # phrases interleave with vocab so you always leave with something sayable
    q += " ORDER BY c.tier ASC, CASE c.kind WHEN 'phrase' THEN 0 ELSE 1 END, c.id ASC LIMIT ?"
    args.append(limit)
    with db() as conn:
        return [dict(r) for r in conn.execute(q, args).fetchall()]


def known_cards(limit: int = 400) -> list[dict]:
    """Cards the learner has actually retained — the i+1 vocabulary pool."""
    with db() as conn:
        return [dict(r) for r in conn.execute(
            """SELECT c.mt, c.en, c.kind, c.topic, s.stability
               FROM cards c JOIN card_state s ON s.card_id = c.id
               WHERE s.state = 'review' AND s.stability >= 3
               ORDER BY s.stability DESC LIMIT ?""", (limit,)
        ).fetchall()]


def counts() -> dict:
    now_iso = iso(srs.now())
    day_ago = iso(srs.now() - timedelta(days=1))
    with db() as conn:
        c = conn.execute(
            """SELECT
                 (SELECT COUNT(*) FROM card_state WHERE state='new') AS new,
                 (SELECT COUNT(*) FROM card_state WHERE state!='new' AND due<=?) AS due,
                 (SELECT COUNT(*) FROM card_state WHERE state='review') AS learned,
                 (SELECT COUNT(*) FROM card_state WHERE state='review' AND stability>=21) AS solid,
                 (SELECT COUNT(*) FROM cards) AS total,
                 (SELECT COUNT(*) FROM reviews WHERE reviewed_at>=?) AS today
            """, (now_iso, day_ago)
        ).fetchone()
    return dict(c)


def stats() -> dict:
    with db() as conn:
        base = counts()
        rows = conn.execute(
            """SELECT date(reviewed_at) AS d, COUNT(*) AS n,
                      AVG(CASE WHEN grade>=3 THEN 1.0 ELSE 0.0 END) AS retention
               FROM reviews GROUP BY d ORDER BY d DESC LIMIT 60"""
        ).fetchall()
        topics = conn.execute(
            """SELECT c.topic,
                      COUNT(*) AS total,
                      SUM(CASE WHEN s.state='review' THEN 1 ELSE 0 END) AS learned
               FROM cards c JOIN card_state s ON s.card_id=c.id
               WHERE c.topic IS NOT NULL
               GROUP BY c.topic ORDER BY total DESC"""
        ).fetchall()
        speaking = conn.execute(
            """SELECT COALESCE(SUM(prod_reps),0) AS attempts,
                      COALESCE(SUM(prod_correct),0) AS correct FROM card_state"""
        ).fetchone()
        weak = conn.execute(
            """SELECT c.id, c.mt, c.en, s.lapses, s.difficulty
               FROM cards c JOIN card_state s ON s.card_id=c.id
               WHERE s.lapses > 0 ORDER BY s.lapses DESC, s.difficulty DESC LIMIT 15"""
        ).fetchall()
        err = conn.execute(
            """SELECT kind, COUNT(*) AS n FROM errors
               GROUP BY kind ORDER BY n DESC"""
        ).fetchall()
    return {
        **base,
        "history": [dict(r) for r in rows][::-1],
        "topics": [dict(r) for r in topics],
        "speaking": dict(speaking),
        "weak": [dict(r) for r in weak],
        "error_kinds": [dict(r) for r in err],
        "streak": _streak([dict(r) for r in rows]),
    }


def _streak(history: list[dict]) -> int:
    """`history` is newest-first as returned by the query above.

    `reviewed_at` is stored in UTC and `date(reviewed_at)` extracts the UTC day, so
    the walk has to start from the UTC day too. Starting from the *local* date broke
    the streak for anyone far enough east: reviewing at 09:00 in Sydney is 23:00 UTC
    the day before, so the first lookup missed and the streak read zero."""
    days = {r["d"] for r in history}
    n, cur = 0, srs.now().date()
    while cur.isoformat() in days:
        n += 1
        cur = cur - timedelta(days=1)
    return n


# ── Conversation ───────────────────────────────────────────────────────────

def add_turn(session_id: str, scenario: str | None, role: str, mt: str | None,
             en: str | None, payload: dict | None = None) -> None:
    with db() as conn:
        conn.execute(
            """INSERT INTO turns (session_id, scenario, role, mt, en, payload, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (session_id, scenario, role, mt, en,
             json.dumps(payload, ensure_ascii=False) if payload else None, iso(srs.now())),
        )


def history(session_id: str, limit: int = 40) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM turns WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    out = []
    for r in reversed(rows):
        d = dict(r)
        if d.get("payload"):
            try:
                d["payload"] = json.loads(d["payload"])
            except json.JSONDecodeError:
                d["payload"] = None
        out.append(d)
    return out


def sessions(limit: int = 20) -> list[dict]:
    with db() as conn:
        return [dict(r) for r in conn.execute(
            """SELECT session_id, scenario, MIN(created_at) AS started,
                      MAX(created_at) AS last, COUNT(*) AS turns
               FROM turns GROUP BY session_id ORDER BY last DESC LIMIT ?""", (limit,)
        ).fetchall()]


def log_error(kind: str, learner: str, corrected: str, why: str,
              card_id: str | None = None) -> None:
    with db() as conn:
        conn.execute(
            """INSERT INTO errors (kind, learner, corrected, why, card_id, created_at)
               VALUES (?,?,?,?,?,?)""",
            (kind, learner, corrected, why, card_id, iso(srs.now())),
        )


def recent_errors(limit: int = 12) -> list[dict]:
    with db() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM errors ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()]


# ── Settings ───────────────────────────────────────────────────────────────

def get_setting(key: str, default: Any = None) -> Any:
    with db() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    if not row:
        return default
    try:
        return json.loads(row["value"])
    except (json.JSONDecodeError, TypeError):
        return default


def set_setting(key: str, value: Any) -> None:
    with db() as conn:
        conn.execute(
            "INSERT INTO settings (key,value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, ensure_ascii=False)),
        )


def reviews_today() -> int:
    return counts()["today"]

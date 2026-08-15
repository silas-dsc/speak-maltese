"""Deck loading and lesson planning.

The pedagogy encoded here:

* **Frequency ordering** — tier 1 items first, so early effort buys the most coverage.
* **Chunks before words** — phrases interleave with vocab (`db.new_cards`), because
  fluent speech is largely prefabricated sequences, not words assembled from scratch.
* **Production over recognition** — scripted turns and review both make the learner
  *say* things, which is the harder and more transferable direction.
* **Interleaving** — review queues mix topics and card kinds rather than blocking.
* **Retrieval practice** — every session includes production (speaking), not just
  recognition; the two are tracked separately per card.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from .config import DATA_DIR, CFG
from . import db

VOCAB_TSV = DATA_DIR / "core_vocab.tsv"
PHRASES_TSV = DATA_DIR / "phrases.tsv"
GRAMMAR_MD = DATA_DIR / "grammar_notes.md"
IMPORT_TSV = DATA_DIR / "frequency_import.tsv"  # optional, produced by scripts/


def _read_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines, delimiter="\t")
    for r in reader:
        if not r.get("id") or not r.get("mt"):
            continue
        rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    return rows


def grammar_notes() -> str:
    return GRAMMAR_MD.read_text(encoding="utf-8") if GRAMMAR_MD.exists() else ""


def seed() -> dict:
    """Load the decks into SQLite. Idempotent — safe to run on every boot."""
    vocab = [
        {
            "id": r["id"], "kind": "vocab", "mt": r["mt"], "en": r["en"],
            "pos": r.get("pos"), "tier": r.get("tier") or 3, "topic": r.get("topic"),
            "example_mt": r.get("ex_mt") or None, "example_en": r.get("ex_en") or None,
            "note": r.get("note") or None, "source": "core",
        }
        for r in _read_tsv(VOCAB_TSV)
    ]
    phrases = [
        {
            "id": r["id"], "kind": "phrase", "mt": r["mt"], "en": r["en"],
            "pos": "phrase", "tier": r.get("tier") or 2, "topic": r.get("topic"),
            "literal": r.get("literal") or None, "note": r.get("note") or None,
            "source": "core",
        }
        for r in _read_tsv(PHRASES_TSV)
    ]
    imported = [
        {
            "id": r["id"], "kind": "vocab", "mt": r["mt"], "en": r["en"],
            "pos": r.get("pos"), "tier": r.get("tier") or 4, "topic": r.get("topic") or "imported",
            "note": r.get("note") or None, "source": "import",
        }
        for r in _read_tsv(IMPORT_TSV)
    ]
    n = db.upsert_cards(vocab + phrases + imported)
    return {"vocab": len(vocab), "phrases": len(phrases), "imported": len(imported), "total": n}


# ── Session planning ───────────────────────────────────────────────────────

def build_queue(limit: int = 20, topics: list[str] | None = None,
                include_new: bool = True, max_tier: int | None = None) -> list[dict]:
    """Interleaved queue of due reviews plus a capped trickle of new material."""
    done_today = db.reviews_today()
    review_budget = max(0, CFG.daily_review_limit - done_today)
    due = db.due_cards(min(limit, review_budget), topics)

    new: list[dict] = []
    if include_new and len(due) < limit:
        new_budget = max(0, CFG.daily_new_limit - _new_introduced_today())
        new = db.new_cards(min(limit - len(due), new_budget), topics, max_tier)

    queue = _interleave(due, new)
    for c in queue:
        c["mode"] = _pick_mode(c)
    return queue[:limit]


def _new_introduced_today() -> int:
    with db.db() as conn:
        row = conn.execute(
            """SELECT COUNT(DISTINCT card_id) AS n FROM reviews
               WHERE reviewed_at >= datetime('now','-1 day')
                 AND card_id IN (SELECT card_id FROM card_state WHERE reps <= 2)"""
        ).fetchone()
    return row["n"] if row else 0


def _interleave(due: list[dict], new: list[dict]) -> list[dict]:
    """Spread new cards through the review queue rather than front-loading them —
    interleaving beats blocking for long-term retention."""
    if not new:
        return due
    if not due:
        return new
    out: list[dict] = []
    gap = max(1, len(due) // (len(new) + 1))
    ni = 0
    for i, card in enumerate(due):
        out.append(card)
        if ni < len(new) and (i + 1) % gap == 0:
            out.append(new[ni])
            ni += 1
    out.extend(new[ni:])
    return out


def _pick_mode(card: dict) -> str:
    """Which retrieval direction to test.

    New cards are *shown* first (listen). After that we favour production, which is
    the harder and more transferable direction, but keep recognition in rotation.
    """
    state = card.get("state", "new")
    if state == "new":
        return "listen"
    prod = card.get("prod_reps") or 0
    reps = card.get("reps") or 0
    if reps < 2:
        return "recognise"
    if prod * 2 < reps:
        return "produce"
    return random.choice(["produce", "recognise", "listen"])


def learner_profile() -> dict:
    """Compact snapshot of what the learner knows, for the UI and the queue."""
    known = db.known_cards(300)
    c = db.counts()
    errors = db.recent_errors(10)
    level = _estimate_level(c["learned"])
    return {
        "level": level,
        "learned_count": c["learned"],
        "known_words": [k["mt"] for k in known if k["kind"] == "vocab"][:180],
        "known_phrases": [k["mt"] for k in known if k["kind"] == "phrase"][:60],
        "recent_errors": [
            {"kind": e["kind"], "said": e["learner"], "correct": e["corrected"], "why": e["why"]}
            for e in errors
        ],
    }


def _estimate_level(learned: int) -> str:
    if learned < 30:
        return "A0"
    if learned < 120:
        return "A1"
    if learned < 320:
        return "A2"
    if learned < 700:
        return "B1"
    return "B2"


def register_new_vocab(items: list[dict], topic: str | None = None) -> list[str]:
    """Persist a phrase met in conversation as a new card."""
    rows, ids = [], []
    for it in items:
        mt = (it.get("mt") or "").strip()
        en = (it.get("en") or "").strip()
        if not mt or not en:
            continue
        cid = "t" + _slug(mt)
        rows.append({
            "id": cid, "kind": "phrase" if " " in mt else "vocab", "mt": mt, "en": en,
            "pos": it.get("pos"), "tier": 3, "topic": topic or "conversation",
            "note": it.get("note"), "source": "drill",
        })
        ids.append(cid)
    if rows:
        db.upsert_cards(rows)
    return ids


def _slug(s: str) -> str:
    keep = [ch.lower() if ch.isalnum() else "-" for ch in s]
    return "".join(keep).strip("-")[:48]

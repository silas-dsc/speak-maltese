"""Deck loading.

Reads the curated TSVs and hands them over as plain rows. Nothing here is stateful:
the lesson planning that used to live alongside it — queue building, interleaving,
the daily new-card budget, the learner profile — now runs in the browser against
IndexedDB, because the schedule is the learner's and a server-side one was shared
by every visitor and lost on every restart. See frontend/schedule.js.

The pedagogy is unchanged and now enforced there: frequency ordering, phrases
interleaved with words, new material spread through the due queue rather than
front-loaded, and production tracked separately from recognition.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .config import DATA_DIR

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


def deck_rows() -> list[dict]:
    """The whole curated deck as plain rows, for the client to seed from.

    Cards are content: they ship with the app and are identical for everyone. The
    schedule built on top of them is the learner's and stays in their browser, so
    this is the only half the server needs to know about."""
    return seed()["cards"]


def seed() -> dict:
    """Read the decks off disk. Idempotent and cheap — no database involved."""
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
    cards = vocab + phrases + imported
    return {"vocab": len(vocab), "phrases": len(phrases),
            "imported": len(imported), "total": len(cards), "cards": cards}

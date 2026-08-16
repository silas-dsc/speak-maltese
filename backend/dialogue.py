"""Scripted conversation: fast path, no model in the loop.

The free-conversation mode needs an LLM, and that costs 8-25s a turn locally — far
too slow to feel like talking to someone. This mode trades open-endedness for speed:

* Every line the app says is written in `data/dialogues.json`, so it is correct
  Maltese by construction and can be synthesised ahead of time and cached.
* What the learner says is matched against a list of accepted answers by *phonetic*
  similarity, which is a few microseconds of string work rather than a model call.
* Grading may be imperfect; the Maltese the learner hears never is. That is the right
  side to be wrong on — a mis-scored answer costs a retry, whereas wrong Maltese in
  the tutor's mouth teaches the wrong thing.

Latency for a turn is therefore just speech recognition plus cached audio.
"""

from __future__ import annotations

import json
import re
import random
from functools import lru_cache

from .config import DATA_DIR
from . import phonetics, text

DIALOGUES = DATA_DIR / "dialogues.json"

# Tuned against the spread in tests: real recogniser variants of a correct answer
# land at 0.88+, a different-but-plausible sentence in the 0.6s, nonsense below.
CORRECT = 0.86
CLOSE = 0.62


@lru_cache(maxsize=1)
def load() -> dict:
    if not DIALOGUES.exists():
        return {"dialogues": []}
    return json.loads(DIALOGUES.read_text(encoding="utf-8"))


def all_dialogues() -> list[dict]:
    return load().get("dialogues", [])


def get(dialogue_id: str) -> dict | None:
    return next((d for d in all_dialogues() if d["id"] == dialogue_id), None)


def node(dialogue_id: str, node_id: str) -> dict | None:
    d = get(dialogue_id)
    return (d or {}).get("nodes", {}).get(node_id)


def start(dialogue_id: str) -> dict | None:
    d = get(dialogue_id)
    if not d:
        return None
    return present(dialogue_id, d["start"])


def present(dialogue_id: str, node_id: str) -> dict | None:
    """The prompt side of a node — what the app says and what it wants back."""
    n = node(dialogue_id, node_id)
    if not n:
        return None
    return {
        "dialogue": dialogue_id,
        "node": node_id,
        "say_mt": n["say_mt"],
        "say_en": n["say_en"],
        "expect_en": n.get("expect_en", ""),
        "options": [a["en"] for a in n.get("accept", []) if not a.get("open")],
    }


# A quoted slot, and *only* that. Maltese writes the apostrophe as a letter —
# `ta' Marija`, `x'inhu`, `ma' ħabib` — so a naive quote pair would span from the
# apostrophe in `ta'` to the one in `x'` and blank out the real words between them.
# The opening quote therefore has to start a word and the closing one end it, which
# is true of `'cheese'` and never of the Maltese apostrophe.
_QUOTED = re.compile(r"(?<!\w)['‘’\"]([^'‘’\"]+)['‘’\"](?!\w)")


def _frame_score(said: str, target: str) -> float:
    """Score a sentence with a quoted foreign word in it, on its Maltese frame only.

    `Kif tgħid 'cheese' bil-Malti?` is the one pattern where the learner is meant to
    say something the recogniser cannot possibly get: a Maltese model has never been
    trained on `cheese` and writes it as whatever Maltese it sounds nearest to —
    here, `kif tgħidx xi s bil-malti`. Grading that against the full sentence
    punishes the learner for the one word they were asked to supply.

    So the quoted slot is removed from the target and the frame words are looked for
    in order. Everything outside the quotes still has to be right.
    """
    frame = _QUOTED.sub(" ", target)
    want = [k for k in (phonetics.soft_key(w) for w in text.normalise(frame).split()) if k]
    got = [phonetics.soft_key(w) for w in text.normalise(said).split()]
    if not want:
        return 0.0
    hits, i = 0, 0
    for w in want:
        # in order, allowing the unrecognisable slot to appear as any junk between
        while i < len(got):
            if phonetics.similarity(got[i], w) >= 0.8:
                hits += 1
                i += 1
                break
            i += 1
    return hits / len(want)


def _best_match(said: str, accepted: list[dict]) -> tuple[dict | None, float]:
    best, best_score = None, 0.0
    for candidate in accepted:
        if candidate.get("open"):
            continue
        # Phonetic similarity carries the decision; the orthographic score is
        # blended in so a spelling-perfect answer is never dragged down by a
        # phonetic near-miss.
        phon = phonetics.similarity(said, candidate["mt"], soft=True)
        score = max(phon, 0.6 * phon + 0.4 * text.score(said, candidate["mt"]))
        if _QUOTED.search(candidate["mt"]):
            score = max(score, _frame_score(said, candidate["mt"]))
        if score > best_score:
            best, best_score = candidate, score
    return best, round(best_score, 4)


MAX_ATTEMPTS = 2


def evaluate(dialogue_id: str, node_id: str, said: str, attempts: int = 0) -> dict:
    """Grade an utterance and return the next thing to say. No model involved."""
    n = node(dialogue_id, node_id)
    if not n:
        return {"error": "unknown node"}

    said = text.normalise(said)
    match, score = _best_match(said, n.get("accept", []))

    if n.get("free"):
        # The answer is a name, a place, a number — something personal that the app
        # has no business checking. Anything that is not silence moves on.
        verdict = "correct" if len(text.fold(said)) >= 2 else "wrong"
    elif score >= CORRECT:
        verdict = "correct"
    elif score >= CLOSE:
        verdict = "close"
    else:
        verdict = "wrong"

    # Never let someone loop on one line. After a couple of tries the target has
    # been shown and spoken twice; repeating it a third time teaches nothing, and
    # being stuck is worse than being waved through.
    moved_on = False
    if verdict != "correct" and attempts >= MAX_ATTEMPTS:
        verdict, moved_on = "correct", True

    reply = n.get(verdict) or n.get("wrong") or {}
    if moved_on:
        reply = {"mt": "Ejja nkomplu.", "en": "Let's carry on."}
    # Advance only on a correct answer; close and wrong re-ask the same node, which
    # is the prompted-repetition pattern the free-conversation mode uses too.
    next_node = n.get("next") if verdict == "correct" else node_id

    out = {
        "verdict": verdict,
        # A name, a town, an age: accepted as given and deliberately not scored.
        # The client needs to know, because showing a percentage here would be
        # reporting a match against sample answers that never applied.
        "free": bool(n.get("free")),
        "moved_on": moved_on,
        "score": score,
        "said": said,
        "matched_mt": match["mt"] if match else None,
        "matched_en": match["en"] if match else None,
        "reply_mt": reply.get("mt", ""),
        "reply_en": reply.get("en", ""),
        "advance": verdict == "correct",
        "dialogue": dialogue_id,
        "node": node_id,
    }

    # On anything short of correct, show the target so they can say it back.
    if verdict != "correct" and match:
        out["say_this_mt"] = match["mt"]
        out["say_this_en"] = match["en"]
        out["diff"] = text.diff_words(said, match["mt"])

    if next_node and verdict == "correct":
        out["next"] = present(dialogue_id, next_node)
    elif verdict == "correct":
        out["next"] = None
        out["finished"] = True
    return out


def every_line() -> list[str]:
    """Every Maltese line the app can speak — used to pre-synthesise audio so a
    scripted turn never waits on text-to-speech."""
    lines: list[str] = []
    for d in all_dialogues():
        for n in d.get("nodes", {}).values():
            lines.append(n["say_mt"])
            for key in ("correct", "close", "wrong"):
                if n.get(key, {}).get("mt"):
                    lines.append(n[key]["mt"])
            for a in n.get("accept", []):
                if not a.get("open"):
                    lines.append(a["mt"])
    seen, out = set(), []
    for line in lines:
        if line and line not in seen:
            seen.add(line)
            out.append(line)
    return out

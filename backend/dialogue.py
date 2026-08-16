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


# The variable slot in a frame: `Jien …`, `Għandi … sena`. Written as an ellipsis in
# dialogues.json; the three-dot spelling is accepted because that is what a keyboard
# produces and the difference is invisible in the file.
_SLOT = re.compile(r"…|\.\.\.")


def _keys(s: str) -> list[str]:
    """Phonetic keys, one per word, with the fused article split off its noun.

    `mill-Awstralja` is one word to `split()` and two to a recogniser, which writes it
    back either way. Splitting at the hyphen lets `Mill-…` anchor on both spellings.
    """
    words = re.split(r"[\s\-]+", text.normalise(s))
    return [k for k in (phonetics.soft_key(w) for w in words) if k]


def _same_word(a: str, b: str) -> bool:
    """Phonetically the same word, counting a pronoun's long form as its short one.

    `hu`/`huwa` and `hi`/`hija` are one word in Maltese and learners use them
    interchangeably, but they are too short for the similarity ratio to see it:
    `hu` against `huwa` scores 0.67, below the bar that `jien`/`jiena` clears at
    0.86. So a key that is the other with up to two letters added counts as the
    same word — which `sena` against `senaturi` never does.
    """
    if a == b:
        return True
    # Two-letter keys are too short for a ratio to mean anything: `in`, the tail of
    # `in-numru`, scores exactly 0.80 against `yin` — which is `jien` — and would
    # anchor the frame on a sentence containing no `jien` at all.
    if min(len(a), len(b)) >= 3 and phonetics.similarity(a, b) >= 0.8:
        return True
    short, long = sorted((a, b), key=len)
    return len(short) >= 2 and long.startswith(short) and len(long) - len(short) <= 2


def _anchor_score(said: str, frame: str) -> float:
    """Does the answer *open with* — and *close with* — the frame around its slot?

    An open question is a fixed Maltese frame with one variable in it, and the frame
    is the part the scene teaches. `Jien …` wants `Jien` before the name; `Għandi …
    sena` wants `Għandi` before the age and `sena` after it. So the words before the
    slot are looked for in order from the start of what was said, the words after it
    in order from the end backwards, and what is left between the two runs is the
    slot — the name, the town, the age, which is never judged.

    Both runs are allowed to step over words the frame does not mention: `Iva,
    għandi ħuti` and `Jien inħobb il-ħut` and `Tliet kmamar żgħar` are all the frame
    with something extra around it. What they cannot do is change places with the
    slot — `Pietru jien` has the keyword and nothing after it, and scores half.

    The slot counts as one more thing to supply, so `Jien Pietru` scores 1.0 and
    `Jien` on its own 0.5: the frame is right, nothing was said in it. An answer with
    none of the frame in it scores 0, however good the name is.
    """
    parts = _SLOT.split(frame, maxsplit=1)
    slot = len(parts) > 1
    want_pre = _keys(parts[0])
    want_post = _keys(parts[1]) if slot else []
    got = _keys(said)

    total = len(want_pre) + len(want_post) + (1 if slot else 0)
    if not total:
        return 0.0

    # Forwards from the start for what comes before the slot…
    i, pre_hits = 0, 0
    for w in want_pre:
        at = next((k for k in range(i, len(got)) if _same_word(got[k], w)), None)
        if at is None:
            break
        i, pre_hits = at + 1, pre_hits + 1
    # …and backwards from the end for what comes after it, never crossing into the
    # words the opening run already claimed.
    j, post_hits = len(got) - 1, 0
    for w in reversed(want_post):
        at = next((k for k in range(j, i - 1, -1) if _same_word(got[k], w)), None)
        if at is None:
            break
        j, post_hits = at - 1, post_hits + 1

    hits = pre_hits + post_hits
    # Whatever is left between the two runs is the answer to the question. It counts
    # only once some of the frame is there — otherwise every stray word would look
    # like a filled slot, and "hello" would score half marks for a name.
    if slot and hits and j >= i:
        hits += 1
    return hits / total


def _best_anchor(said: str, frames: list[str]) -> float:
    return round(max((_anchor_score(said, f) for f in frames), default=0.0), 4)


def _outside_frames(candidate: dict | None, frames: list[str]) -> bool:
    """Is this listed answer a deliberate step outside the frame?

    Most accepted answers on an open question are the frame with an example in the
    slot — `Għandi tletin sena.` A few use none of it: `Dak sigriet!` when asked your
    age, `Ma niftakarx!` when asked a name. Saying one of those is a real answer and
    keeps its ordinary match score, instead of being marked down against a frame it
    was never meant to use.

    None of it, not some of it. An answer that half-uses the frame is graded on the
    frame like any other, which is what keeps a sloppy frame visible instead of
    quietly excusing the answers it fits worst.
    """
    return bool(candidate) and _best_anchor(candidate["mt"], frames) == 0.0


def _frame_recall(said: str, target: str) -> float:
    """How much of an accepted answer the learner actually produced, in order.

    The fallback for the handful of `free` nodes with no slot in them — "is your
    family big or small", "how do you feel" — where the accepted answers are whole
    sentences rather than a frame with a name in it, so there is nothing to anchor.

    What fraction of the target's words appear in what was said, in order and
    phonetically? `Jien waħdi.` against "jien waħdi ħafna" gives 1.0, against
    "hello" it gives 0.
    """
    want = [k for k in (phonetics.soft_key(w) for w in text.normalise(target).split()) if k]
    got = [phonetics.soft_key(w) for w in text.normalise(said).split()]
    if not want:
        return 0.0
    hits, i = 0, 0
    for w in want:
        while i < len(got):
            if phonetics.similarity(got[i], w) >= 0.8:
                hits += 1
                i += 1
                break
            i += 1
    return hits / len(want)


def _best_frame(said: str, accepted: list[dict]) -> tuple[dict | None, float]:
    best, best_score = None, 0.0
    for candidate in accepted:
        if candidate.get("open"):
            continue
        s = _frame_recall(said, candidate["mt"])
        if s > best_score:
            best, best_score = candidate, s
    return best, round(best_score, 4)


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

    frame_scored = False
    if n.get("free"):
        # A name, a place, a number: never marked wrong, because the app cannot
        # know it. But the frame around the slot is ordinary Maltese, so that part
        # is scored and reported — saying "Jien Pietru" is not the same as saying
        # "hello", and the app should not pretend it cannot tell.
        framed, recall = _best_frame(said, n.get("accept", []))
        frames = n.get("frames") or []
        if frames:
            # The node says where the slot is, so the frame is looked for where it
            # belongs: `Jien` at the start, `sena` at the end, the town or the age
            # in between and unjudged. Nothing else counts as the frame — a name on
            # its own scores 0 here even when it happens to sound like the example
            # answer's name, because the sentence the scene teaches was not said.
            anchor = _best_anchor(said, frames)
            # Unless what they said is one of the deliberate steps outside the frame.
            # `Dak sigriet!` is a real answer to how old you are, and how near they
            # came to *it* is a real score — better than the 0 its frame would give.
            # Near it, though: below the bar the rest of the app calls "almost", the
            # answer is neither the frame nor the sentence, and 31% of a line nobody
            # was aiming at is not feedback.
            if not (_outside_frames(match, frames) and score > anchor and score >= CLOSE):
                score, frame_scored = anchor, True
            # `framed` is whichever example answer is nearest, for the correction
            # card; it is None when nothing overlapped at all, and then the ordinary
            # match is still the best line to show.
            match = framed or match
        elif recall > score:
            match, score, frame_scored = (framed or match), recall, True
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
        # A name, a town, an age: accepted as given, whatever it is.
        "free": bool(n.get("free")),
        # …and whether the score is the frame around that slot rather than a match
        # against a listed answer. The client says so out loud, because "100%" means
        # two different things: the frame was right, or the whole sentence was.
        "frame_scored": frame_scored,
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

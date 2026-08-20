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
from functools import lru_cache

from .config import DATA_DIR
from . import phonetics, text

DIALOGUES = DATA_DIR / "dialogues.json"

# Tuned against the spread in tests: real recogniser variants of a correct answer
# land at 0.88+, a different-but-plausible sentence in the 0.6s, nonsense below.
CORRECT = 0.86
CLOSE = 0.62

# …and the other way of being right.
#
# One absolute threshold has to do two jobs that pull against each other: high
# enough that a different sentence is turned down, low enough that a garbled correct
# one is not. It cannot do both, and 0.86 resolves that by siding with strictness —
# which is why an answer said right comes back rejected when the recogniser drops a
# word. `Naħseb li iva` heard as `naħseb iva` scores 0.900. One more slip and it is
# out, on a sentence the learner said perfectly well.
#
# Rank does what a threshold cannot. If what was said is nearer to what this node
# accepts than to anything the app accepts anywhere else, and clearly nearer, then
# it is that answer — garbled — and not some other sentence. The question stops
# being "is this close enough" and becomes "close to *what*", which is the question
# with an answer. The app already believes this at the acoustic level: `MIN_CONFIDENCE`
# in app.js accepts the target outright when the *audio* ranks it clear of a field of
# other lines. This asks the same question of the transcript, for the turns where the
# audio could not answer it — which are exactly the turns where the transcript is
# worst, and where demanding 0.86 of it is least reasonable.
#
# Measured by degrading all 334 real transcripts with the recogniser's own observed
# error types, at 3x, 5x and 8x the rate observed on clean synthesised speech — a
# learner's accented voice against a Maltese model is somewhere in there — and asking
# each node to reject the nearest line it does *not* accept:
#
#                             said right, accepted     said wrong, accepted
#                              3x     5x     8x         3x     5x     8x
#   threshold 0.86 (before)   98.2%  91.3%  75.7%      3.6%   2.4%   1.5%
#   threshold 0.78            99.4%  99.4%  94.6%     14.1%  12.3%   9.9%   ← too lenient
#   0.86 with sound distance  99.1%  93.7%  79.0%      3.6%   2.4%   1.5%
#   …and accepted on a lead  100.0% 100.0%  97.9%      3.6%   2.4%   1.5%   ← this
#
# The last row turns nothing away that the old threshold accepted, and turns nothing
# *in* either: the wrong-answer column is identical at all three levels. Dropping the
# threshold to 0.78 instead buys less and costs five times as many wrong answers.
#
# Cost: the rival scan is 377 comparisons, ~20ms in a browser, and it only runs for a
# score in [0.66, 0.86) — the band where the app was about to turn the learner down
# anyway, and where it can afford to think about it.
NEAREST = 0.66   # below this, nearest to the target or not, nothing was said
LEAD = 0.06      # how far clear of every rival the target has to be


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
        # The Maltese frame the answer is scored on, `Jien …`, `Għandi … sena`. It
        # goes on the screen beside the English: grading the frame while telling the
        # learner "anything goes" asks them to guess the half that is marked.
        "frames": n.get("frames") or [],
        "options": [a["en"] for a in n.get("accept", []) if not a.get("open")],
        # Whether the answer is the learner's own — their name, their town — which
        # changes "say this" into "say something like this".
        "free": bool(n.get("free")),
        # One model answer, in Maltese, so the app can show and *say* what it is
        # waiting for before the learner has to produce it. `options` was English
        # only, which tells somebody what to mean and not what to utter.
        #
        # Deliberately a non-open one. Those are the entries `every_line()`
        # pre-synthesises, so the line offered here is always one the static build
        # has audio for; an open entry is a frame with a gap in it (`Jisimni …`),
        # which is neither speakable nor a model of anything.
        "answer": next(({"mt": a["mt"], "en": a.get("en", "")}
                        for a in n.get("accept", []) if not a.get("open")), None),
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
            if phonetics.key_similarity(got[i], w) >= 0.8:
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


# Keys that are one word in Maltese, where nothing general can tell:
#   `hu`/`huwa` and `hi`/`hija` are the pronoun long and short. Two letters is too
#   few for a ratio (0.67), and a rule about added letters would swallow `huma`,
#   which is `hu` plus two and means *they*.
#   `minn` fuses with the article and assimilates to the noun — mill-, mir-, mit-,
#   mid-, mis-, miċ-, miż-, mix-. One preposition, spelled nine ways, and a frame
#   anchored on any of them means all of them.
_ONE_WORD = [
    {"hu", "huwa"},
    {"hi", "hiya"},
    {"min", "mil", "mir", "mit", "mid", "mis", "mic", "miz"},
]


def _same_word(said: str, want: str) -> bool:
    """Is this word of the answer the frame's word? Keys, not spellings.

    Tolerant, because the recogniser is inventive — but not about a first letter that
    has been *swapped*. Maltese conjugates on it: `nibda` is I start, `tibda` is you
    start, `jibda` is he starts, and the ratio puts them 0.80 apart, inside the bar.
    A learner answering `X'ħin tibda x-xogħol?` with `Tibda fid-disgħa` has read the
    question back rather than answered it, and `frame right · 100%` would be a lie
    about the one thing the scene teaches. `m'għandix` against `għandi` likewise: the
    negation is not the frame, it is the answer beside it.

    A *vowel or glide* added or dropped in front is the recogniser, not the learner:
    `nħobb` for `inħobb`, `isimni` for `jisimni`, `inqum` for `nqum`. A consonant is
    not — `x'jaħdem` is the question with its interrogative still attached, and
    `żmien` is not `minn`. Below three letters even that is guesswork: `in`, the tail
    of `in-numru`, is `jien` with the glide gone and also just `in`, so short keys
    have to match outright or be listed above.

    The last thing it will not do is read a negation as its affirmative. Maltese
    negates with `ma …-x`, the `-x` keys to a trailing s, and `m'għandix` against
    `għandi` is close enough for the ratio at 0.89 — but `Ma niekolx laħam` is not an
    answer to what you like eating, it is the answer beside it.
    """
    if said == want:
        return True
    if any(said in group and want in group for group in _ONE_WORD):
        return True
    if min(len(said), len(want)) < 3:
        return False
    if said == want + "s":
        return False
    if said[:1] != want[:1]:
        added = said[:1] if said[1:] == want else (want[:1] if want[1:] == said else "")
        return added in ("a", "e", "i", "o", "u", "y")
    return phonetics.key_similarity(said, want) >= 0.8


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
            if phonetics.key_similarity(got[i], w) >= 0.8:
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


def pair_score(said: str, want: str) -> float:
    """How close what was said is to one particular line.

    Three readings, and the most generous wins, because each is blind to something
    the others see. `phonetics.similarity` counts characters shared in order and
    cannot say whether a change was a small one. `sound_similarity` can — it charges
    a substitution by kind, so `qadima` heard as `qatima` costs a third of a
    character rather than a whole one. And `text.score` is word-aligned, which is the
    only one of the three that notices a word is in the wrong place.
    """
    phon = phonetics.similarity(said, want, soft=True)
    return max(
        phon,
        phonetics.sound_similarity(said, want),
        0.6 * phon + 0.4 * text.score(said, want),
    )


def _best_match(said: str, accepted: list[dict]) -> tuple[dict | None, float]:
    best, best_score = None, 0.0
    for candidate in accepted:
        if candidate.get("open"):
            continue
        score = pair_score(said, candidate["mt"])
        if _QUOTED.search(candidate["mt"]):
            score = max(score, _frame_score(said, candidate["mt"]))
        if score > best_score:
            best, best_score = candidate, score
    return best, round(best_score, 4)


@lru_cache(maxsize=1)
def _rivals() -> tuple[tuple[str, str], ...]:
    """Every line the app accepts anywhere, each with its phonetic key.

    The competition. A node's own answers are excluded per call by key rather than
    by identity, because the same sentence is accepted at several nodes and a rival
    that is *the same words* is not a rival.
    """
    seen: dict[str, str] = {}
    for d in all_dialogues():
        for n in d.get("nodes", {}).values():
            for a in n.get("accept", []):
                if not a.get("open"):
                    seen.setdefault(phonetics.soft_key(a["mt"]), a["mt"])
    return tuple(sorted(seen.items()))


def _nearest_rival(said: str, accepted: list[dict]) -> float:
    """The best score reached by any line this node does *not* accept.

    Scored with `pair_score`, the same function the match itself goes through. A
    rival measured more meanly than the target would make the lead free.
    """
    ours = {phonetics.soft_key(a["mt"]) for a in accepted if not a.get("open")}
    return round(max((pair_score(said, mt) for k, mt in _rivals() if k not in ours),
                     default=0.0), 4)


MAX_ATTEMPTS = 2


def evaluate(dialogue_id: str, node_id: str, said: str, attempts: int = 0) -> dict:
    """Grade an utterance and return the next thing to say. No model involved."""
    n = node(dialogue_id, node_id)
    if not n:
        return {"error": "unknown node"}

    said = text.normalise(said)
    match, score = _best_match(said, n.get("accept", []))

    frame_scored = False
    on_lead = False
    if n.get("free"):
        # A name, a place, a number: never marked wrong, because the app cannot
        # know it. But the frame around the slot is ordinary Maltese, so that part
        # is scored and reported — saying "Jien Pietru" is not the same as saying
        # "hello", and the app should not pretend it cannot tell.
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
            # `match` is left as the nearest listed answer either way: it is the line
            # the correction card shows, and on the escape path it is the line the
            # score was measured against. The frame has no better candidate to offer —
            # recall favours the shortest answer, not the nearest one.
        else:
            # No slot to anchor on: score by how much of an example answer they
            # produced — not a frame, and not claimed as one.
            framed, recall = _best_frame(said, n.get("accept", []))
            if recall > score:
                match, score = (framed or match), recall
        verdict = "correct" if len(text.fold(said)) >= 2 else "wrong"
    elif score >= CORRECT:
        verdict = "correct"
    elif score >= NEAREST and score - _nearest_rival(said, n.get("accept", [])) >= LEAD:
        # Nearer to this answer than to anything else the app knows, by a clear
        # margin. Right, then, and heard badly.
        verdict, on_lead = "correct", True
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
        # Accepted because it was the clear nearest, not because it scored well. The
        # client says so and still shows the line: waving through a mangled answer
        # without showing what it should have sounded like teaches the mangling.
        "on_lead": on_lead,
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

    # On anything short of correct — and on anything accepted only on its lead —
    # show the target so they can say it back.
    if (verdict != "correct" or on_lead) and match:
        out["say_this_mt"] = match["mt"]
        out["say_this_en"] = match["en"]
        out["diff"] = text.diff_words(said, match["mt"])

    if next_node and verdict == "correct":
        out["next"] = present(dialogue_id, next_node)
    elif verdict == "correct":
        out["next"] = None
        out["finished"] = True
    return out


def answers_in(did: str) -> list[str]:
    """Every answer one dialogue accepts, deduplicated.

    The twin of `dialogue.answersIn` in the client, and here for the same reason the rest
    of this module is duplicated: the field a spoken answer is ranked against is a graded
    decision, and a graded decision that only one of the two engines can compute is a
    decision that cannot be swept. `FIELD_LOCAL` in `app.js` is the switch this feeds."""
    out, seen = [], set()
    for d in all_dialogues():
        if d.get("id") != did:
            continue
        for n in d.get("nodes", {}).values():
            for a in n.get("accept", []):
                if a.get("open"):
                    continue
                mt = (a.get("mt") or "").strip()
                if mt and mt not in seen:
                    seen.add(mt)
                    out.append(mt)
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

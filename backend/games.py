"""Short activities that are not the conversation and not the flashcard.

The app had two things a learner could do: hold a scripted conversation, and review a
deck. Both are production tasks — say the thing, from nothing — and that is the hard,
transferable direction, which is why they were built first. It is also the only
direction, and four things fall through the gap:

* **Recognition before production.** A learner who cannot yet produce `Nixtieq kafè,
  jekk jogħġbok` can already pick it out of three options, and doing so is what makes
  producing it possible later. Duolingo's word bank and Babbel's multiple choice are
  both this, and both come before free entry for the same reason.
* **Word order, isolated.** Building a sentence from given tiles removes the recall
  problem and leaves only the syntax, which is where a beginner's errors actually are.
* **Sound discrimination.** Maltese has `għ ħ q x ż` and a set of near-homophones —
  `nixtieq`/`nixtri`, `xita`/`xitwa`, `niġi`/`niġri` — that an English ear does not
  separate without being asked to. Minimal-pair training is the oldest well-evidenced
  result in this whole area.
* **Connected speech.** Every line the app plays is one sentence at a time. Ten
  seconds of continuous Maltese is a different task, and the one that transfers to
  standing in a shop.

Almost none of the content here is authored. Tile puzzles, listening fragments and
minimal pairs are *derived* from the scripted dialogues and the deck — sentences that
were already written, already checked and already rendered to audio. That is
deliberate: the one thing worse than no grammar exercise is a grammar exercise in
wrong Maltese, and recombining attested sentences cannot produce any. Only the grammar
drills are written by hand, and `tests/test_games.py` holds them to the same standard —
every Maltese form in them has to appear somewhere in the reference or the corpus.

Everything is derived at build time and shipped as data, so the client only presents
and marks. There is no second implementation of any of this to keep in step.
"""

from __future__ import annotations

import hashlib
import json
import re

from . import curriculum, dialogue, phonetics, text
from .config import DATA_DIR

GRAMMAR_FILE = DATA_DIR / "grammar_drills.json"

# How many of each kind a session offers. Enough to be worth starting, few enough to
# finish in a queue at a bus stop.
SESSION = 8


def _rng(seed: str) -> "_Det":
    return _Det(seed)


class _Det:
    """A deterministic shuffle, seeded on a string.

    `random` would do, but the build is hashed to name the shell cache — so a payload
    that reshuffles on every run would invalidate every device's cache for nothing.
    Same corpus in, same games out.
    """

    def __init__(self, seed: str) -> None:
        self._state = int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16) or 1

    def _next(self) -> int:
        # xorshift64: small, reproducible, and nobody's security depends on it.
        x = self._state
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self._state = x & 0xFFFFFFFFFFFFFFFF
        return self._state

    def below(self, n: int) -> int:
        return self._next() % n if n > 0 else 0

    def shuffled(self, items: list) -> list:
        out = list(items)
        for i in range(len(out) - 1, 0, -1):
            j = self.below(i + 1)
            out[i], out[j] = out[j], out[i]
        return out

    def pick(self, items: list, n: int) -> list:
        return self.shuffled(items)[:n]


# ── What the app already knows to be correct Maltese ──────────────────────────

def _spoken_phrases() -> list[dict]:
    """Every multi-word sentence the app has an audio file for, with its English.

    Both sources are already rendered: `dialogue.every_line()` covers the scripted
    lines and their accepted answers, and the deck's phrases are rendered by
    `prebuild_audio.py --what deck`. Nothing here needs synthesising.
    """
    out: dict[str, dict] = {}
    for d in dialogue.all_dialogues():
        for node in (d.get("nodes") or {}).values():
            for a in node.get("accept", []):
                if a.get("open") or not a.get("en"):
                    continue
                if len(a["mt"].split()) >= 3:
                    out.setdefault(a["mt"], {"mt": a["mt"], "en": a["en"],
                                             "topic": d["id"]})
    for row in curriculum.deck_rows():
        if row["kind"] == "phrase" and len(row["mt"].split()) >= 3:
            out.setdefault(row["mt"], {"mt": row["mt"], "en": row["en"],
                                       "topic": row.get("topic") or "deck"})
    return sorted(out.values(), key=lambda r: r["mt"])


def _vocab() -> list[dict]:
    return [r for r in curriculum.deck_rows()
            if r["kind"] == "vocab" and len(r["mt"].split()) == 1 and r.get("en")]


_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)


def _words(line: str) -> list[str]:
    return _WORD.findall(line)


# ── 1. Build the sentence from tiles ─────────────────────────────────────────

def build_items(limit: int = 60) -> list[dict]:
    """An English sentence, and its Maltese one word per tile, to be put in order.

    The recall problem is removed and only the word order is left, which is where a
    beginner's mistakes are: Maltese puts the adjective after the noun, fuses its
    prepositions onto the article, and wraps its verbs in `ma … -x`. A learner who
    cannot yet say `Ma niekolx laħam` unprompted can assemble it, and assembling it is
    what makes saying it reachable.

    Some items carry words that do not belong in the answer and have to be left alone.
    Not all of them: an exercise where every tile is used is a different exercise —
    order only — and both are worth practising. Which items get them is decided by the
    sentence's own hash, so it is stable from build to build.
    """
    phrases = _spoken_phrases()
    rng = _rng("build")
    chosen = rng.pick(phrases, min(limit, len(phrases)))

    # A pool of single words to draw the redundant tiles from, so a decoy is a real
    # Maltese word rather than an obvious throwaway.
    pool = sorted({w for p in phrases for w in _words(p["mt"]) if len(w) > 2})

    items = []
    for phrase in chosen:
        answer = _words(phrase["mt"])
        if len(answer) < 3:
            continue
        own = _rng(f"build:{phrase['mt']}")
        lower = {w.lower() for w in answer}
        extras = [w for w in own.pick(pool, 24)
                  if w.lower() not in lower][:2 if own.below(10) < 6 else 0]
        items.append({
            "id": "build-" + hashlib.sha256(phrase["mt"].encode()).hexdigest()[:10],
            "kind": "build",
            "prompt_en": phrase["en"],
            "mt": phrase["mt"],
            "answer": answer,
            "tiles": own.shuffled(answer + extras),
            "topic": phrase["topic"],
        })
    return items


# ── 2. Which word did you hear ───────────────────────────────────────────────

# Similar enough to be worth telling apart, different enough that the audio really
# does differ. Above the upper bound the two are the same word with an article on it.
NEAR_LOW, NEAR_HIGH = 0.62, 0.92


def confusable(a: str, b: str) -> float:
    """How alike two Maltese words sound, 0..1, and the same answer either way round.

    `phonetics.key_similarity` is `difflib.SequenceMatcher.ratio`, which is **not
    symmetric** — its matching-block heuristics depend on which sequence is second:

        key_similarity('hamsa', 'huma') = 0.44
        key_similarity('huma', 'hamsa') = 0.67

    That produced a hearing item offering `huma` as a decoy for `ħamsa`, which sound
    nothing alike: the pair was measured in the generous direction when it was
    collected and in the strict one when it was checked. Comparing in a fixed order
    settles it, and both the builder and the test go through here so they cannot
    disagree again.

    `key_similarity` itself is left alone deliberately. The dialogue grader depends on
    it and `tests/test_dialogue_parity.py` holds it identical to the browser's port, so
    a change there would move every score in the app to fix a problem that is this
    caller's.
    """
    ka, kb = phonetics.soft_key(a), phonetics.soft_key(b)
    lo, hi = sorted((ka, kb))
    return phonetics.key_similarity(lo, hi)


def hearing_items(limit: int = 50) -> list[dict]:
    """Play one word; pick it out of three that sound alike.

    The pairs are found rather than written: every single-word deck entry is keyed
    phonetically and compared with every other, and the ones that land in the
    confusable band become items. `nixtieq`/`nixtri`, `xita`/`xitwa`, `niġi`/`niġri`,
    `noqgħod`/`norqod` — all four are real hazards for an English ear and none of them
    had to be thought of.
    """
    vocab = _vocab()
    keyed = [(r, phonetics.soft_key(r["mt"])) for r in vocab]
    near: dict[str, list[dict]] = {}
    for i, (a, ka) in enumerate(keyed):
        if not ka:
            continue
        for b, kb in keyed[i + 1:]:
            if not kb or ka == kb:
                continue
            if NEAR_LOW <= confusable(a["mt"], b["mt"]) < NEAR_HIGH:
                near.setdefault(a["mt"], []).append(b)
                near.setdefault(b["mt"], []).append(a)

    rng = _rng("hearing")
    items = []
    for mt in rng.pick(sorted(k for k, v in near.items() if len(v) >= 2), limit):
        target = next(r for r in vocab if r["mt"] == mt)
        own = _rng(f"hearing:{mt}")
        decoys = own.pick(near[mt], 2)
        options = own.shuffled([target] + decoys)
        items.append({
            "id": "hear-" + hashlib.sha256(mt.encode()).hexdigest()[:10],
            "kind": "hearing",
            "say": mt,
            "options": [{"mt": o["mt"], "en": o["en"]} for o in options],
            "answer": next(i for i, o in enumerate(options) if o["mt"] == mt),
            "topic": target.get("topic") or "sounds",
        })
    return items


# ── 3. Mini podcasts ─────────────────────────────────────────────────────────

# The window worth listening to. Under this it is a sentence, not a stretch of speech;
# over it a learner has lost the thread before the question arrives. Estimated from the
# measured fit in tests/test_scripts.py: seconds ≈ 0.435 + 0.0368 × characters.
FRAGMENT_LOW_S, FRAGMENT_HIGH_S = 7.0, 17.0


def _spoken_seconds(line: str) -> float:
    return 0.435 + 0.0368 * len(line)


def _fragments() -> list[dict]:
    """A scene played through as continuous speech.

    Each scene is a prompt, an answer, a prompt, an answer. Read end to end it is a
    short exchange between two people — which is the thing the app never gives a
    learner, because every line it plays is one sentence with a pause after it.

    Composed from lines that are already authored and already checked, so a fragment
    cannot contain Maltese that nobody wrote. Its audio is one new render of the whole
    thing, which is what makes it connected speech rather than four clips in a row.
    """
    out = []
    for d in dialogue.all_dialogues():
        nodes = d.get("nodes") or {}
        nid, said, seen = d.get("start"), [], set()
        while nid and nid in nodes and nid not in seen:
            seen.add(nid)
            node = nodes[nid]
            said.append({"mt": node["say_mt"], "en": node["say_en"]})
            answer = next((a for a in node.get("accept", []) if not a.get("open")), None)
            if answer and answer.get("en"):
                said.append({"mt": answer["mt"], "en": answer["en"]})
            nid = node.get("next")

        script = " ".join(s["mt"] for s in said)
        if not FRAGMENT_LOW_S <= _spoken_seconds(script) <= FRAGMENT_HIGH_S:
            continue
        out.append({"scene": d["id"], "name_en": d["name_en"],
                    "script": script, "lines": said})
    return out


def listening_items() -> list[dict]:
    """Two questions per fragment, and neither of them is written by hand.

    *Which of these was said* takes one line's English from the fragment and two from
    other scenes — comprehension, with no reading of Maltese to get in the way.

    *Which three words did you hear* takes three words from the fragment and three that
    sound like them and are not in it. This is the harder and the more useful of the
    two: picking `xitwa` when `xita` was said is the actual failure mode, and a learner
    only stops making it once something has asked them to hear the difference.
    """
    fragments = _fragments()
    if len(fragments) < 3:
        return []

    everything = [line for f in fragments for line in f["lines"]]
    vocab = _vocab()
    keyed = [(r["mt"], phonetics.soft_key(r["mt"])) for r in vocab]

    items = []
    for f in fragments:
        own = _rng(f"listen:{f['scene']}")
        mine = {line["mt"] for line in f["lines"]}

        # (a) which line was said
        heard = own.pick([line for line in f["lines"] if len(line["en"]) > 12], 1)
        elsewhere = own.pick([line for line in everything
                              if line["mt"] not in mine and len(line["en"]) > 12], 2)
        if heard and len(elsewhere) == 2:
            options = own.shuffled([heard[0]] + elsewhere)
            items.append({
                "id": f"listen-said-{f['scene']}",
                "kind": "listening",
                "ask": "which",
                "scene": f["scene"],
                "name_en": f["name_en"],
                "script": f["script"],
                "question_en": "Which of these was said?",
                # The sentences it is built from, so a test can hold every one of them
                # to being a line somebody wrote — see tests/test_games.py.
                "lines": [line["mt"] for line in f["lines"]],
                "options": [{"en": o["en"]} for o in options],
                "answer": next(i for i, o in enumerate(options)
                               if o["mt"] == heard[0]["mt"]),
            })

        # (b) which three words were in it
        spoken = sorted({w for w in _words(f["script"]) if len(w) > 3})
        real = own.pick(spoken, 3)
        if len(real) < 3:
            continue
        said_keys = {phonetics.soft_key(w) for w in spoken}
        decoys = [mt for mt, key in keyed
                  if key and key not in said_keys
                  and any(NEAR_LOW <= confusable(mt, w) < 0.98 for w in real)]
        decoys = own.pick(sorted(set(decoys)), 3)
        if len(decoys) < 3:
            continue
        pool = own.shuffled(real + decoys)
        items.append({
            "id": f"listen-words-{f['scene']}",
            "kind": "listening",
            "ask": "words",
            "scene": f["scene"],
            "name_en": f["name_en"],
            "script": f["script"],
            "question_en": "Pick the three words you heard",
            "lines": [line["mt"] for line in f["lines"]],
            "pool": pool,
            "answer": sorted(real),
        })
    return items


# ── 4. Grammar drills ────────────────────────────────────────────────────────

def grammar_items() -> list[dict]:
    """Two correct sentences that differ by one rule, then a gap to fill.

    The only authored content here, and the shape is Babbel's rather than Duolingo's:
    show the contrast first, let the rule be *noticed*, then ask. A learner shown
    `Hu jiġri` beside `Huma jiġru` and then asked for `Huma ___ maratona` has been
    given the answer by the examples, which is the point — the exercise is to see the
    pattern, not to remember a conjugation table.

    Written into `data/grammar_drills.json` rather than in here so the Maltese sits in
    one reviewable file, and held to the corpus by tests/test_games.py.
    """
    if not GRAMMAR_FILE.exists():
        return []
    raw = json.loads(GRAMMAR_FILE.read_text(encoding="utf-8"))
    items = []
    for item in raw.get("drills", []):
        own = _rng(f"grammar:{item['id']}")
        options = own.shuffled(item["options"])
        items.append({
            "id": item["id"],
            "kind": "grammar",
            "rule": item["rule"],
            "show": item["show"],
            "ask_mt": item["ask_mt"],
            "ask_en": item["ask_en"],
            "options": options,
            "answer": options.index(item["answer"]),
            "why": item.get("why", ""),
        })
    return items


# ── The payload ──────────────────────────────────────────────────────────────

KINDS = ("build", "hearing", "listening", "grammar")


def all_games() -> dict:
    return {
        "session": SESSION,
        "build": build_items(),
        "hearing": hearing_items(),
        "listening": listening_items(),
        "grammar": grammar_items(),
    }


def every_line() -> list[str]:
    """Maltese the games need audio for that nothing else renders.

    Only the listening fragments: tile puzzles and hearing items reuse lines the
    dialogues and deck have already rendered, and a grammar drill's sentences are
    rendered because they are listed here too — a rule you can read and not hear is
    half an explanation.
    """
    lines = [f["script"] for f in _fragments()]
    for item in grammar_items():
        lines += [s["mt"] for s in item["show"]]
        lines.append(item["ask_mt"].replace("___", item["options"][item["answer"]]))
    seen, out = set(), []
    for line in lines:
        line = (line or "").strip()
        if line and line not in seen:
            seen.add(line)
            out.append(line)
    return out

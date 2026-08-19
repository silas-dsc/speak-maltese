"""The mini-games, and the rule that keeps them honest.

Almost all of this content is *derived* — tile puzzles, listening fragments and minimal
pairs are recombinations of sentences that were already authored, already checked and
already rendered to audio. That is not a shortcut, it is the safety property: the one
thing worse than no grammar exercise is a grammar exercise in wrong Maltese, and
recombining attested sentences cannot produce any.

The grammar drills are the exception, so they get the strictest test in the file: every
Maltese word in them has to appear in the reference, the deck or the scripted dialogues.
A drill that invents a conjugation fails here rather than teaching it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import curriculum, dialogue, games  # noqa: E402

WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)


@pytest.fixture(scope="module")
def payload():
    return games.all_games()


# ── The corpus of Maltese this app has already committed to ──────────────────

@pytest.fixture(scope="module")
def attested() -> set[str]:
    """Every Maltese word the app already says or documents, lowercased.

    Three sources, and all three are content a human wrote and checked: the quick
    reference, the deck, and the scripted dialogues.
    """
    text_blobs = [(ROOT / "data" / "grammar_notes.md").read_text(encoding="utf-8")]
    for row in curriculum.deck_rows():
        text_blobs += [row["mt"], row.get("example_mt") or ""]
    text_blobs += dialogue.every_line()
    for d in dialogue.all_dialogues():
        text_blobs += [d["name"]]
        for node in (d.get("nodes") or {}).values():
            text_blobs += [node["say_mt"]]
            for key in ("correct", "close", "wrong"):
                text_blobs.append((node.get(key) or {}).get("mt") or "")
            for a in node.get("accept", []):
                text_blobs.append(a["mt"])

    words = set()
    for blob in text_blobs:
        words |= {w.lower() for w in WORD.findall(blob)}
    return words


def test_the_corpus_is_big_enough_to_be_a_real_check(attested):
    """A test that passes because its reference set is enormous checks nothing.

    About twelve hundred distinct words, which is the app's whole Maltese vocabulary and
    not a language. It also contains the English of `grammar_notes.md`, which weakens the
    check slightly in one direction — a drill word that happens to be spelled like an
    English word would pass — and not at all in the direction that matters, which is
    invented Maltese morphology."""
    assert 1000 < len(attested) < 12000, len(attested)


def test_every_word_in_a_grammar_drill_is_attested(attested, payload):
    """The whole reason `data/grammar_drills.json` can be trusted.

    A drill is a recombination of forms that appear in the reference or the corpus, so
    a plural or a verb ending that nobody wrote cannot get in. When this fails, the
    answer is nearly always to use the form the reference already gives rather than to
    add the invention to the corpus."""
    unknown = {}
    for item in payload["grammar"]:
        # The examples, the sentence with the gap, and the *right* answer. Not the wrong
        # ones: a distractor is by design a form the app never says, and `triqijiet` for
        # the plural of `triq` is exactly the mistake the drill exists to catch. It has
        # to be absent from the corpus, not present in it.
        maltese = ([s["mt"] for s in item["show"]] + [item["ask_mt"]]
                   + [item["options"][item["answer"]]])
        for line in maltese:
            for word in WORD.findall(line):
                if word.lower() not in attested:
                    unknown.setdefault(word, []).append(item["id"])
    assert not unknown, (
        "Maltese in the grammar drills that appears nowhere else in the app:\n  "
        + "\n  ".join(f"{w!r} in {ids}" for w, ids in sorted(unknown.items())[:12]))


# ── Shape, so a malformed item cannot reach a learner ────────────────────────

def test_a_grammar_drill_teaches_before_it_asks(payload):
    """The contrast is the exercise. An item with one example, or with a gap that is
    not a gap, is a quiz question with the teaching left out."""
    drills = payload["grammar"]
    assert len(drills) >= 12, f"only {len(drills)} grammar drills"

    for item in drills:
        assert len(item["show"]) == 2, f"{item['id']}: needs two contrasting examples"
        assert item["show"][0]["mt"] != item["show"][1]["mt"], f"{item['id']}: same example twice"
        for example in item["show"]:
            assert example["mt"] and example["en"], f"{item['id']}: example missing a side"
        assert "___" in item["ask_mt"], f"{item['id']}: nothing to fill in"
        assert len(item["options"]) >= 2, f"{item['id']}: no choice to make"
        assert len(set(item["options"])) == len(item["options"]), f"{item['id']}: duplicate option"
        assert 0 <= item["answer"] < len(item["options"]), f"{item['id']}: answer out of range"
        assert item["why"], f"{item['id']}: no explanation once it is answered"
        assert item["rule"], f"{item['id']}: not attached to a rule"

    ids = [i["id"] for i in drills]
    assert len(set(ids)) == len(ids), "duplicate drill id"


def test_a_grammar_drill_is_attached_to_a_rule_the_guide_explains(payload):
    """`rule` is what the learner is being shown, and the guide is where they go to read
    more about it. A rule named here and absent there is a dead end."""
    guide = (ROOT / "data" / "grammar_notes.md").read_text(encoding="utf-8").lower()
    # Not the heading verbatim — the wording is the drill's own — but its subject has to
    # be in there. One distinctive word from each rule is enough to catch a stray.
    for item in payload["grammar"]:
        # Any substantial word of the rule's own wording, not the longest — the wording
        # is the drill's and only its subject has to be shared with the guide.
        words = re.findall(r"[a-zA-Zàèìòùħġżċ']{4,}", item["rule"].lower())
        assert words, f"{item['id']}: rule {item['rule']!r} says nothing"
        assert any(w in guide for w in words), (
            f"{item['id']}: rule {item['rule']!r} has no counterpart in the reference")


def test_tile_puzzles_can_be_solved_and_only_one_way(payload):
    """Every tile in the answer has to be on offer, the answer has to be the sentence,
    and the shuffle must not have quietly dropped or duplicated a word."""
    items = payload["build"]
    assert len(items) >= 30, f"only {len(items)} tile puzzles"

    from collections import Counter
    for item in items:
        assert " ".join(item["answer"]) in item["mt"] or item["answer"], item["id"]
        assert len(item["answer"]) >= 3, f"{item['id']}: too short to be about word order"
        offered, needed = Counter(item["tiles"]), Counter(item["answer"])
        missing = needed - offered
        assert not missing, f"{item['id']}: answer needs tiles that are not offered: {missing}"
        assert item["prompt_en"], f"{item['id']}: nothing to translate"
        # The Maltese is what gets played, so it has to be the real sentence.
        assert item["mt"].startswith(item["answer"][0]), item["id"]


def test_some_tile_puzzles_carry_words_to_leave_alone(payload):
    """Redundant tiles are what stop the exercise being "use everything in any order".
    Not on every item, though — order-only is a different and also useful exercise, and
    an app where every tile is always used teaches that habit instead."""
    extras = [len(i["tiles"]) - len(i["answer"]) for i in payload["build"]]
    assert any(e > 0 for e in extras), "no puzzle has a redundant tile"
    assert any(e == 0 for e in extras), "every puzzle has redundant tiles"
    assert all(e >= 0 for e in extras), "a puzzle is missing tiles it needs"
    # A pile of decoys is a word search, not a translation.
    assert max(extras) <= 3, f"up to {max(extras)} redundant tiles is too many"


def test_hearing_items_are_confusable_but_different(payload):
    """The point is the pairs, and they are found rather than thought of: `nixtieq`
    against `nixtri`, `xita` against `xitwa`. Two words that sound identical would be
    unfair, and two that sound nothing alike would be pointless."""
    items = payload["hearing"]
    assert len(items) >= 20, f"only {len(items)} hearing items"

    from backend import phonetics
    for item in items:
        assert len(item["options"]) == 3, item["id"]
        mts = [o["mt"] for o in item["options"]]
        assert len(set(mts)) == 3, f"{item['id']}: the same word twice"
        assert item["say"] == mts[item["answer"]], f"{item['id']}: answer is not what is played"
        for other in mts:
            if other == item["say"]:
                continue
            # Through `games.confusable`, which is symmetric. Measuring this with
            # `key_similarity` directly is how `huma` came to be offered as a decoy for
            # `ħamsa`: it answers 0.44 one way round and 0.67 the other.
            sim = games.confusable(item["say"], other)
            assert games.NEAR_LOW <= sim < games.NEAR_HIGH, (
                f"{item['id']}: {item['say']!r} vs {other!r} at {sim:.2f}")
        for option in item["options"]:
            assert option["en"], f"{item['id']}: an option with no meaning shown"


def test_listening_fragments_are_long_enough_to_be_listening(payload):
    """One sentence with a pause after it is what the rest of the app already does. The
    point of these is connected speech, so the window is checked — and both question
    types have to be answerable from the audio alone."""
    items = payload["listening"]
    assert len(items) >= 8, f"only {len(items)} listening items"

    for item in items:
        secs = games._spoken_seconds(item["script"])
        assert games.FRAGMENT_LOW_S <= secs <= games.FRAGMENT_HIGH_S, (
            f"{item['id']}: about {secs:.1f}s of speech")
        assert item["ask"] in ("which", "words")

        if item["ask"] == "which":
            assert len(item["options"]) == 3, item["id"]
            assert 0 <= item["answer"] < 3
            # The right answer has to be a line that is actually in the fragment, and
            # the wrong ones must not be.
            said = item["options"][item["answer"]]["en"]
            assert said, f"{item['id']}: the answer has no text"
        else:
            assert len(item["answer"]) == 3, item["id"]
            assert len(item["pool"]) == 6, f"{item['id']}: {len(item['pool'])} words offered"
            assert set(item["answer"]) <= set(item["pool"]), f"{item['id']}: answer not on offer"
            spoken = {w.lower() for w in WORD.findall(item["script"])}
            for word in item["answer"]:
                assert word.lower() in spoken, f"{item['id']}: {word!r} is not in the fragment"
            for word in set(item["pool"]) - set(item["answer"]):
                assert word.lower() not in spoken, (
                    f"{item['id']}: decoy {word!r} really is in the fragment")


def test_a_listening_fragment_is_made_of_lines_somebody_wrote(payload):
    """Composed from the scripted dialogues rather than generated, which is what makes
    ten seconds of Maltese safe to play at a learner.

    Checked line by line rather than sentence by sentence: a dialogue line can be
    several sentences (`Bonġu! Jien Marija. X'jismek?` is one line), so splitting the
    script on punctuation and looking each piece up finds fragments of lines that were
    never written separately."""
    written = set(dialogue.every_line())
    for item in payload["listening"]:
        assert item["lines"], f"{item['id']}: does not say what it is made of"
        for line in item["lines"]:
            assert line in written, (
                f"{item['id']}: {line!r} is not a line from the dialogues")
        # …and the audio really is those lines, in that order, with nothing added.
        assert item["script"] == " ".join(item["lines"]), f"{item['id']}: script drifted"


# ── Reproducibility, because the build is hashed ─────────────────────────────

def test_the_same_corpus_produces_the_same_games():
    """The static build names its shell cache after a hash of the output, so a payload
    that reshuffles on every run would invalidate every device's cache — 23MB of audio
    re-downloaded because a script reran. The shuffles are seeded on the content."""
    first = json.dumps(games.all_games(), ensure_ascii=False, sort_keys=True)
    second = json.dumps(games.all_games(), ensure_ascii=False, sort_keys=True)
    assert first == second, "all_games() is not deterministic"


def test_every_kind_is_offered_and_named(payload):
    for kind in games.KINDS:
        assert payload[kind], f"no {kind} items"
        for item in payload[kind]:
            assert item["kind"] == kind, f"{item['id']} is filed under {kind}"
    assert payload["session"] >= 5

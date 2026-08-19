"""Marking the mini-games, in the browser's own engine.

The items are built in Python and shipped as data, so there is no second
implementation of the *derivation* to keep in step — which is why this file is short.
What is left in `frontend/games.js` is the part a learner feels directly: whether an
answer counts, what they are told when it does not, what order the exercises come in,
and which correct answers are allowed to reach the review deck.

That last one is the reason this is tested rather than eyeballed. FSRS is only as good
as what it is told, and a card filed as known because it was recognised among three
options comes back at the interval of something produced from memory.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node is not installed")

DRIVER = r"""
import { mark, critique, session, earned, audioFor } from '../frontend/games.js';

const build = {
  kind: 'build', mt: 'Nixtieq kafè, jekk jogħġbok.', prompt_en: "I'd like a coffee, please.",
  answer: ['Nixtieq', 'kafè', 'jekk', 'jogħġbok'],
  tiles: ['jekk', 'grazzi', 'Nixtieq', 'jogħġbok', 'kafè'],
};
const hearing = { kind: 'hearing', say: 'nixtieq', answer: 1,
                  options: [{ mt: 'nixtri' }, { mt: 'nixtieq' }, { mt: 'nifhem' }] };
const which = { kind: 'listening', ask: 'which', answer: 2, script: 'Bonġu!',
                options: [{ en: 'a' }, { en: 'b' }, { en: 'c' }] };
const words = { kind: 'listening', ask: 'words', script: 'Bonġu! Kif int?',
                pool: ['fejn', 'ilma', 'banju', 'bard', 'imbagħad', 'ġimgħa'],
                answer: ['banju', 'fejn', 'ilma'] };
const grammar = { kind: 'grammar', ask_mt: 'Huma ___ ittra.', answer: 0,
                  options: ['jiktbu', 'jikteb'], why: 'because' };

const out = {};

// ── marking ────────────────────────────────────────────────────────────────
out.buildRight = mark(build, ['Nixtieq', 'kafè', 'jekk', 'jogħġbok']);
out.buildWrongOrder = mark(build, ['kafè', 'Nixtieq', 'jekk', 'jogħġbok']);
out.buildWithExtra = mark(build, ['Nixtieq', 'kafè', 'jekk', 'jogħġbok', 'grazzi']);
out.buildShort = mark(build, ['Nixtieq', 'kafè']);
out.buildEmpty = mark(build, []);

out.hearingRight = mark(hearing, 1);
out.hearingWrong = mark(hearing, 0);
out.whichRight = mark(which, 2);
out.whichWrong = mark(which, 0);
out.wordsRightAnyOrder = mark(words, ['ilma', 'banju', 'fejn']);
out.wordsMissingOne = mark(words, ['ilma', 'banju']);
out.wordsWithDecoy = mark(words, ['ilma', 'banju', 'bard']);
out.grammarRight = mark(grammar, 0);
out.grammarWrong = mark(grammar, 1);

// ── what the learner is told ───────────────────────────────────────────────
out.critiqueOrder = critique(build, ['kafè', 'Nixtieq', 'jekk', 'jogħġbok']);
out.critiqueExtra = critique(build, ['Nixtieq', 'kafè', 'jekk', 'jogħġbok', 'grazzi']);
out.critiqueShort = critique(build, ['Nixtieq', 'kafè']);
out.critiqueNone = critique(build, []);
out.critiqueOther = critique(hearing, 0);

// ── the deck only hears about production ───────────────────────────────────
out.earnedBuild = earned(build, true);
out.earnedBuildWrong = earned(build, false);
out.earnedHearing = earned(hearing, true);
out.earnedGrammar = earned(grammar, true);
out.earnedListening = earned(words, true);

// ── sessions ───────────────────────────────────────────────────────────────
const payload = {
  build: Array.from({ length: 9 }, (_, i) => ({ ...build, id: `b${i}` })),
  hearing: Array.from({ length: 9 }, (_, i) => ({ ...hearing, id: `h${i}` })),
  listening: Array.from({ length: 9 }, (_, i) => ({ ...words, id: `l${i}` })),
  grammar: Array.from({ length: 9 }, (_, i) => ({ ...grammar, id: `g${i}` })),
};
out.mixedKinds = session(payload, { count: 8, seed: 1 }).map((i) => i.kind);
out.oneKind = session(payload, { count: 5, kinds: ['grammar'], seed: 1 }).map((i) => i.kind);
out.sameSeed = JSON.stringify(session(payload, { count: 8, seed: 7 }).map((i) => i.id))
  === JSON.stringify(session(payload, { count: 8, seed: 7 }).map((i) => i.id));
out.differentSeed = JSON.stringify(session(payload, { count: 8, seed: 7 }).map((i) => i.id))
  !== JSON.stringify(session(payload, { count: 8, seed: 8 }).map((i) => i.id));
out.noDuplicates = (() => {
  const ids = session(payload, { count: 8, seed: 3 }).map((i) => i.id);
  return new Set(ids).size === ids.length;
})();
// Asking for more than exists returns what exists rather than looping or hanging.
out.overAsk = session({ grammar: payload.grammar.slice(0, 2) },
                      { count: 8, kinds: ['grammar'], seed: 1 }).length;
out.emptyPayload = session({}, { count: 8, seed: 1 }).length;

// ── what gets played ───────────────────────────────────────────────────────
out.audioBuild = audioFor(build);
out.audioHearing = audioFor(hearing);
out.audioListening = audioFor(words);
out.audioGrammar = audioFor(grammar);

console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def result():
    driver = ROOT / "tests" / "_games_driver.mjs"
    driver.write_text(DRIVER, encoding="utf-8")
    try:
        proc = subprocess.run([node, str(driver)], cwd=ROOT,
                              capture_output=True, text=True, timeout=30)
    finally:
        driver.unlink(missing_ok=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_a_built_sentence_has_to_be_in_the_right_order(result):
    """The exercise is word order — Maltese puts the adjective after the noun and fuses
    its prepositions onto the article — so accepting the right words in the wrong order
    would teach that word order is optional, which is the one thing it exists to teach
    against."""
    assert result["buildRight"] is True
    assert result["buildWrongOrder"] is False, "the right words in the wrong order is wrong"
    assert result["buildShort"] is False
    assert result["buildEmpty"] is False


def test_a_redundant_tile_has_to_be_left_alone(result):
    """Some puzzles carry words that do not belong. Using one is a wrong answer, not a
    harmless extra — leaving it is the whole of that variant."""
    assert result["buildWithExtra"] is False


def test_choosing_between_options_is_marked_exactly(result):
    for right, wrong in (("hearingRight", "hearingWrong"),
                         ("whichRight", "whichWrong"),
                         ("grammarRight", "grammarWrong")):
        assert result[right] is True, right
        assert result[wrong] is False, wrong


def test_three_words_count_in_any_order(result):
    """Which words were in the clip is a set, not a sequence — nobody heard them in the
    order the tiles happen to be laid out."""
    assert result["wordsRightAnyOrder"] is True
    assert result["wordsMissingOne"] is False
    assert result["wordsWithDecoy"] is False


def test_a_wrong_answer_says_what_was_wrong_with_it(result):
    """"Wrong" is the least useful thing an exercise can say. The two ways to fail a tile
    puzzle need different corrections and the app knows which one it was."""
    assert "order" in result["critiqueOrder"].lower()
    assert "grazzi" in result["critiqueExtra"], "must name the word that does not belong"
    assert "missing" in result["critiqueShort"].lower()
    assert result["critiqueNone"], "an empty answer still deserves a sentence"
    # Only the tile puzzle has anything worth saying; the rest show it on the buttons.
    assert result["critiqueOther"] == ""


def test_only_production_reaches_the_review_deck(result):
    """FSRS is only as good as what it is told. A card filed as known because it was
    recognised among three options comes back at the interval of something produced
    from memory, and the deck fills with words the learner cannot say.

    Assembling a sentence tile by tile is production of a sort — the words are given,
    the sentence is theirs — so that one counts. Picking one word out of three is
    recognition, and nothing is filed for it. Same reasoning as the drill's `peeked`."""
    assert result["earnedBuild"] == [{"mt": "Nixtieq kafè, jekk jogħġbok.",
                                     "en": "I'd like a coffee, please."}]
    assert result["earnedBuildWrong"] == [], "a wrong answer is not evidence of anything"
    assert result["earnedHearing"] == [], "recognising a word is not producing it"
    assert result["earnedGrammar"] == [], "choosing an ending is not producing a phrase"
    assert result["earnedListening"] == [], "hearing a word is not producing it"


def test_a_mixed_round_interleaves_the_kinds(result):
    """Round-robin rather than a shuffle of the pool. A shuffle gives runs — four tile
    puzzles together happens often in eight draws — and the second and third of a run
    are answered by momentum rather than by knowing anything."""
    kinds = result["mixedKinds"]
    assert len(kinds) == 8
    assert len(set(kinds)) == 4, f"only {set(kinds)} appeared"
    # No kind twice in a row.
    assert all(a != b for a, b in zip(kinds, kinds[1:])), kinds


def test_a_single_kind_round_is_all_that_kind(result):
    """Someone who has opened the listening game has said what they want to practise."""
    assert result["oneKind"] == ["grammar"] * 5


def test_a_session_is_reproducible_and_does_not_repeat_itself(result):
    assert result["sameSeed"], "the same seed gave a different round"
    assert result["differentSeed"], "every round is the same round"
    assert result["noDuplicates"], "an item appeared twice in one round"


def test_asking_for_more_than_exists_returns_what_exists(result):
    """The pools are finite and unequal — 19 grammar drills against 60 tile puzzles — so
    a round of eight has to end rather than loop or hang."""
    assert result["overAsk"] == 2
    assert result["emptyPayload"] == 0


def test_each_kind_knows_what_to_play(result):
    """Every item has a line of Maltese behind it, including the grammar drills — a rule
    you can read and not hear is half an explanation. The drill plays the sentence with
    the gap *filled*, which is the only version worth hearing."""
    assert result["audioBuild"] == "Nixtieq kafè, jekk jogħġbok."
    assert result["audioHearing"] == "nixtieq"
    assert result["audioListening"] == "Bonġu! Kif int?"
    assert result["audioGrammar"] == "Huma jiktbu ittra.", "the gap has to be filled in"

"""Maltese phonetic keying, for matching what a learner said against a set of
accepted answers without calling a model.

The point is speed. A scripted turn has to feel like conversation, so there is no
budget for an LLM round trip — but a plain string compare is far too brittle, because
the recogniser and the learner both vary the spelling constantly. Reducing both sides
to how they *sound* absorbs almost all of that variance for the price of a regex pass.

Maltese orthography is unusually well suited to this: it is close to phonemic, so a
small set of rewrites gets you most of the way.

    ie → long i          għ → silent (lengthens its neighbour)
    x  → ʃ               ħ, h → h (h is silent word-finally)
    ż  → z               z → ts
    ċ  → tʃ              ġ → dʒ, j → j (as in *yes*)
    q  → glottal stop, kept distinct so `qalb` ≠ `kelb`
"""

from __future__ import annotations

import difflib
import re
import unicodedata

# Order matters: digraphs are consumed before the single letters inside them.
# Distinctions that exist only in the diacritic (ġ/g, ż/z, ċ/c) are deliberately
# folded together: losing the dot is the single most common thing a recogniser does
# to Maltese, and this key exists to be tolerant, not to be phonemically pure. `q`
# stays distinct, because that one is transcribed reliably and conflating it would
# merge real words (`qalb` heart / `kelb` dog).
_RULES: list[tuple[str, str]] = [
    ("għ", ""),      # silent; lengthening is handled by vowel collapsing below
    ("gh", ""),      # de-diacriticised form recognisers produce
    ("ie", "I"),     # one long vowel
    ("ċ", "C"), ("ch", "C"), ("c", "C"),
    ("ġ", "J"), ("g", "J"),
    ("x", "S"),      # ʃ
    ("ż", "z"), ("ts", "z"),   # ts is how `z` is often written back
    ("q", "Q"),      # glottal stop
    ("ħ", "h"),
    ("j", "y"),
]

_VOWELS = str.maketrans({"à": "a", "è": "e", "ì": "i", "ò": "o", "ù": "u"})
_NON_WORD = re.compile(r"[^a-zA-Z\s]")
_WS = re.compile(r"\s+")


def key(s: str) -> str:
    """Phonetic key for a Maltese word or phrase."""
    s = unicodedata.normalize("NFC", (s or "").strip().lower())
    s = s.replace("’", "'").replace("`", "'")
    s = s.translate(_VOWELS)

    for src, dst in _RULES:
        s = s.replace(src, dst)

    s = s.replace("'", "")
    # h is silent at the end of a word
    s = re.sub(r"h\b", "", s)
    s = _NON_WORD.sub(" ", s)
    s = s.lower()
    # collapse doubled letters: `ommi` and `omi` sound the same to a learner
    s = re.sub(r"(.)\1+", r"\1", s)
    return _WS.sub(" ", s).strip()


def key_nospace(s: str) -> str:
    """Same, but ignoring word boundaries — recognisers split the fused article
    (`mill-Awstralja` vs `mill Awstralja`) unpredictably."""
    return key(s).replace(" ", "")


def soft_key(s: str) -> str:
    """Phonetic key with `q` dropped.

    `q` is a glottal stop, and recognisers do not write it as a consonant: across
    334 spoken answers the Maltese wav2vec2 rendered it as `għ` (`qasira` →
    `għasira`), as nothing (`qadima` → `adima`), or as whatever consonant followed
    (`wisq` → `wist`). `għ` is already folded away, so dropping `q` too puts the
    two on the same footing and those three land exactly on their targets.

    It was `q` → `k` before, which is what a speaker of the language would expect
    and not what the recogniser does. Measured over the same 334 answers, dropping
    is better on both sides of the decision: one more correct answer accepted, and
    the highest score any *wrong* sibling achieves falls from 0.909 to 0.900. The
    vowels still carry the distinctions that matter — `qalb` against `kelb` is
    nowhere near the threshold either way.
    """
    return key_nospace(s).replace("q", "")


# ── How wrong is wrong? ────────────────────────────────────────────────────────
#
# `difflib` charges the same for every character it has to change, and that is not
# how a recogniser goes wrong. Aligning the soft keys of all 334 real transcripts
# against the line they were meant to be gives 71 character edits over 4,793
# characters — 21 substitutions, 29 deletions, 21 insertions — and the substitutions
# are the interesting third. Fourteen of the 21 are consonant for consonant, and
# eleven of those fourteen are between *neighbours*: r↔l twice, t↔d twice, d↔t, c↔k,
# c↔t, t↔c, n↔m, y↔h, h↔y. Liquids for liquids, nasals for nasals, a stop for the
# same stop voiced. Almost nothing is a randomly wrong letter.
#
# So a substitution costs what kind of substitution it is: a vowel for another vowel,
# or a consonant for its articulatory neighbour, is about a third of a character, and
# anything else is full price. This is tolerance about *how* a sound was heard and not
# about which sounds were there — the cheap swaps are exactly the ones no available
# Maltese recogniser resolves in a learner's speech.
_VOWEL_SET = frozenset("aeiou")

# Pairs a sound is genuinely mistaken for. The key alphabet is post-`_RULES`, so:
# `c` is ċ (tʃ), `j` is ġ *or* g, `s` is x (ʃ) or s, `z` is ż/z/ts, `y` is j (the
# glide), and `q`/`għ` are already gone.
# Every pair here was *observed* in the 334 transcripts. The list is short on purpose:
# the phonetically obvious additions — ċ against ġ, s against ż, p against b — were
# tried and one of them cost a real distinction. `Jien bil-għatx` (thirsty) heard as
# `jien bilaċ` was credited to `Jien bil-ġuħ` (hungry), because ċ↔ġ at a third of a
# character plus a vowel appearing beat the true line. So a pair earns its place by
# having been made, not by being plausible; `tests/test_q4_recogniser.py` is what
# catches the difference.
#
# `j` is deliberately absent. It is the key for ġ *and* g, so any pair involving it
# drags two sounds along, and that is the one merge that broke a minimal pair.
_NEAR_PAIRS = frozenset({
    ("d", "t"),                 # a stop for the same stop voiced — three times
    ("c", "k"), ("c", "t"),     # ċ for k, ċ for t — the affricate against its stops
    ("l", "r"),                 # liquids, twice
    ("m", "n"),                 # nasals
    ("h", "y"),                 # both nearly nothing, and both what għ leaves behind
    ("u", "w"), ("i", "y"),     # a glide for the vowel it is made of
})

_NEAR_COST = 0.35     # heard as its neighbour
_VOWEL_COST = 0.30    # heard as another vowel


def _swap_cost(x: str, y: str) -> float:
    if x == y:
        return 0.0
    if x in _VOWEL_SET and y in _VOWEL_SET:
        return _VOWEL_COST
    return _NEAR_COST if (min(x, y), max(x, y)) in _NEAR_PAIRS else 1.0


def key_sound_similarity(ka: str, kb: str) -> float:
    """0..1 over two keys already made, charging a substitution by kind.

    Edit distance rather than `difflib`'s longest-common-subsequence ratio, because
    the ratio has no way to say that a change was a small one. A weighted alignment
    does, and that is the whole point.

    A character *appearing or vanishing* costs full price, though the observed
    insertions and deletions were mostly vowels and a discount was the obvious thing
    to try. Two ways of trying it were measured and both were worse. Charging a
    vowel indel a third of a character let `hello` score 0.58 against `Aħna erbgħa`
    — nothing in common but the discount, four times over, on a node where the app
    has to be able to say that nothing was said. Making the discount affine, so only
    the first character of a run is cheap, changed not one number: the alignment
    simply scatters the gaps as single vowels between matches, and every one of them
    opens its own run.

    Nothing is lost by charging full price. Where a vowel really has slipped —
    `nħobb` for `inħobb` — the plain ratio scores it 0.889 and `pair_score` takes
    the larger of the two anyway. The value here is all in the substitutions.
    """
    if not ka and not kb:
        return 1.0
    if not ka or not kb:
        return 0.0
    prev = list(range(len(kb) + 1))
    for i in range(1, len(ka) + 1):
        cur = [float(i)] + [0.0] * len(kb)
        for j in range(1, len(kb) + 1):
            cur[j] = min(
                prev[j - 1] + _swap_cost(ka[i - 1], kb[j - 1]),
                prev[j] + 1.0,
                cur[j - 1] + 1.0,
            )
        prev = cur
    return max(0.0, 1.0 - prev[len(kb)] / max(len(ka), len(kb)))


def sound_similarity(a: str, b: str) -> float:
    """0..1 similarity between two Maltese utterances, by how they sound."""
    return key_sound_similarity(soft_key(a), soft_key(b))


def key_similarity(a: str, b: str) -> float:
    """0..1 similarity between two keys that have already been made.

    `similarity` keys whatever it is given, which is the wrong thing to do to a key:
    a second pass collapses the doubled vowels the first one produced out of silent
    `għ` — `noqgħod` keys to `nood`, and keying that again gives `nod`. The comparison
    then moves without saying so, and `nohod` against `nood` falls from 0.89 to 0.75.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def similarity(a: str, b: str, soft: bool = False) -> float:
    """0..1 phonetic similarity, insensitive to word splits."""
    ka, kb = (soft_key(a), soft_key(b)) if soft else (key_nospace(a), key_nospace(b))
    if not ka and not kb:
        return 1.0
    if not ka or not kb:
        return 0.0
    return difflib.SequenceMatcher(None, ka, kb).ratio()


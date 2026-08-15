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
    """Phonetic key with `q` folded to `k`.

    `q` is a glottal stop, and recognisers render it as h, k or nothing —
    `Noqgħod` comes back as `noħog`, `noħok`, `noħħok`. Folding it costs little in a
    closed set: `qalb` against `kelb` still scores 0.75, well under the accept
    threshold, because the vowels carry the difference.
    """
    return key_nospace(s).replace("q", "k")


def similarity(a: str, b: str, soft: bool = False) -> float:
    """0..1 phonetic similarity, insensitive to word splits."""
    ka, kb = (soft_key(a), soft_key(b)) if soft else (key_nospace(a), key_nospace(b))
    if not ka and not kb:
        return 1.0
    if not ka or not kb:
        return 0.0
    return difflib.SequenceMatcher(None, ka, kb).ratio()


def sounds_like(a: str, b: str, threshold: float = 0.88) -> bool:
    return similarity(a, b) >= threshold

"""Maltese-aware text normalisation, comparison and diffing.

Speech recognisers routinely drop the Maltese diacritics (ġ ħ ż ċ) and mangle `għ`,
which is silent anyway. So grading has to be tolerant of orthography while still
being able to *show* the learner the correct spelling.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

# Diacritic-insensitive fold. Note `għ` → `` because it is silent; this makes
# "għandi" and a recogniser's "andi" compare equal.
_FOLD = str.maketrans({
    "ġ": "g", "Ġ": "g", "ħ": "h", "Ħ": "h",
    "ż": "z", "Ż": "z", "ċ": "c", "Ċ": "c",
    "à": "a", "è": "e", "ì": "i", "ò": "o", "ù": "u",
    "’": "'", "‘": "'", "`": "'", "´": "'",
})

_PUNCT = re.compile(r"[^\w\s'\-]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalise(s: str) -> str:
    """Display-safe normalisation: unicode NFC, tidy whitespace and apostrophes."""
    s = unicodedata.normalize("NFC", s or "").strip()
    s = s.replace("’", "'").replace("‘", "'")
    return _WS.sub(" ", s)


def fold(s: str) -> str:
    """Aggressive fold used only for comparison, never for display."""
    s = normalise(s).lower().translate(_FOLD)
    s = s.replace("gh", "")          # silent għ, and its de-diacriticised form
    s = _PUNCT.sub(" ", s)
    s = s.replace("-", " ").replace("'", "")
    # The article is frequently dropped or misheard; treat it as optional.
    s = re.sub(r"\b(il|l|ic|id|in|ir|is|it|ix|iz)\b\s*", "", s)
    return _WS.sub(" ", s).strip()


def tokens(s: str) -> list[str]:
    return [t for t in normalise(s).split(" ") if t]


def units(s: str) -> list[tuple[str, str]]:
    """Split into (display, comparison-key) pairs.

    Maltese writes the article and fused prepositions onto the next word with a
    hyphen (`mill-Awstralja`, `id-dar`), and recognisers split them about half the
    time. Splitting at the hyphen — while keeping it attached to the left piece for
    display — makes both spellings tokenise identically, so `mill Awstralja` and
    `mill-Awstralja` compare as equal instead of costing a third of the score.
    """
    out: list[tuple[str, str]] = []
    for word in tokens(s):
        parts = word.split("-")
        for i, part in enumerate(parts):
            if not part:
                continue
            display = part + ("-" if i < len(parts) - 1 else "")
            key = fold(part).replace(" ", "")
            out.append((display, key))
    return out


def similarity(a: str, b: str) -> float:
    """0..1 similarity on folded forms."""
    fa, fb = fold(a), fold(b)
    if not fa and not fb:
        return 1.0
    if not fa or not fb:
        return 0.0
    return difflib.SequenceMatcher(None, fa, fb).ratio()


def word_similarity(a: str, b: str) -> float:
    """Token-level F1 on folded words — more forgiving of word order than raw ratio."""
    ta = [k for _, k in units(a) if k]
    tb = [k for _, k in units(b) if k]
    if not ta or not tb:
        return 0.0
    overlap = 0
    pool = list(tb)
    for t in ta:
        match = next((x for x in pool if difflib.SequenceMatcher(None, t, x).ratio() >= 0.8), None)
        if match:
            pool.remove(match)
            overlap += 1
    precision = overlap / len(ta)
    recall = overlap / len(tb)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score(said: str, target: str) -> float:
    """Blended score used to grade a spoken/typed attempt against a target."""
    return round(0.45 * similarity(said, target) + 0.55 * word_similarity(said, target), 4)


def diff_words(said: str, target: str) -> list[dict]:
    """Word-level diff for the correction card.

    Returns a list of {op, said, target} where op is equal|sub|del|ins.
    `del` = the learner said something extra, `ins` = they left something out.
    """
    ua, ub = units(said), units(target)
    a, fa = [d for d, _ in ua], [k for _, k in ua]
    b, fb = [d for d, _ in ub], [k for _, k in ub]
    out: list[dict] = []
    matcher = difflib.SequenceMatcher(None, fa, fb)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                out.append({"op": "equal", "said": a[i1 + k], "target": b[j1 + k]})
        elif op == "replace":
            span = max(i2 - i1, j2 - j1)
            for k in range(span):
                out.append({
                    "op": "sub",
                    "said": a[i1 + k] if i1 + k < i2 else "",
                    "target": b[j1 + k] if j1 + k < j2 else "",
                })
        elif op == "delete":
            for k in range(i1, i2):
                out.append({"op": "del", "said": a[k], "target": ""})
        elif op == "insert":
            for k in range(j1, j2):
                out.append({"op": "ins", "said": "", "target": b[k]})
    return out


# ── Definite-article assimilation, used for hints and for generating drills ──

SUN_LETTERS = set("ċdnrstxzżĊDNRSTXZŻ")
VOWELS = set("aeiouàèìòùAEIOU")


def definite(word: str) -> str:
    """Attach the definite article to a Maltese noun, with assimilation."""
    w = normalise(word)
    if not w:
        return w
    first = w[0]
    if first in VOWELS or first in "hHħĦ" or w[:2].lower() == "għ":
        return f"l-{w}"
    # Romance loans with "s impura" (s + voiceless stop) take a prosthetic i-,
    # which also blocks sun-letter assimilation: skola → l-iskola, not *is-skola.
    # Native clusters do not: żmien → iż-żmien, snien → is-snien.
    if len(w) > 1 and first.lower() == "s" and w[1].lower() in "ptkcq":
        return f"l-i{w}"
    if first in SUN_LETTERS:
        return f"i{first.lower()}-{w}"
    return f"il-{w}"


PREP_FUSION = {
    "fi": "fil-", "f'": "fil-", "bi": "bil-", "b'": "bil-", "ta'": "tal-",
    "minn": "mill-", "ma'": "mal-", "għal": "għall-", "sa": "sal-", "lil": "lill-",
}

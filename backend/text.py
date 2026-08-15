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


# ── Language classification ────────────────────────────────────────────────
#
# The tutor contract says reply_mt is Maltese and reply_en is English. A weak model
# breaks that constantly — answering in English, or filling the fields the wrong way
# round. Rather than trusting it, every turn is classified on the way out.

# ħ ġ ż ċ and the digraph għ are near-unique to Maltese orthography.
_MT_CHARS = re.compile(r"[ħĦġĠżŻċĊ]|għ", re.I)

_EN_MARKERS = {
    "the", "and", "you", "your", "what", "who", "where", "when", "how", "why",
    "is", "are", "was", "were", "do", "does", "did", "have", "has", "had",
    "with", "for", "from", "that", "this", "would", "will", "can", "could",
    "want", "hello", "good", "morning", "please", "thanks", "thank", "sorry",
    "mean", "means", "say", "said", "tell", "like", "there", "here", "about",
    "name", "person", "i", "am", "my", "we", "they", "she", "her", "his",
    "of", "to", "it", "be", "been", "at", "on", "by", "or", "but", "if",
    "not", "very", "much", "many", "need", "should", "must", "let", "get",
    "going", "come", "see", "know", "think", "time", "day", "yes", "please",
}

# High-frequency inflected forms the decks store only in a base form, plus the
# pronoun set — without these a short, perfectly good Maltese sentence like
# "Sarah huwa ismek. U inti?" scores as unknown.
_MT_EXTRA = {
    "jien", "jiena", "int", "inti", "hu", "huwa", "hi", "hija", "aħna", "intom",
    "huma", "ismi", "ismek", "ismu", "isimha", "jisimni", "jismek", "jisimhom",
    "tiegħi", "tiegħek", "tiegħu", "tagħha", "tagħna", "tagħkom", "tagħhom",
    "għandi", "għandek", "għandu", "għandha", "għandna", "għandkom", "għandhom",
    "mhux", "mhix", "hemm", "hawn", "issa", "illum", "għada", "mela", "iva",
    "grazzi", "bonġu", "saħħa", "jekk", "jogħġbok", "kif", "fejn", "kemm",
    "liema", "għaliex", "għax", "biex", "imma", "jew", "wkoll", "ukoll",
}

_MT_MARKERS_CACHE: set[str] | None = None


def maltese_markers() -> set[str]:
    """Maltese vocabulary drawn from the app's own decks, so it grows with them."""
    global _MT_MARKERS_CACHE
    if _MT_MARKERS_CACHE is not None:
        return _MT_MARKERS_CACHE
    from . import curriculum

    # Words that are also ordinary English/Italian would fire on the wrong language.
    ambiguous = {"u", "le", "ma", "no", "si", "e", "a", "in", "me", "te", "la", "il"}
    markers: set[str] = set()
    try:
        rows = (curriculum._read_tsv(curriculum.VOCAB_TSV)
                + curriculum._read_tsv(curriculum.PHRASES_TSV))
    except Exception:  # noqa: BLE001 — classification must never break a turn
        rows = []
    for row in rows:
        for word in re.split(r"[\s\-]+", (row.get("mt") or "").lower()):
            word = word.strip(".,!?;:'’\"")
            if len(word) >= 2 and word not in ambiguous:
                markers.add(word)
    markers |= {"mill", "fil", "tal", "bil", "sal", "mal", "għall", "lill", "fl", "bl"}
    markers |= _MT_EXTRA
    _MT_MARKERS_CACHE = markers
    return markers


def _words(s: str) -> set[str]:
    return {w.strip(".,!?;:'’\"()").lower() for w in (s or "").split()}


def looks_maltese(s: str) -> bool:
    w = _words(s)
    hits = len(w & maltese_markers())
    return hits >= 2 or (hits >= 1 and bool(_MT_CHARS.search(s or "")))


def looks_english(s: str) -> bool:
    w = _words(s)
    en = len(w & _EN_MARKERS)
    return en >= 2 and en > len(w & maltese_markers())


_PARENTHETICAL = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]\s*$")


def strip_translation(s: str) -> str:
    """Drop a trailing parenthetical gloss — models like to append `(Xi trid tgħid?)`
    to an English sentence, which belongs in reply_en, not reply_mt."""
    return _PARENTHETICAL.sub("", normalise(s)).strip()


PREP_FUSION = {
    "fi": "fil-", "f'": "fil-", "bi": "bil-", "b'": "bil-", "ta'": "tal-",
    "ta": "tal-", "minn": "mill-", "ma'": "mal-", "għal": "għall-",
    "sa": "sal-", "lil": "lill-",
}

# Maltese prepositions must fuse with a following definite article:
#   minn + il-Belt → mill-Belt ;  fi + l-Awstralja → fl-Awstralja
# Written unfused, they are simply wrong. This is the single most common error an
# English speaker makes, and — unlike most grammar — it is fully mechanical, so it
# can be checked without a model. See `lint_fusion`.
_FUSION_RE = re.compile(
    r"\b(minn|fi|f'|bi|b'|ta'|ta|ma'|għal|sa|lil)\s+(l-|il-|i[ċdnrstxzż]-)",
    re.IGNORECASE,
)

# Before a vowel the article reduces to l-, and some fused forms lose their vowel
# with it: fi + l-Awstralja → fl-Awstralja, not *fil-Awstralja.
_FUSION_BEFORE_VOWEL = {
    "minn": "mill-", "fi": "fl-", "f'": "fl-", "bi": "bl-", "b'": "bl-",
    "ta'": "tal-", "ta": "tal-", "ma'": "mal-", "għal": "għall-",
    "sa": "sal-", "lil": "lill-",
}

# Before an assimilated article (is-, id-, ix-…) the preposition keeps its stem and
# takes the doubled consonant: minn + is-sena → mis-sena, fi + id-dar → fid-dar.
_FUSION_STEM = {
    "minn": "mi", "fi": "fi", "f'": "fi", "bi": "bi", "b'": "bi",
    "ta'": "ta", "ta": "ta", "ma'": "ma", "għal": "għa", "sa": "sa", "lil": "li",
}


def lint_fusion(s: str) -> list[dict]:
    """Find unfused preposition + article sequences.

    Returns [{"found", "should_be", "why"}]. Used as a safety net over tutor output:
    a weaker model will sometimes hand back a "corrected" sentence that still
    contains this error, and shipping that to a learner is worse than not correcting
    at all.
    """
    out: list[dict] = []
    for m in _FUSION_RE.finditer(normalise(s)):
        prep_raw, article = m.group(1), m.group(2)
        prep = prep_raw.lower()
        low = article.lower()
        if low == "l-":
            fused = _FUSION_BEFORE_VOWEL.get(prep, "")
        elif low == "il-":
            fused = PREP_FUSION.get(prep, "")
        else:
            # assimilated article: minn + is- → mis-, fi + id- → fid-
            stem = _FUSION_STEM.get(prep, "")
            fused = f"{stem}{article[1]}-" if stem else ""
        if not fused:
            continue
        rest = m.group(0)[len(prep_raw):].lstrip()
        out.append({
            "found": m.group(0),
            "should_be": fused,
            "why": f"{prep} fuses with the article {rest.rstrip('-')}- to make {fused}",
        })
    return out


def apply_fusion(s: str) -> str:
    """Rewrite unfused sequences into their correct fused forms."""
    def _sub(m: re.Match) -> str:
        fixes = lint_fusion(m.group(0))
        return fixes[0]["should_be"] if fixes else m.group(0)

    return _FUSION_RE.sub(_sub, normalise(s))

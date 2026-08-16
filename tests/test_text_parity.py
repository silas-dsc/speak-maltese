"""The JavaScript grader must agree with the Python one, string for string.

`frontend/text.js` is a hand translation of `backend/text.py` and
`backend/phonetics.py`, so the app can grade with no server. Two implementations
of one algorithm drift silently, and the symptom here is a learner marked wrong
for a sentence they said correctly — or, worse, marked right for one they didn't.

The hard part is `difflib.SequenceMatcher.ratio()`. It is not edit distance and
not an LCS ratio: it is 2*M/T over the blocks Python's recursive longest-match
finds, and every threshold in this app (0.86 to accept a dialogue answer, 0.8
inside word matching, 0.95/0.78/0.55 for grades) was tuned against that exact
function. A plausible substitute moves every decision boundary at once, so the
algorithm is reimplemented rather than approximated, and checked here.

The inputs are the real ones: all 334 recorded 4-bit transcripts, each against the
line it was meant to be.
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

from backend import phonetics, text  # noqa: E402
from backend.main import _assess  # noqa: E402

node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node is not installed")

FIXTURE = ROOT / "tests" / "fixtures" / "q4_transcripts.tsv"

EXTRA = [
    ("", ""),
    ("Bonġu!", "Bonġu!"),
    ("jien mill awstralja", "Jien mill-Awstralja."),
    ("noqgħod fit tas-sliema", "Noqgħod Tas-Sliema."),
    ("ma nifhimx", "Ma nifhimx."),
    ("iż-żmien", "iz zmien"),
    ("qalb", "kelb"),
    ("għasira wist", "Qasira wisq."),
    ("x'inhu l-prezz", "X'inhu l-prezz?"),
    ("Ta' Marija", "ta marija"),
    ("kif tgħid cheese bil-malti", "Kif tgħid 'cheese' bil-Malti?"),
    ("birrakisħa jekk jogħġbok", "Birra kiesħa, jekk jogħġbok."),
    ("xi xi xi", "Grazzi ħafna."),
]

DRIVER = r"""
import { fold, normalise, score, similarity, wordSimilarity, softKey, keyNospace,
         phoneticSimilarity, assess, diffWords } from '../frontend/text.js';
const pairs = JSON.parse(process.argv[2]);
console.log(JSON.stringify(pairs.map(([said, target]) => ({
  foldSaid: fold(said),
  foldTarget: fold(target),
  normalised: normalise(said),
  score: score(said, target),
  similarity: similarity(said, target),
  wordSimilarity: wordSimilarity(said, target),
  softKey: softKey(said),
  keyNospace: keyNospace(said),
  phonSoft: phoneticSimilarity(said, target, true),
  phonHard: phoneticSimilarity(said, target, false),
  assess: (({ score, grade, verdict }) => ({ score, grade, verdict }))(assess(said, target)),
  diff: diffWords(said, target).map((d) => `${d.op}:${d.said}|${d.target}`),
}))));
"""


def pairs() -> list[tuple[str, str]]:
    out = list(EXTRA)
    for line in FIXTURE.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        _d, _n, expected, *heard = line.split("\t")
        out.append((heard[0] if heard else "", expected))
    return out


@pytest.fixture(scope="module")
def both():
    data = pairs()
    driver = ROOT / "tests" / "_text_driver.mjs"
    driver.write_text(DRIVER, encoding="utf-8")
    try:
        proc = subprocess.run([node, str(driver), json.dumps(data, ensure_ascii=False)],
                              cwd=ROOT, capture_output=True, text=True, timeout=120)
    finally:
        driver.unlink(missing_ok=True)
    if proc.returncode != 0:
        pytest.fail(f"node driver failed:\n{proc.stderr}")
    js = json.loads(proc.stdout)

    py = []
    for said, target in data:
        a = _assess(said, target)
        py.append({
            "foldSaid": text.fold(said),
            "foldTarget": text.fold(target),
            "normalised": text.normalise(said),
            "score": text.score(said, target),
            "similarity": text.similarity(said, target),
            "wordSimilarity": text.word_similarity(said, target),
            "softKey": phonetics.soft_key(said),
            "keyNospace": phonetics.key_nospace(said),
            "phonSoft": phonetics.similarity(said, target, soft=True),
            "phonHard": phonetics.similarity(said, target, soft=False),
            "assess": {k: a[k] for k in ("score", "grade", "verdict")},
            "diff": [f"{d['op']}:{d['said']}|{d['target']}" for d in text.diff_words(said, target)],
        })
    return data, js, py


@pytest.mark.parametrize("field", ["foldSaid", "foldTarget", "normalised",
                                   "softKey", "keyNospace"])
def test_string_transforms_match(both, field):
    data, js, py = both
    bad = [(data[i], js[i][field], py[i][field])
           for i in range(len(data)) if js[i][field] != py[i][field]]
    assert not bad, f"{len(bad)} differ, first: {bad[0]}"


@pytest.mark.parametrize("field", ["score", "similarity", "wordSimilarity",
                                   "phonSoft", "phonHard"])
def test_scores_match(both, field):
    """Exactly, not approximately. These feed thresholds; a 1e-6 drift is a
    different verdict for anything sitting on a boundary."""
    data, js, py = both
    bad = [(data[i], js[i][field], py[i][field])
           for i in range(len(data)) if abs(js[i][field] - py[i][field]) > 1e-9]
    assert not bad, f"{len(bad)} differ, first: {bad[0]}"


def test_grades_and_verdicts_match(both):
    data, js, py = both
    bad = [(data[i], js[i]["assess"], py[i]["assess"])
           for i in range(len(data)) if js[i]["assess"] != py[i]["assess"]]
    assert not bad, f"{len(bad)} differ, first: {bad[0]}"


def test_diffs_match(both):
    """The diff is what the learner sees highlighted on the correction card."""
    data, js, py = both
    bad = [(data[i], js[i]["diff"], py[i]["diff"])
           for i in range(len(data)) if js[i]["diff"] != py[i]["diff"]]
    assert not bad, f"{len(bad)} differ, first: {bad[0]}"

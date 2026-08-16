"""Can the grader live with the 4-bit recogniser?

`tests/fixtures/q4_transcripts.tsv` is not synthetic. Every one of the 334 accepted
answers in the scripted scenes was rendered to speech, played through the 4-bit
(MatMulNBits) wav2vec2 export running in a browser on WebGPU, and the transcript
recorded next to the line it was meant to be. So this replays a real recogniser's
real output against the real grader, with no model in the test.

Why it matters: 4-bit is what makes the app deployable without a server — 201MB at
30x realtime, against 631MB for fp16 — but it is the first quantization that
actually costs accuracy. The question is not "is the WER higher" (it is, slightly)
but "does the learner get stuck", and those are different questions. A recogniser
error that lands inside the phonetic tolerance costs nothing; one that pushes a
correct answer under the threshold costs a retry on a line the learner said right.

Two fixes came out of running this, both in the fixture:

* `q` is dropped rather than folded to `k` in the soft key. The recogniser never
  writes the glottal stop as a consonant — `qasira` came back as `għasira`,
  `qadima` as `adima`, `wisq` as `wist`.
* A quoted foreign word (`Kif tgħid 'cheese' bil-Malti?`) is scored on its Maltese
  frame only. A Maltese model has never seen `cheese` and cannot transcribe it, so
  grading the whole sentence punished the learner for the one word they were asked
  to supply.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import dialogue, text  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "q4_transcripts.tsv"


def rows() -> list[dict]:
    out = []
    for line in FIXTURE.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        d, n, expected, *heard = line.split("\t")
        out.append({"dialogue": d, "node": n, "expected": expected,
                    "heard": heard[0] if heard else ""})
    return out


def graded() -> list[tuple[dict, dict]]:
    return [(r, dialogue.evaluate(r["dialogue"], r["node"], r["heard"], attempts=0))
            for r in rows()]


def test_fixture_covers_every_accepted_answer():
    """If a scene gains an answer, this fixture is stale and the coverage claim
    below is measuring less than it says."""
    have = {(r["dialogue"], r["node"], text.fold(r["expected"])) for r in rows()}
    want = {(d["id"], nid, text.fold(a["mt"]))
            for d in dialogue.all_dialogues()
            for nid, n in d["nodes"].items() if not n.get("free")
            for a in n.get("accept", [])}
    missing = want - have
    assert not missing, (
        f"{len(missing)} accepted answers have no recorded transcript — "
        f"re-run scripts/bench_onnx.py's browser harness. First: {sorted(missing)[:3]}")


def test_the_4bit_recogniser_clears_the_grader():
    """Every answer the recogniser heard should be graded correct *and* matched to
    the line it actually was — matching a sibling would credit the learner for a
    sentence they did not say and schedule the wrong card."""
    wrong = [(r, g) for r, g in graded()
             if g["verdict"] != "correct"
             or text.fold(g["matched_mt"] or "") != text.fold(r["expected"])]
    rate = 1 - len(wrong) / len(rows())
    detail = "\n".join(
        f"  [{g['verdict']} {g['score']}] want {r['expected']!r} heard {r['heard']!r}"
        for r, g in wrong)
    assert rate >= 0.99, f"only {rate:.1%} of answers graded correct:\n{detail}"


def test_no_answer_is_credited_to_a_sibling():
    """The failure that matters most is silent: a near-miss that scores highest
    against a *different* accepted answer still advances the scene, but teaches and
    schedules the wrong sentence."""
    swapped = [(r, g) for r, g in graded()
               if g["verdict"] == "correct"
               and text.fold(g["matched_mt"] or "") != text.fold(r["expected"])]
    assert not swapped, "\n".join(
        f"  heard {r['heard']!r} for {r['expected']!r} → credited {g['matched_mt']!r}"
        for r, g in swapped)


def test_the_glottal_stop_survives_being_dropped():
    """Dropping `q` is what let three real transcripts through. It must not also let
    genuinely different words through — the vowels have to carry the distinction."""
    from backend import phonetics

    for heard, want in (("għasira", "Qasira"), ("adima", "qadima"), ("wist", "wisq")):
        assert phonetics.similarity(heard, want, soft=True) >= 0.85, (heard, want)
    # still separable
    assert phonetics.similarity("kelb", "qalb", soft=True) < 0.8


def test_a_quoted_foreign_word_is_not_graded():
    """`cheese` is the word the learner is asked to supply and the one word a
    Maltese recogniser cannot write. The frame around it still has to be right."""
    r = dialogue.evaluate("stuck", "s3", "kif tgħidx xi s bil-malti")
    assert r["verdict"] == "correct", r["score"]
    # ...but the frame is not a free pass
    assert dialogue.evaluate("stuck", "s3", "xi xi xi")["verdict"] == "wrong"

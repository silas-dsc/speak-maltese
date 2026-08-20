"""The field a spoken answer is ranked against, and the scale the margin is measured in.

Two staged changes live here, both switched off, and both of a kind that would be easy to
turn on by accident and hard to notice afterwards. `MARGIN_SIGMAS` scales the required
lead by how spread out the field's own scores are; `FIELD_LOCAL` draws part of the field
from the scene being spoken rather than uniformly from the whole script. Either one
changes which answers are accepted, and a grader that has quietly become stricter marks
correct answers wrong and feeds that into the review scheduler.

What can be checked without the learner's recordings is the arithmetic and the wiring:
that the spread is a spread, that it is computed in a way that survives a tightly
clustered field, and that the scene's own answers are the same set on both sides of the
port. The accept-rate question needs the clips, and the constants stay at zero until it
is answered.
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
import { readFileSync } from 'node:fs';
import { spread } from '../frontend/nanostt.js';
import * as dialogue from '../frontend/dialogue.js';

dialogue.load(JSON.parse(readFileSync(process.argv[2], 'utf-8')));

const tight = Array.from({ length: 24 }, (_, i) => 1e6 + i * 1e-6);

console.log(JSON.stringify({
  empty: spread([]),
  single: spread([1]),
  flat: spread([1, 1, 1, 1]),
  known: spread([2, 4, 4, 4, 5, 5, 7, 9]),
  tight: spread(tight),
  answers: Object.fromEntries(
    dialogue.all().map((d) => [d.id, dialogue.answersIn(d.id).sort()])),
  unknown: dialogue.answersIn('no-such-scene'),
}));
"""


@pytest.fixture(scope="module")
def js():
    data = ROOT / "data" / "dialogues.json"
    if not data.exists():
        pytest.skip("no dialogues.json")
    driver = ROOT / "tests" / "_field_driver.mjs"
    driver.write_text(DRIVER, encoding="utf-8")
    try:
        proc = subprocess.run([node, str(driver), str(data)], cwd=ROOT,
                              capture_output=True, text=True, timeout=120)
    finally:
        driver.unlink(missing_ok=True)
    if proc.returncode != 0:
        pytest.fail(f"node driver failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


def test_the_spread_is_a_sample_standard_deviation(js):
    assert js["empty"] == 0
    assert js["single"] == 0, "one alternative gives no scale, and must not give a fake one"
    assert js["flat"] == 0
    # Textbook figure for this set, with the n-1 divisor.
    assert js["known"] == pytest.approx(2.13808993, rel=1e-6)


def test_the_spread_survives_a_tightly_clustered_field(js):
    """The scores of 24 near-identical hypotheses cluster hard, and that is exactly where
    the `sum(x^2) - sum(x)^2/n` shortcut subtracts two nearly equal numbers and returns
    something negative — a NaN once it reaches the square root, and a margin that then
    silently compares against NaN and accepts everything."""
    assert js["tight"] > 0
    assert js["tight"] == pytest.approx(7.0710678e-6, rel=1e-3)


def test_the_scene_answers_match_the_python_engine(js):
    """`answersIn` decides the field under `FIELD_LOCAL`, so the two engines have to agree
    on it or the sweep prices a field the app does not use."""
    from backend import dialogue as py

    got = js["answers"]
    assert got, "no dialogues came back from the client engine"
    for scene_id, lines in got.items():
        assert lines == sorted(py.answers_in(scene_id)), f"{scene_id} differs"


def test_an_unknown_scene_is_empty_on_both_sides(js):
    from backend import dialogue as py

    assert js["unknown"] == []
    assert py.answers_in("no-such-scene") == []


def test_the_scene_answers_are_a_subset_of_the_whole_script(js):
    """A local draw tops up from the global pool, so anything local has to be in it —
    otherwise the field could contain a line the script never accepts."""
    from backend import dialogue as py

    every = set(py.every_line())
    for scene_id, lines in js["answers"].items():
        missing = [line for line in lines if line not in every]
        assert not missing, f"{scene_id} offers lines outside the script: {missing}"


def test_a_scene_field_is_too_thin_to_stand_alone(js):
    """The reason `FIELD_LOCAL` tops up rather than replaces. A scene holds a dozen or two
    answers, so a field drawn only from one would be smaller than `RANK_AGAINST` — and
    ranking against fewer alternatives is a different change wearing the same clothes."""
    sizes = [len(v) for v in js["answers"].values()]
    assert sizes
    assert max(sizes) < 24, (
        "a scene now holds a full field; the top-up reasoning needs revisiting")

"""The JavaScript scheduler must agree with the Python one, review for review.

`frontend/srs.js` is a hand translation of `backend/srs.py`. Two implementations of
the same algorithm drift silently: a transposed weight or a different rounding rule
produces intervals that are plausible, wrong, and invisible until someone notices
their reviews are coming back at the wrong time — by which point the history that
would show it is the thing that was scheduled badly.

So both are run over the same review sequences and compared. Python keeps the role
of reference implementation even though it no longer schedules anything.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import srs  # noqa: E402

node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node is not installed")

# Grades chosen to walk every branch: the learning steps, an early lapse, the
# relearning path, same-day repeats, and a long gap that drops retrievability.
SEQUENCES = [
    [(3, 0), (3, 0), (3, 1), (3, 10), (3, 30)],
    [(1, 0), (3, 0), (3, 1), (1, 5), (3, 0), (3, 3)],
    [(4, 0), (4, 30), (4, 200)],
    [(2, 0), (2, 0), (2, 1), (2, 8), (2, 40)],
    [(3, 0), (4, 0), (2, 2), (1, 1), (3, 0), (4, 14), (3, 400)],
    [(1, 0), (1, 0), (1, 0), (3, 0), (3, 1)],
]

START = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)

DRIVER = """
import { review, preview, intervalFor, humanise } from '../frontend/srs.js';
const seqs = JSON.parse(process.argv[2]);
const start = Number(process.argv[3]);
const out = [];
for (const seq of seqs) {
  let card = {
    stability: 0, difficulty: 0, reps: 0, lapses: 0,
    state: 'new', step: 0, due: null, lastReview: null,
  };
  const steps = [];
  for (const [grade, days] of seq) {
    const at = start + days * 86400000;
    card = review(card, grade, 0.9, at);
    steps.push({
      state: card.state, reps: card.reps, lapses: card.lapses, step: card.step,
      stability: card.stability, difficulty: card.difficulty,
      dueSec: Math.round((card.due - at) / 1000),
      preview: preview(card, 0.9, at),
    });
  }
  out.push(steps);
}
console.log(JSON.stringify({
  runs: out,
  intervals: [0.01, 1, 3.5, 10, 47.5, 365, 100000].map((s) => intervalFor(s, 0.9)),
  humanised: [30, 3600, 7200, 86400, 86400 * 15, 86400 * 90, 86400 * 800].map(humanise),
}));
"""


def _js() -> dict:
    driver = ROOT / "tests" / "_srs_driver.mjs"
    driver.write_text(DRIVER, encoding="utf-8")
    try:
        proc = subprocess.run(
            [node, str(driver), json.dumps(SEQUENCES), str(int(START.timestamp() * 1000))],
            cwd=ROOT, capture_output=True, text=True, timeout=60,
        )
    finally:
        driver.unlink(missing_ok=True)
    if proc.returncode != 0:
        pytest.fail(f"node driver failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


def _python() -> dict:
    runs = []
    for seq in SEQUENCES:
        card = srs.CardState()
        steps = []
        for grade, days in seq:
            at = START + timedelta(days=days)
            card = srs.review(card, grade, 0.9, at)
            steps.append({
                "state": card.state, "reps": card.reps, "lapses": card.lapses,
                "step": card.step, "stability": card.stability,
                "difficulty": card.difficulty,
                "dueSec": round((card.due - at).total_seconds()),
                "preview": {str(k): v for k, v in srs.preview(card, 0.9, at).items()},
            })
        runs.append(steps)
    return {
        "runs": runs,
        "intervals": [srs.interval_for(s, 0.9)
                      for s in (0.01, 1, 3.5, 10, 47.5, 365, 100000)],
        "humanised": [srs._humanise(s) for s in
                      (30, 3600, 7200, 86400, 86400 * 15, 86400 * 90, 86400 * 800)],
    }


@pytest.fixture(scope="module")
def both():
    return _js(), _python()


def test_interval_table_matches(both):
    js, py = both
    assert js["intervals"] == py["intervals"]


def test_humanised_labels_match(both):
    """These are what the grade buttons show. A mismatch here is the learner being
    promised '3d' and getting four."""
    js, py = both
    assert js["humanised"] == py["humanised"]


@pytest.mark.parametrize("i", range(len(SEQUENCES)))
def test_review_sequence_matches(both, i):
    js, py = both
    jrun, prun = js["runs"][i], py["runs"][i]
    assert len(jrun) == len(prun)
    for n, (a, b) in enumerate(zip(jrun, prun)):
        where = f"sequence {i}, review {n} (grade {SEQUENCES[i][n][0]})"
        assert a["state"] == b["state"], where
        assert (a["reps"], a["lapses"], a["step"]) == (b["reps"], b["lapses"], b["step"]), where
        assert a["stability"] == pytest.approx(b["stability"], rel=1e-9), where
        assert a["difficulty"] == pytest.approx(b["difficulty"], rel=1e-9), where
        # The due date is what the learner actually experiences.
        assert a["dueSec"] == b["dueSec"], where
        assert a["preview"] == b["preview"], where

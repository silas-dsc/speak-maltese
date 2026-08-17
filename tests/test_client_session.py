"""The conversation, kept across a reload.

It used to live only in the DOM, so a reload restarted the scene from its first
line — and on a phone the browser reloads a backgrounded tab on its own schedule,
so that happened to learners who had not asked for it. The app also reloads itself
now when a new build takes over, which made the loss something the app was doing
to them rather than something the browser did.

The store is small and its failure modes are all storage failures, which is why
they are tested here rather than in a browser: a full quota, a private window that
refuses to write, and half-written data from an older build must each degrade to
"no saved conversation" instead of taking the turn down with them.
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
import { store } from '../frontend/session.js';

/* A Storage-shaped stub. `mode` makes it misbehave the way a real one does. */
function fake(initial = {}, mode = 'ok') {
  const map = new Map(Object.entries(initial));
  return {
    map,
    getItem: (k) => (mode === 'unreadable' ? (() => { throw new Error('denied'); })()
      : (map.has(k) ? map.get(k) : null)),
    setItem: (k, v) => {
      if (mode === 'full') throw new Error('QuotaExceededError');
      map.set(k, v);
    },
    removeItem: (k) => { map.delete(k); },
  };
}

const turn = (i) => ({ role: i % 2 ? 'user' : 'tutor', mt: `line ${i}` });
const session = (turns) => ({
  dialogue: 'greet', node: 'g2', attempts: 1,
  run: { first: 1, retried: 0, movedOn: 0, learned: [], startedAt: 1 },
  present: { node: 'g2', say_mt: 'Minn fejn int?', frames: ['Minn …'] },
  turns,
});

const out = {};

// ── the round trip: what went in is what comes back ────────────────────────
const s1 = fake();
const c1 = store(s1);
c1.save(session([turn(0), turn(1)]));
const back = c1.load();
out.roundTrip = JSON.stringify(back.turns) === JSON.stringify([turn(0), turn(1)])
  && back.node === 'g2' && back.attempts === 1
  && back.present.frames[0] === 'Minn …';

// ── a long conversation is bounded, and it is the oldest turns that go ──────
const many = Array.from({ length: 400 }, (_, i) => turn(i));
c1.save(session(many));
const trimmed = c1.load().turns;
out.capped = trimmed.length < many.length;
out.keptTheEnd = trimmed[trimmed.length - 1].mt === 'line 399';
out.droppedTheStart = trimmed[0].mt !== 'line 0';

// ── clearing means clearing ────────────────────────────────────────────────
c1.clear();
out.cleared = c1.load() === null;

// ── nothing usable is nothing, never a half-restored conversation ──────────
out.empty = store(fake()).load() === null;
out.corrupt = store(fake({ 'sm.drillSession': '{oh no' })).load() === null;
out.oldShape = store(fake({ 'sm.drillSession': JSON.stringify({ v: 0, dialogue: 'greet' }) }))
  .load() === null;
out.noPresent = store(fake({ 'sm.drillSession': JSON.stringify({ v: 1, dialogue: 'greet' }) }))
  .load() === null;

// ── storage that refuses to work must not throw into the turn ──────────────
const full = store(fake({}, 'full'));
full.save(session([turn(0)]));            // throws inside, swallowed
out.fullSurvives = full.load() === null;
out.unreadableSurvives = store(fake({}, 'unreadable')).load() === null;

console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def result():
    driver = ROOT / "tests" / "_session_driver.mjs"
    driver.write_text(DRIVER, encoding="utf-8")
    try:
        proc = subprocess.run([node, str(driver)], cwd=ROOT,
                              capture_output=True, text=True, timeout=30)
    finally:
        driver.unlink(missing_ok=True)
    if proc.returncode != 0:
        pytest.fail(f"node driver failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


def test_a_conversation_survives_a_reload(result):
    assert result["roundTrip"], "what was saved is not what came back"


def test_a_long_conversation_is_bounded_from_the_front(result):
    """A learner can retry one line all afternoon. The cap keeps the write from
    failing; dropping the oldest turns keeps the part they are looking at."""
    assert result["capped"]
    assert result["keptTheEnd"]
    assert result["droppedTheStart"]


def test_clearing_leaves_nothing_to_restore(result):
    assert result["cleared"]


@pytest.mark.parametrize("case", ["empty", "corrupt", "oldShape", "noPresent"])
def test_unusable_data_restores_nothing(result, case):
    """Half a conversation is worse than none: the composer would be pointed at a
    node the transcript never reached. Anything not recognisable is dropped."""
    assert result[case], f"{case} should restore nothing"


@pytest.mark.parametrize("case", ["fullSurvives", "unreadableSurvives"])
def test_storage_failure_never_reaches_the_learner(result, case):
    """A full quota or a locked-down private window costs the conversation its
    memory, and nothing else — the turn on screen must still finish."""
    assert result[case]

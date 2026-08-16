"""The scheduling rules, now that they run in the browser.

These are the same properties the Python tests held before the schedule moved to
IndexedDB — the daily new-card window, the streak, and card ids that do not
collide. Two of them exist because the server got them wrong:

* the new-card budget compared an ISO timestamp against SQLite's own format, and
  'T' sorts above ' ', so yesterday's reviews counted as today's and the learner
  was handed nothing new;
* the streak walked back from the local date over UTC-stamped rows, so an early
  morning review east of Greenwich landed on the previous day and read as zero.

The JavaScript is written to avoid both, and this is what says so.
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
import { slug, interleave, streak, localDay, newIntroducedSince, countSince }
  from '../frontend/schedule.js';

const DAY = 86400000;
const now = Date.now();
const out = {};

// ── new-card budget: a 25-hour-old review is not "today" ───────────────────
const live = [{ id: 'a', reps: 1 }, { id: 'b', reps: 1 }, { id: 'c', reps: 9 }];
const reviews = [
  { cardId: 'a', at: now - 2 * 3600000 },     // 2h ago  → counts
  { cardId: 'b', at: now - 25 * 3600000 },    // 25h ago → does not
  { cardId: 'c', at: now - 1 * 3600000 },     // not a young card
];
out.introduced = newIntroducedSince(reviews, live, now - DAY);
out.doneToday = countSince(reviews, now - DAY);

// ── streak walks the same local days the reviews were stamped with ─────────
const day = (n) => localDay(now - n * DAY);
out.streakToday = streak(new Set([day(0), day(1), day(2)]));
out.streakGap = streak(new Set([day(1), day(2)]));
out.streakNone = streak(new Set());

// ── ids: long phrases sharing a prefix must not collapse ───────────────────
const stem = 'Kollox tajjeb ħafna, grazzi. Il-kont, jekk jogħġbok, u ';
out.slugA = slug(stem + 'ilma');
out.slugB = slug(stem + 'ħobż');
out.slugShort = slug('Bonġu, kif int?');
out.slugStable = slug(stem + 'ilma') === slug(stem + 'ilma');

// ── interleaving spreads new cards rather than blocking them ───────────────
const due = Array.from({ length: 9 }, (_, i) => ({ id: `d${i}` }));
const fresh = Array.from({ length: 3 }, (_, i) => ({ id: `n${i}` }));
out.mixed = interleave(due, fresh).map((c) => c.id);
out.noneFresh = interleave(due, []).length;
out.noneDue = interleave([], fresh).length;

console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def js():
    driver = ROOT / "tests" / "_schedule_driver.mjs"
    driver.write_text(DRIVER, encoding="utf-8")
    try:
        proc = subprocess.run([node, str(driver)], cwd=ROOT,
                              capture_output=True, text=True, timeout=60)
    finally:
        driver.unlink(missing_ok=True)
    if proc.returncode != 0:
        pytest.fail(f"node driver failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


def test_new_card_budget_excludes_yesterdays_reviews(js):
    """`a` was two hours ago and counts; `b` was twenty-five and does not; `c` is
    past its first few reps so it is not a new card at all."""
    assert js["introduced"] == 1
    assert js["doneToday"] == 2


def test_streak_counts_consecutive_local_days(js):
    assert js["streakToday"] == 3
    assert js["streakGap"] == 0, "no review today ends the streak"
    assert js["streakNone"] == 0


def test_long_phrases_get_distinct_ids(js):
    assert js["slugA"] != js["slugB"]
    assert len(js["slugA"]) <= 48 and len(js["slugB"]) <= 48
    assert js["slugStable"], "the same phrase must always produce the same id"
    assert js["slugShort"] == "bonġu--kif-int", "short ids stay readable and unhashed"


def test_new_cards_are_spread_through_the_due_queue(js):
    mixed = js["mixed"]
    assert len(mixed) == 12
    positions = [i for i, x in enumerate(mixed) if x.startswith("n")]
    assert positions != [9, 10, 11], "new cards were blocked at the end"
    assert min(positions) > 0, "the queue should open with something due"
    assert js["noneFresh"] == 9
    assert js["noneDue"] == 3

"""Which container the recorder asks for, and why it is remembered.

`MediaRecorder.isTypeSupported` said yes to `audio/webm;codecs=opus` on a real
iPhone and then encoded nothing into it: 4455ms of speech came back as five bytes,
twice, with the app telling the learner nothing was recorded and the next attempt
asking for the same dead format. Nothing in the API reports this, so the only way
to know is to look at what came out — and the only useful place to keep the answer
is the device it is true of.

These are the rules for that: prefer what has been seen to work, never ask again
for something that produced nothing, and treat storage failures as forgetfulness
rather than as an error.
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
import { CANDIDATES, pickMime, fileNameFor, store } from '../frontend/capture.js';

function fake(initial = {}, mode = 'ok') {
  const map = new Map(Object.entries(initial));
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => { if (mode === 'full') throw new Error('quota'); map.set(k, v); },
    removeItem: (k) => { map.delete(k); },
  };
}

const all = () => true;
const out = {};

// ── with nothing known, the first supported container wins ─────────────────
out.first = pickMime({ supported: all });
out.appleOnly = pickMime({ supported: (m) => m === 'audio/mp4' });
out.nothingSupported = pickMime({ supported: () => false });

// ── a container that produced nothing is not asked for again ───────────────
out.afterBlockingOpus = pickMime({ supported: all, blocked: ['audio/webm;codecs=opus'] });
out.afterBlockingWebm = pickMime({
  supported: all, blocked: ['audio/webm;codecs=opus', 'audio/webm'],
});

// ── what worked is preferred, unless it has since been struck off ──────────
out.verifiedWins = pickMime({ supported: all, verified: 'audio/mp4' });
out.verifiedButBlocked = pickMime({
  supported: all, verified: 'audio/mp4', blocked: ['audio/mp4'],
});
out.verifiedButUnsupported = pickMime({
  supported: (m) => m !== 'audio/mp4', verified: 'audio/mp4',
});

// ── the memory of the device ───────────────────────────────────────────────
const s = fake();
const mem = store(s);
out.blankVerified = mem.verified();
out.blankBlocked = mem.blocked();
mem.block('audio/webm;codecs=opus');
mem.block('audio/webm');
mem.block('audio/webm');                      // twice is once
out.remembered = mem.blocked();
out.picksMp4Now = pickMime({ supported: all, verified: mem.verified(), blocked: mem.blocked() });
mem.verify('audio/mp4');
out.verifiedStuck = mem.verified();
// A format that starts working again is no longer blocked, and one that stops
// working loses its verification.
mem.verify('audio/webm');
out.unblockedOnVerify = mem.blocked().includes('audio/webm');
mem.block('audio/webm');
out.verificationDropped = mem.verified();

// ── storage that refuses to write is forgetfulness, not an error ───────────
const full = store(fake({}, 'full'));
full.block('audio/webm');
full.verify('audio/mp4');
out.fullSurvives = full.verified() === '' && full.blocked().length === 0;
out.corruptSurvives = store(fake({ 'sm.capture': 'not json' })).verified() === '';

// ── an upload says what it is ──────────────────────────────────────────────
out.names = ['audio/webm;codecs=opus', 'audio/mp4', 'audio/ogg;codecs=opus', '']
  .map(fileNameFor);
out.candidateCount = CANDIDATES.length;

console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def result():
    driver = ROOT / "tests" / "_capture_driver.mjs"
    driver.write_text(DRIVER, encoding="utf-8")
    try:
        proc = subprocess.run([node, str(driver)], cwd=ROOT,
                              capture_output=True, text=True, timeout=30)
    finally:
        driver.unlink(missing_ok=True)
    if proc.returncode != 0:
        pytest.fail(f"node driver failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


def test_opus_is_preferred_where_it_works(result):
    assert result["first"] == "audio/webm;codecs=opus"
    assert result["appleOnly"] == "audio/mp4"
    assert result["nothingSupported"] == "", "must fall back to letting the browser choose"


def test_a_format_that_produced_nothing_is_never_asked_for_again(result):
    """The iPhone bug, and the whole point: the second attempt has to ask for
    something else, or it fails exactly as the first one did."""
    assert result["afterBlockingOpus"] == "audio/webm"
    assert result["afterBlockingWebm"] == "audio/mp4"


def test_what_worked_is_what_gets_used(result):
    assert result["verifiedWins"] == "audio/mp4"
    assert result["verifiedButBlocked"] == "audio/webm;codecs=opus"
    assert result["verifiedButUnsupported"] == "audio/webm;codecs=opus"


def test_the_device_remembers_across_sessions(result):
    assert result["blankVerified"] == ""
    assert result["blankBlocked"] == []
    assert result["remembered"] == ["audio/webm;codecs=opus", "audio/webm"]
    assert result["picksMp4Now"] == "audio/mp4"
    assert result["verifiedStuck"] == "audio/mp4"
    assert result["unblockedOnVerify"] is False, "a format seen working is not still blocked"
    assert result["verificationDropped"] == "", "a format seen failing is not still trusted"


def test_storage_failure_costs_the_memory_and_nothing_else(result):
    assert result["fullSurvives"]
    assert result["corruptSurvives"]


def test_an_upload_is_named_for_what_it_is(result):
    """The server picks a decoder by content type and extension, and iOS records
    mp4 — a clip called speech.webm is a decode failure waiting to happen."""
    assert result["names"] == ["speech.webm", "speech.mp4", "speech.ogg", "speech.webm"]
    assert result["candidateCount"] >= 3

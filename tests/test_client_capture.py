"""Which container the recorder asks for, and who to blame when nothing comes out.

`MediaRecorder.isTypeSupported` said yes to `audio/webm;codecs=opus` on a real
iPhone and returned five bytes for 4455ms of speech, twice — and recognition on
that same phone worked *sometimes*, which is the fact that decides how this has to
behave. A format that never encodes never works, so the container cannot be the
only explanation: something intermittent was taking the microphone away, and the
two faults are indistinguishable from the outside.

So: prefer what has been seen to work, strike a format off only with evidence that
sound reached it and was dropped, reopen the capture session when there is no such
evidence, and treat storage failures as forgetfulness rather than as an error.
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
import { CANDIDATES, pickMime, fileNameFor, store, diagnose } from '../frontend/capture.js';

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

// ── why a recording came back empty, and what to do about it ───────────────
const empty = { ms: 4455, bytes: 5, chunks: 1, mime: 'audio/webm;codecs=opus' };
out.tooShort = diagnose({ ...empty, ms: 100, bytes: 5000 });
out.fine = diagnose({ ms: 3000, bytes: 40000, mime: 'audio/mp4' });
out.unmetered = diagnose(empty);                       // first failure: no meter yet
out.silent = diagnose({ ...empty, peak: 0 });          // the mic gave nothing
out.roomTone = diagnose({ ...empty, peak: 0.04 });     // sound was there
out.loud = diagnose({ ...empty, peak: 0.8 });

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


def test_a_recording_that_worked_is_left_alone(result):
    assert result["fine"]["ok"] is True
    assert result["fine"]["block"] is False and result["fine"]["stale"] is False


def test_a_clip_too_short_to_grade_blames_nothing(result):
    """Let go of the button too early and there is nothing to attribute — no format
    to strike off, no stream to reopen."""
    assert result["tooShort"]["blame"] == "short"
    assert result["tooShort"]["block"] is False and result["tooShort"]["stale"] is False


def test_the_first_empty_recording_concludes_nothing(result):
    """Recognition on the phone this was reported from worked *sometimes*, which
    rules out a format that never encodes. With no level meter yet the two possible
    causes are indistinguishable, so the cheap and likely fix is taken — reopen the
    microphone — and the format is left alone rather than condemned on one failure."""
    d = result["unmetered"]
    assert d["blame"] == "unknown"
    assert d["block"] is False, "must not strike off a format on a guess"
    assert d["stale"] is True, "must reopen the capture session"
    assert d["meter"] is True, "must measure the next attempt"
    assert "try again" in d["reason"]


def test_silence_blames_the_microphone_not_the_format(result):
    """iOS mutes a capture track when another app takes the mic, and `readyState`
    still reads live. No sound reaching us is not the container's fault."""
    d = result["silent"]
    assert d["blame"] == "silence"
    assert d["block"] is False
    assert d["stale"] is True
    assert "another app" in d["reason"]


@pytest.mark.parametrize("case", ["roomTone", "loud"])
def test_sound_that_was_dropped_blames_the_format(result, case):
    """Audio reached the encoder and nothing came out of it. That is the one case
    where striking the container off is the right answer."""
    d = result[case]
    assert d["blame"] == "encoder"
    assert d["block"] is True
    assert "another format" in d["reason"]


def test_an_upload_is_named_for_what_it_is(result):
    """The server picks a decoder by content type and extension, and iOS records
    mp4 — a clip called speech.webm is a decode failure waiting to happen."""
    assert result["names"] == ["speech.webm", "speech.mp4", "speech.ogg", "speech.webm"]
    assert result["candidateCount"] >= 3

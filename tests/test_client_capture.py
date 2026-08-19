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
import re
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
import { CANDIDATES, pickMime, fileNameFor, store, diagnose, EMPTY_BYTES }
  from '../frontend/capture.js';

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
out.afterBlockingMp4 = pickMime({ supported: all, blocked: ['audio/mp4'] });
out.afterBlockingMp4AndOpus = pickMime({
  supported: all, blocked: ['audio/mp4', 'audio/webm;codecs=opus'],
});
// The case an iPhone actually produces: mp4 works, and it is asked for first, so
// nothing has to come back empty before it is reached.
out.appleFirst = pickMime({ supported: (m) => m !== 'audio/webm' });

// ── what worked is preferred, unless it has since been struck off ──────────
out.verifiedWins = pickMime({ supported: all, verified: 'audio/webm' });
out.verifiedButBlocked = pickMime({
  supported: all, verified: 'audio/webm', blocked: ['audio/webm'],
});
out.verifiedButUnsupported = pickMime({
  supported: (m) => m !== 'audio/webm', verified: 'audio/webm',
});

// ── the memory of the device ───────────────────────────────────────────────
const s = fake();
const mem = store(s);
out.blankVerified = mem.verified();
out.blankBlocked = mem.blocked();
mem.block('audio/mp4');
mem.block('audio/webm');
mem.block('audio/webm');                      // twice is once
out.remembered = mem.blocked();
out.picksOpusNow = pickMime({ supported: all, verified: mem.verified(), blocked: mem.blocked() });
mem.verify('audio/webm;codecs=opus');
out.verifiedStuck = mem.verified();
// A format that starts working again is no longer blocked, and one that stops
// working loses its verification.
mem.verify('audio/mp4');
out.unblockedOnVerify = mem.blocked().includes('audio/mp4');
mem.block('audio/mp4');
out.verificationDropped = mem.verified();

// ── storage that refuses to write is forgetfulness, not an error ───────────
const full = store(fake({}, 'full'));
full.block('audio/webm');
full.verify('audio/mp4');
out.fullSurvives = full.verified() === '' && full.blocked().length === 0;
out.corruptSurvives = store(fake({ 'sm.capture': 'not json' })).verified() === '';

// ── why a recording came back empty, and what to do about it ───────────────
out.emptyBytes = EMPTY_BYTES;
out.tooShort = diagnose({ ms: 100, bytes: 5000, chunks: 1, mime: 'audio/mp4' });
out.fine = diagnose({ ms: 3000, bytes: 40000, mime: 'audio/mp4' });

// No container written at all — the iPhone SE, verbatim from the screenshot.
const noContainer = { ms: 2782, bytes: 5, chunks: 1, mime: 'audio/webm;codecs=opus' };
out.noContainer = diagnose(noContainer);
out.noContainerMetered = diagnose({ ...noContainer, peak: 0.04 });

// A container that *was* written and holds almost nothing. This is the band the
// meter decides, and now the only one.
const thin = { ms: 400, bytes: 300, chunks: 1, mime: 'audio/webm;codecs=opus' };
out.unmetered = diagnose(thin);                        // first failure: no meter yet
out.silent = diagnose({ ...thin, peak: 0 });           // the mic gave nothing
out.roomTone = diagnose({ ...thin, peak: 0.04 });      // sound was there
out.loud = diagnose({ ...thin, peak: 0.8 });

// ── an upload says what it is ──────────────────────────────────────────────
out.names = ['audio/webm;codecs=opus', 'audio/mp4', 'audio/ogg;codecs=opus', '']
  .map(fileNameFor);
out.candidateCount = CANDIDATES.length;
out.order = CANDIDATES;

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
    assert result["first"] == "audio/mp4"
    assert result["appleOnly"] == "audio/mp4"
    assert result["appleFirst"] == "audio/mp4", (
        "an iPhone must reach mp4 without a container failing first — a struck-off "
        "format costs the utterance that struck it off")
    assert result["nothingSupported"] == "", "must fall back to letting the browser choose"


def test_a_format_that_produced_nothing_is_never_asked_for_again(result):
    """The iPhone bug, and the whole point: the second attempt has to ask for
    something else, or it fails exactly as the first one did."""
    assert result["afterBlockingMp4"] == "audio/webm;codecs=opus"
    assert result["afterBlockingMp4AndOpus"] == "audio/webm"


def test_what_worked_is_what_gets_used(result):
    assert result["verifiedWins"] == "audio/webm"
    assert result["verifiedButBlocked"] == "audio/mp4"
    assert result["verifiedButUnsupported"] == "audio/mp4"


def test_the_device_remembers_across_sessions(result):
    assert result["blankVerified"] == ""
    assert result["blankBlocked"] == []
    assert result["remembered"] == ["audio/mp4", "audio/webm"]
    assert result["picksOpusNow"] == "audio/webm;codecs=opus"
    assert result["verifiedStuck"] == "audio/webm;codecs=opus"
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


def test_a_container_that_was_never_written_is_struck_off_at_once(result):
    """Five bytes for 2782ms, which is what an iPhone SE returns for a container it
    advertises and does not implement. No meter is needed to read that: a muted
    microphone would still have got its headers written — half a kilobyte of them,
    measured — so nothing under `EMPTY_BYTES` can be the microphone's doing.

    Waiting for a meter before saying so is what made this failure repeat on the first
    press of every session, through three attempts at fixing it."""
    assert result["emptyBytes"] < 600, "the bar has to sit under MIN_BYTES to mean anything"
    for case in ("noContainer", "noContainerMetered"):
        d = result[case]
        assert d["blame"] == "encoder", case
        assert d["block"] is True, f"{case}: the container has to be struck off"
        assert d["stale"] is True
        assert "nothing was encoded" in d["reason"]
    # Unmetered or metered, the verdict is the same — that is the whole point.
    assert result["noContainer"]["meter"] is False, \
        "no meter is needed to read an empty container"


def test_the_first_almost_empty_recording_concludes_nothing(result):
    """A container that *was* written and holds almost nothing is genuinely
    ambiguous — that is the band the meter is for, and now the only one. With no
    meter yet the cheap and likely fix is taken (reopen the microphone) and the
    format is left alone rather than condemned on one failure."""
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
    """Audio reached the encoder and almost nothing came out of it. With the meter
    saying there was signal, striking the container off is the right answer."""
    d = result[case]
    assert d["blame"] == "encoder"
    assert d["block"] is True
    assert "another format" in d["reason"]


def test_an_upload_is_named_for_what_it_is(result):
    """The server picks a decoder by content type and extension, and iOS records
    mp4 — a clip called speech.webm is a decode failure waiting to happen."""
    assert result["names"] == ["speech.webm", "speech.mp4", "speech.ogg", "speech.webm"]
    assert result["order"][0] == "audio/mp4", (
        "mp4 is asked for first: it is what Apple hardware records, and the local "
        "recogniser decodes any container through decodeAudioData")
    assert result["candidateCount"] >= 3


# ── The first press ────────────────────────────────────────────────────────────
#
# Reported from an iPhone SE: the microphone fails on the very first try, every
# time, and works on the second. Two separate faults in `app.js` produce exactly
# that, and both live in the gesture path rather than in this module — so they are
# asserted structurally here, next to the reasoning, because the alternative is a
# browser and a real microphone.


def _app_js() -> str:
    return (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")


def test_the_format_probe_stands_aside_for_a_real_recording():
    """`prewarmMic` is bound to pointerdown on `window`, so on the first press it
    runs when the event *bubbles* — after the button's own handler has called
    `begin()` and found `probing` still null. The guard written to stop two
    MediaRecorders sharing one stream therefore could not fire on the one press it
    was needed for, and the 300ms probe recorded over the first utterance.

    It stops happening for a different reason on the second press — by then a format
    is known and the probe returns immediately — which is why it read as a first-try
    problem rather than as a race."""
    src = _app_js()
    assert "let recordingNow = false;" in src
    assert "if (recordingNow) return;" in src, "verifyCapture no longer stands aside"
    # Set before anything is awaited, or the bubbling listener runs first anyway.
    begin = src.split("async function begin() {")[1].split("async function end()")[0]
    # Comments first: the one above the guard says "after the await below", and
    # splitting on the word would cut the body off before any of it.
    head = re.sub(r"(?m)^\s*//.*$", "", begin).split("await")[0]
    assert "recordingNow = true;" in head, "the flag is set after an await"


def test_letting_go_before_the_microphone_opened_still_sends():
    """Opening the microphone costs 100-500ms and `end()` returned early on
    `!active` — precisely the state a press spends that time in. So on a cold page
    the whole first utterance went into a recorder that had not started yet and a
    release that nothing acted on; the button then turned red by itself and the
    *second* press was what sent anything.

    Unlike the probe race, this does not depend on the device having a format on
    file: it happens on every page load."""
    src = _app_js()
    assert "let releasedEarly = false;" in src
    assert "if (starting) { releasedEarly = true; return; }" in src, \
        "a release during startup is being dropped again"
    assert "if (releasedEarly) await end();" in src, "…and never acted on"


def test_the_button_says_something_before_the_microphone_is_open():
    """Half a second of a button that does nothing is what teaches somebody to press
    it twice — which is how both faults above were being reached in the first
    place."""
    src = _app_js()
    begin = src.split("async function begin() {")[1].split("async function end()")[0]
    # Comments first: the one above the guard says "after the await below", and
    # splitting on the word would cut the body off before any of it.
    head = re.sub(r"(?m)^\s*//.*$", "", begin).split("await")[0]
    assert "Opening the microphone…" in head
    assert "button.classList.add('is-recording');" in head


def test_the_microphone_is_opened_before_the_first_press_where_it_may_be():
    """The fixes above cope with the wait; not paying it is better. Where the browser
    will confirm the microphone is already granted, the stream is opened at startup
    and the first press has nothing to wait for. Only where it will *say* so — a
    `getUserMedia` out of nowhere on a page without permission is a prompt the
    learner did not ask for, and one they are likely to refuse."""
    src = _app_js()
    assert "async function prewarmIfAlreadyAllowed()" in src
    assert "navigator.permissions.query({ name: 'microphone' })" in src
    assert "if (status.state === 'granted')" in src
    assert "prewarmIfAlreadyAllowed();" in src, "defined but never called"


def test_the_probe_runs_before_the_press_it_has_to_inform():
    """`begin()` holds the recording until `probing` settles, which only means
    anything if `probing` has been *set* by then. Bound in the bubble phase, the
    warm-up ran after the button's own handler on the very press that needed it — so
    the guard was dead code and the first utterance went into whatever container
    `isTypeSupported` had claimed. In the capture phase the window listener runs
    first, which is what makes the await load-bearing.

    Reported twice from an iPhone SE: first as a mic that failed on the first press
    and worked on the second, then — once the release-during-startup fault was fixed
    and the utterance survived — as `2444ms recorded but only 5 bytes captured
    (1 chunk, audio/webm;codecs=opus)`."""
    src = _app_js()
    for line in ("window.addEventListener('pointerdown', prewarmMic, "
                 "{ once: true, capture: true });",
                 "window.addEventListener('keydown', prewarmMic, "
                 "{ once: true, capture: true });"):
        assert line in src, f"not in the capture phase: {line}"
    assert "if (probing) await probing;" in src, "…and nothing waits for it"


def test_the_probe_strikes_off_a_container_without_waiting_for_a_rival():
    """The probe used to condemn nothing unless some *other* container had proved
    itself first, on the reasoning that a muted microphone makes them all look broken.
    It does not — see `EMPTY_BYTES` — and that caution was load-bearing in the wrong
    direction: on a phone whose only advertised container is a broken one, nothing was
    ever struck off and every first press went into it.

    Striking off the last one is not a dead end. `pickMime` then returns the empty
    string, which is `new MediaRecorder(stream)` with no container named — the browser
    recording in whatever it actually implements, which is the answer on that phone."""
    src = _app_js()
    verify = src.split("async function verifyCapture(stream) {")[1].split("\nfunction ")[0]
    assert "if (bytes < capture.EMPTY_BYTES) capabilities.block(mime);" in verify
    assert "usable.length < 2" not in verify, \
        "the short-circuit is back, and it skips the striking-off that fixes this"
    assert "supportsMime(m) && !blocked.includes(m)" in verify

    # And the fallback it opens up has to survive in pickMime.
    cap = (ROOT / "frontend" / "capture.js").read_text(encoding="utf-8")
    assert "|| ''" in cap.split("export function pickMime")[1].split("\n}")[0], \
        "nothing falls back to letting the browser choose"


def test_a_failure_says_what_the_device_offered():
    """This bug was diagnosed wrongly twice from a screenshot, because "5 bytes of
    audio/webm;codecs=opus" says which container failed and nothing about why that one
    was asked for — whether mp4 was offered and passed over, whether something had
    already been struck off, whether the probe ran at all. That decides which fix is
    right, and it fits on one line."""
    src = _app_js()
    assert "function captureState()" in src
    assert "${reason}${captureState()}" in src, "the state is computed and not shown"
    state = src.split("function captureState() {")[1].split("\n}")[0]
    assert "capture.CANDIDATES" in state and "supportsMime(m)" in state
    assert "capabilities.blocked()" in state and "capabilities.verified()" in state

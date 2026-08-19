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
// And no chunk at all, which is the recorder rather than the container.
const noChunk = { ms: 2959, bytes: 0, chunks: 0, mime: 'audio/mp4;codecs=mp4a.40.2' };
out.noChunk = diagnose(noChunk);
out.noChunkMetered = diagnose({ ...noChunk, peak: 0.4 });
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


def test_the_probe_never_touches_the_microphone():
    """An iPhone SE returned the identical five-byte stub for `audio/webm;codecs=opus`
    and then for `audio/mp4;codecs=mp4a.40.2` — two unrelated encoders — before a third
    container recorded properly through the same microphone. That rules out both stories
    this code was built on: the microphone was working, and "the container is broken"
    cannot hold for two unrelated codecs at once while a third succeeds.

    What is left is that the device advertises more containers than it implements, and
    the only way to know which is to try. So the probe tries all of them, on an
    oscillator rather than on the microphone:

    * it needs no permission, so it can run before anyone has agreed to anything;
    * it cannot disturb the capture session the first utterance is about to use, which
      was the first theory of this bug and is worth keeping ruled out;
    * and with no capture session to conflict over, the containers can be measured at
      the same time — 2.2s for three against 6.0s in sequence, same verdicts."""
    src = _app_js()
    probe = src.split("async function verifyCapture() {")[1].split("\n/**")[0]

    assert "createMediaStreamDestination()" in probe, "the probe is back on the mic"
    assert "createOscillator()" in probe
    assert "getUserMedia" not in probe, "the probe must not open a capture session"
    # All of them, at once.
    assert "await Promise.all(" in probe
    assert "capture.CANDIDATES.filter(" in probe

    # A context that will not start produces silence, and silence must never be
    # recorded as evidence against an encoder.
    assert "if (ctx.state !== 'running') return;" in probe
    # Nor may a probe where *nothing* wrote condemn anything.
    assert "if (!wrote.length) {" in probe
    assert "for (const m of empty) capabilities.block(m);" in probe


def test_the_press_no_longer_waits_for_the_probe():
    """The probe used to record 300ms through the microphone the press was about to
    use, so `begin()` held the recording until it finished — which on a device with
    three containers to try cost up to two seconds of the first utterance. On a
    synthetic stream there is nothing to collide with, so the press starts at once."""
    src = _app_js()
    begin = src.split("async function begin() {")[1].split("async function end()")[0]
    assert "await probing" not in begin, \
        "the press is waiting on the probe again — it no longer has to"
    # …and the recording still starts before anything else can be awaited.
    head = re.sub(r"(?m)^\s*//.*$", "", begin).split("await")[0]
    assert "recordingNow = true;" in head


# ── Acquired is not delivering ─────────────────────────────────────────────────

DELIVERY_DRIVER = r"""
import { readFileSync } from 'node:fs';

/* `whenDelivering` is lifted out and run, the same way test_api.py lifts `routeFor`
   out of the service worker: the rest of app.js needs a DOM and this does not. */
const src = readFileSync('frontend/app.js', 'utf8');
const start = src.indexOf('function whenDelivering');
if (start < 0) throw new Error('app.js has no whenDelivering to test');
const end = src.indexOf('\n}', start) + 2;
const whenDelivering = new Function(
  'DELIVERY_MS', `${src.slice(start, end)}; return whenDelivering;`)(700);

const fake = (muted) => {
  const listeners = {};
  const track = {
    muted,
    addEventListener: (k, f) => { (listeners[k] ||= []).push(f); },
    removeEventListener: (k, f) => { listeners[k] = (listeners[k] || []).filter((x) => x !== f); },
    fire: (k) => (listeners[k] || []).slice().forEach((f) => f()),
    listenerCount: () => Object.values(listeners).flat().length,
  };
  return { track, stream: { getAudioTracks: () => [track] } };
};

const out = {};
let t0;

// Already delivering: nothing to wait for.
const live = fake(false);
t0 = Date.now();
await whenDelivering(live.stream);
out.deliveringMs = Date.now() - t0;

// Muted, then the first samples arrive.
const late = fake(true);
t0 = Date.now();
setTimeout(() => late.track.fire('unmute'), 120);
await whenDelivering(late.stream);
out.unmutedMs = Date.now() - t0;
out.listenersLeft = late.track.listenerCount();

// A browser that never fires `unmute` must not hold the microphone shut.
const never = fake(true);
t0 = Date.now();
await whenDelivering(never.stream, 250);
out.boundedMs = Date.now() - t0;

// And nothing to wait on is not an error.
await whenDelivering({ getAudioTracks: () => [] });
await whenDelivering(undefined);
out.tolerant = true;

console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def delivery():
    driver = ROOT / "tests" / "_delivery_driver.mjs"
    driver.write_text(DELIVERY_DRIVER, encoding="utf-8")
    try:
        proc = subprocess.run([node, str(driver)], cwd=ROOT,
                              capture_output=True, text=True, timeout=30)
    finally:
        driver.unlink(missing_ok=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_the_recorder_waits_for_the_first_samples(delivery):
    """`getUserMedia` resolves when permission is granted and a track exists, which is
    before the source has started. A track from a live source arrives `muted: true` and
    fires `unmute` when audio begins flowing, and that gap is where four rounds of this
    bug lived — most clearly in the last one:

        audio/mp4;codecs=mp4a.40.2   2034ms →  0 bytes, 0 chunks · mic live

    Zero chunks means `dataavailable` never fired at all, in a private tab with no
    stored state, on a container the probe had just verified, with the track reading
    `live` by the time the message was drawn. The container was never the problem: the
    recorder was started against a microphone that had not begun, and the second press
    worked because by then it had.

    Waiting on `unmute` is the fix, and it has to be bounded — a browser that never
    fires it must not hold the microphone shut."""
    assert delivery["deliveringMs"] < 60, "a live track must not be waited on"
    assert 100 <= delivery["unmutedMs"] < 700, \
        "must resolve when the samples arrive, not on the timeout"
    assert 240 <= delivery["boundedMs"] < 700, "the wait has to be bounded"
    assert delivery["listenersLeft"] == 0, "the unmute listener outlived the wait"
    assert delivery["tolerant"], "no track at all must resolve, not throw"


def test_a_muted_track_is_waited_for_and_not_replaced():
    """`start()` used to treat "muted" as "stale" and throw the stream away — which is
    the one response guaranteed not to help, because the replacement arrives muted too.
    Ended and muted are different faults: one needs a new track, the other needs a
    moment."""
    src = _app_js()
    start = src.split("  async start() {")[1].split("\n  }")[0]
    assert "micTrack().readyState !== 'live'" in start, \
        "an ended capture session still has to be replaced"
    assert "!t.muted" not in start, \
        "muted is being treated as a reason to reopen the stream again"
    assert "await whenDelivering(sharedStream);" in start

    # Paid on the warm-up too, so in the ordinary case a press waits for nothing.
    ensure = src.split("async function ensureStream() {")[1].split("\n}")[0]
    assert "await whenDelivering(sharedStream);" in ensure


def test_the_encoder_is_given_a_webaudio_reroute_of_the_microphone():
    """The fault that survived every other fix:

        audio/mp4;codecs=mp4a.40.2   2959ms → 0 bytes, 0 chunks
                                     · mic live · at start delivering

    A container the probe had verified, a private tab with no stored state, and a track
    that was live and delivering when the recorder started. Three seconds of speech and
    `dataavailable` never fired once — which rules out the container, the candidate
    order, the probe's timing and the track's readiness, in that order, being the four
    things already tried.

    What has never failed on that phone is the probe, and the probe records a
    `MediaStreamAudioDestinationNode`. So the microphone is routed through WebAudio and
    the encoder is given *that*: measured at 17833 bytes where the raw stream gives
    nothing. The capture session is still opened, watched and stopped as before — this
    only changes what sits between it and the encoder."""
    src = _app_js()
    route = src.split("async function throughWebAudio(stream) {")[1].split("\n}")[0]
    assert "createMediaStreamSource(stream)" in route
    assert "createMediaStreamDestination()" in route
    assert "micSource.connect(dest)" in route
    # A suspended context passes no samples, which would turn a working microphone into
    # silence — worse than the fault being fixed.
    assert "if (audioCtx.state !== 'running') return stream;" in route
    # The last catch is the outer one; the inner catch is the disconnect.
    assert "return stream;" in route.split("catch")[-1], "no fallback to the raw stream"

    ensure = src.split("async function ensureStream() {")[1].split("\n}")[0]
    assert "recordable = await throughWebAudio(sharedStream);" in ensure
    assert "return recordable;" in ensure

    # The health of the capture session is still read from the microphone's own track.
    assert "const micTrack = () => sharedStream?.getAudioTracks?.()[0] || null;" in src
    # …and the report says which path the encoder was actually given.
    assert "· via ${audioCtx?.state === 'running' && micSource ? 'webaudio' : 'raw'}" in src


def test_the_failure_says_whether_the_microphone_had_started():
    """`mic live` is read when the failure is drawn, by which point the track has
    usually unmuted and the evidence has gone. What decides whether anything could have
    been captured is its state when the recorder *started*, so that is recorded then and
    reported alongside."""
    src = _app_js()
    assert "lastMutedAtStart = !!micTrack()?.muted;" in src
    assert "at start ${" in src
    assert "· mic ${mic}${atStart}" in src


@pytest.mark.parametrize("case", ["noChunk", "noChunkMetered"])
def test_no_chunk_at_all_is_not_the_containers_fault(result, case):
    """`dataavailable` never fired — not one chunk, empty or otherwise. There is nothing
    to accuse the container of, because it was never handed anything to write; an
    encoder that cannot use a container it accepted still emits its stub, which is the
    five-byte case.

    Learned the hard way: `audio/webm;codecs=opus` and then `audio/mp4;codecs=mp4a.40.2`
    were both struck off on this evidence and both were innocent. Blaming the container
    here burns through the whole list, one wasted utterance at a time."""
    d = result[case]
    assert d["blame"] == "recorder"
    assert d["block"] is False, "a container is being struck off for the recorder again"
    assert d["stale"] is True, "the capture session is the thing to reopen"
    assert "never delivered any audio" in d["reason"]

/* Nitkellmu — Speak Maltese
   Single-page client. No build step, no dependencies.

   The learner's progress lives here, in IndexedDB, not on the server: see
   store.js for the database and schedule.js for the FSRS scheduling that used to
   run in Python. The server is stateless — decks, dialogue, recognition, speech. */

import * as dialogueEngine from './dialogue.js';
import * as nanostt from './nanostt.js';
import * as splash from './splash.js';
import * as store from './store.js';
import * as schedule from './schedule.js';
import * as srs from './srs.js';
import * as mtext from './text.js';
import * as session from './session.js';
import * as capture from './capture.js';
import * as games from './games.js';

const $ = (id) => document.getElementById(id);

const state = {
  caps: null,
  settings: store.loadSettings(),
  queue: [],
  qIndex: 0,
  card: null,
  revealed: false,
  attempted: false,
  // Set during startup when the recogniser had to be switched off, and shown once
  // the UI exists — a toast during the splash is a toast nobody sees.
  sttNotice: '',
};

/* ── API helpers ──────────────────────────────────────────────────────── */

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: opts.body instanceof FormData ? {} : { 'content-type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* non-JSON body */ }
    throw new Error(detail);
  }
  return res.headers.get('content-type')?.includes('application/json') ? res.json() : res;
}

const post = (path, body) => api(path, { method: 'POST', body: JSON.stringify(body) });

/* ── Static mode ────────────────────────────────────────────────────────────
   The same client runs two ways. Against the FastAPI app it posts to /api/*; on a
   dumb host (GitHub Pages) there is no server, so the endpoints that were pure
   computation run here instead — the Maltese rules, the dialogue matcher and the
   scheduler are all ported and parity-tested against the Python they came from.
   Speech recognition is the one thing with no static equivalent: it runs on the
   device or not at all. */

const STATIC = { on: false, audio: null, sttBase: '' };

/** Where an utterance goes to be recognised, when it is not this device.

    Kept for a deployment that wants recognition centralised, but it is no longer the
    way a phone gets to speak: the recogniser is 2.1MB now and ships with the page.
    Empty — the default — means everything happens here. */
const remoteStt = () => (STATIC.on ? STATIC.sttBase : '');

/* How long the startup screen waits for the recogniser before opening without it.
   It is 2.1MB from our own origin, so this is generous rather than a real limit —
   and if it is exceeded the app opens anyway and the model arrives when it arrives. */
const MODEL_WAIT_MS = 15000;

/** Pre-rendered speech, looked up by line rather than synthesised on demand. */
function staticAudioUrl(line) {
  const file = STATIC.audio?.[mtext.normalise(line)] ?? STATIC.audio?.[line];
  return file ? `audio/${file}` : null;
}

function toast(msg, ms = 3200) {
  const el = $('toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, ms);
}

function escapeHtml(s = '') {
  return s.replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

/* ── Audio ─────────────────────────────────────────────────────────────── */

let currentAudio = null;
/** Resolver for the line currently being spoken, so superseding it settles its
    promise rather than leaving an `await speak(...)` hanging on stopped audio. */
let currentDone = null;

function speak(text, { rate } = {}) {
  if (!text) return Promise.resolve();
  const r = rate ?? state.settings.rate;
  // Static builds ship one render per line, at the default rate. A slow replay
  // asks for a rate that was never rendered, so it falls back to the normal one
  // played slower — <audio> can do that itself.
  const url = STATIC.on
    ? staticAudioUrl(text)
    : `/api/tts?text=${encodeURIComponent(text)}&rate=${r}&voice=${encodeURIComponent(state.settings.voice)}`;
  if (STATIC.on && !url) return Promise.resolve();
  // Whatever was playing is superseded — and its promise has to be settled, or an
  // `await speak(...)` upstream of it waits for a sound that has already stopped.
  if (currentAudio) { currentAudio.pause(); currentAudio = null; }
  if (currentDone) { currentDone(); currentDone = null; }

  const audio = new Audio(url);
  if (STATIC.on && rate && rate !== state.settings.rate) {
    audio.playbackRate = Math.max(0.5, Math.min(1.5, rate / state.settings.rate));
  }
  currentAudio = audio;

  /* Resolves when the line has been *said*, not when it started being said.

     This used to resolve on `play()`, which returns as soon as playback begins — so
     `await speak(reply)` in a drill turn waited for nothing, and 450ms later the next
     prompt called `speak()` again, which pauses whatever is playing. Every tutor reply
     longer than half a second was therefore cut off mid-sentence, every turn. Reported
     as audio that "seems missing or cut off", and it was the second half of nearly all
     of it. */
  return new Promise((resolve) => {
    const settle = () => {
      if (currentAudio === audio) currentAudio = null;
      if (currentDone === resolve) currentDone = null;
      resolve();
    };
    currentDone = resolve;
    audio.addEventListener('ended', settle, { once: true });
    // A line with no file, or a decode failure: the turn must not wait on it forever.
    audio.addEventListener('error', settle, { once: true });
    audio.play().catch((err) => {
      // Autoplay policies block the first sound until the user interacts.
      if (err.name !== 'NotAllowedError') toast('Audio unavailable — check TTS setup');
      settle();
    });
  });
}

/* ── Recording ─────────────────────────────────────────────────────────── */

/* The microphone stream is acquired once and kept open for the session.
   getUserMedia costs 100-500ms, and previously that was paid *after* the button
   was pressed — so the first syllable was simply never recorded. That is what
   turned "Bonġu" into "onġi" and "Silas" into "salas": the recogniser was not
   mishearing the onset, it never received it. */
let sharedStream = null;
let pendingStream = null;

/* …but a stream you hold is not a stream that works. iOS mutes a capture track when
   another app takes the microphone, when a call arrives, or when the page goes to the
   background, and `readyState` keeps saying `live` afterwards — so the app records
   silence and cannot tell. That is the shape of "it works sometimes": nothing here
   was ever watching for the moment it stopped.

   So the stream is marked stale on every signal there is, and a stale one is thrown
   away and reopened rather than reused. Reopening costs 100-500ms, once, after
   something has already gone wrong. */
let streamStale = false;

function markStreamStale() { streamStale = true; }

document.addEventListener('visibilitychange', () => {
  if (document.hidden) markStreamStale();
});

async function ensureStream() {
  if (sharedStream?.active && !streamStale) return recordable ?? sharedStream;
  if (sharedStream && streamStale) {
    // Let the device go before asking for it again: two live capture sessions on iOS
    // is how you get one that produces nothing.
    for (const t of sharedStream.getTracks()) t.stop();
    sharedStream = null;
    recordable = null;
  }
  // Share the in-flight request. prewarmMic() and a mic press can land together,
  // and without this each opened its own stream — the loser was overwritten but
  // never stopped, leaving the recording indicator on for a stream nothing held.
  if (pendingStream) return pendingStream;
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('Microphone not available');
  pendingStream = navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1,
    },
  });
  try {
    sharedStream = await pendingStream;
  } finally {
    pendingStream = null;
  }
  streamStale = false;
  // The events iOS does give us when it takes the microphone away. `mute` is the one
  // that matters — the track stays `live` through it, which is why checking
  // `readyState` alone was never enough.
  for (const t of sharedStream.getAudioTracks()) {
    t.addEventListener('mute', markStreamStale);
    t.addEventListener('ended', markStreamStale);
  }
  // Acquired is not the same as delivering. Paid here, on the warm-up, so a press
  // does not pay it. See `whenDelivering`.
  await whenDelivering(sharedStream);
  recordable = await throughWebAudio(sharedStream);
  return recordable;
}

/* ── What MediaRecorder is actually given ──────────────────────────────────────

   Not the microphone. The microphone, re-routed through WebAudio:

       getUserMedia → MediaStreamAudioSourceNode → MediaStreamAudioDestinationNode
                    → MediaRecorder

   Five rounds of this bug were spent on the wrong suspects — the container, the
   candidate order, the probe's timing, a track that had not started — and each theory
   was refuted by the next screenshot. The last one refuted the lot:

       audio/mp4;codecs=mp4a.40.2   2959ms → 0 bytes, 0 chunks
                                    · mic live · at start delivering

   A container the probe had verified, in a private tab with no stored state, on a
   track that was live and *delivering when the recorder started*. Three seconds of
   speech and `dataavailable` never fired once. Nothing about the format, the ordering
   or the readiness of the microphone can explain that, and the second press worked on
   the same container through the same microphone.

   What can explain it is the one thing that has worked on the first attempt every
   single time, on this same phone and this same browser: the format probe. It records
   a `MediaStreamAudioDestinationNode`, and it has never once come back empty. Measured
   again here before committing to it — a MediaStream re-routed through WebAudio and
   recorded gives 17833 bytes where the raw stream gives nothing.

   So the recorder is given the re-routed stream. The microphone is still opened,
   watched and stopped exactly as before — it is the capture session, and `sharedStream`
   remains the thing that goes stale, mutes and ends. This only changes what sits
   between it and the encoder.

   If the AudioContext will not run, the raw stream is handed over as it always was.
   That is the old behaviour, which is broken on this phone and fine everywhere else,
   and it is better than no recording at all. */
let audioCtx = null;
let micSource = null;
let recordable = null;

async function throughWebAudio(stream) {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return stream;
  try {
    audioCtx = audioCtx || new Ctx();
    await audioCtx.resume();
    // Suspended means no samples will flow through the graph, which would turn a
    // working microphone into silence. Better the raw stream than that.
    if (audioCtx.state !== 'running') return stream;

    try { micSource?.disconnect(); } catch { /* already gone */ }
    micSource = audioCtx.createMediaStreamSource(stream);
    const dest = audioCtx.createMediaStreamDestination();
    micSource.connect(dest);
    return dest.stream;
  } catch {
    return stream;                        // no graph; record the microphone directly
  }
}

/** The capture session's own track — for health, staleness and the failure report.
    `ensureStream()` returns what the *encoder* is given, which is a WebAudio stream
    whose track is always live and says nothing about the microphone. */
const micTrack = () => sharedStream?.getAudioTracks?.()[0] || null;

/* How long to wait for a freshly-opened microphone to start producing samples.

   Short on purpose. In the normal case this is paid on the first gesture and costs the
   learner nothing; when it is paid on a press it is added to the front of the
   utterance, and a wait longer than this would be worse than the fault it is avoiding. */
const DELIVERY_MS = 700;

/** Resolve once the microphone is actually producing audio — not merely open.

    `getUserMedia` resolves when permission is granted and a track exists, which is
    before the source has started. Per spec a track from a live source arrives
    `muted: true` and fires `unmute` when the first samples flow, and that gap is where
    three rounds of this bug lived:

        audio/webm;codecs=opus   2055ms →  5 bytes,  1 chunk
        audio/mp4;codecs=mp4a…   2055ms →  5 bytes,  1 chunk
        audio/mp4;codecs=mp4a…   2034ms →  0 bytes,  0 chunks   ← private tab, fresh

    The last one settles it. Zero chunks means `dataavailable` never fired at all, in a
    tab with no stored state, on a container the probe had just verified, with the track
    reading `live` by the time the failure was drawn. Nothing was wrong with the
    container — the recorder was started against a microphone that had not begun, and
    the second press worked because by then it had. That is also why the earlier two
    produced a five-byte stub rather than nothing: WebM writes its header before the
    first sample arrives and MP4 does not.

    `start()` used to react to `muted` by throwing the stream away and opening another
    one, which is the one thing guaranteed not to help: the replacement arrives muted
    too. Waiting is the whole fix. */
function whenDelivering(stream, timeoutMs = DELIVERY_MS) {
  const track = stream?.getAudioTracks?.()[0];
  if (!track || !track.muted) return Promise.resolve(stream);
  return new Promise((resolve) => {
    const settle = () => {
      clearTimeout(timer);
      track.removeEventListener('unmute', settle);
      resolve(stream);
    };
    // Bounded: a browser that never fires `unmute` must not hold the microphone shut.
    const timer = setTimeout(settle, timeoutMs);
    track.addEventListener('unmute', settle);
  });
}

/* Was there any sound at all? Without this, "nothing recorded" cannot tell a
   microphone that has been taken away from a container that threw the audio out —
   and those two want opposite fixes.

   Only attached after something has already failed. It costs an AudioContext, and on
   iOS every extra audio node is another way to disturb the capture session the app
   is trying to protect; paying that on every turn to answer a question that has not
   been asked would be the wrong trade. */
let meterWanted = false;
let meterCtx = null;

function levelMeter(stream) {
  if (!meterWanted) return null;
  try {
    // The capture graph if there is one: on iOS every extra audio node is another way
    // to disturb a capture session, and there is no reason to build a second context
    // when `throughWebAudio` already has one running.
    const Ctx = window.AudioContext || window.webkitAudioContext;
    meterCtx = audioCtx || meterCtx || new Ctx();
    const analyser = meterCtx.createAnalyser();
    analyser.fftSize = 512;
    const src = meterCtx.createMediaStreamSource(stream);
    src.connect(analyser);
    const buf = new Float32Array(analyser.fftSize);
    let peak = 0;
    const timer = setInterval(() => {
      analyser.getFloatTimeDomainData(buf);
      for (const v of buf) peak = Math.max(peak, Math.abs(v));
    }, 50);
    return {
      peak: () => peak,
      stop() {
        clearInterval(timer);
        try { src.disconnect(); analyser.disconnect(); } catch { /* already gone */ }
      },
    };
  } catch {
    return null;      // no metering, so a failure stays honestly unattributed
  }
}

/* Try the chosen container for a fraction of a second, on the first interaction and
   off the answer path, to find out whether the browser encodes into it at all.

   Without this the discovery costs a turn: on an iPhone the first thing a learner
   says goes into a format the browser recommended and does not implement, and comes
   back as five bytes. The probe is 300ms of nothing, and after it the first real
   recording asks for a container this device has been seen to produce. */
const PROBE_MS = 300;
/* 300ms of Opus or AAC is a couple of thousand bytes; an empty container is tens.
   The bar only has to tell "something" from "nothing". */
const PROBE_BYTES = 120;

let probing = null;

function recordBriefly(stream, mime) {
  // Outside the promise on purpose: a browser that refuses the format throws here,
  // and the caller's try/catch is the right place for that — it strikes the format
  // off and moves down the list.
  const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  return new Promise((done) => {
    let bytes = 0;
    rec.ondataavailable = (e) => { bytes += e.data?.size || 0; };
    // Safari can deliver the last chunk after `stop`, so the count is read a moment
    // later rather than in the handler that fired first.
    rec.onstop = () => setTimeout(() => done(bytes), 250);
    rec.onerror = () => done(bytes);
    rec.start();
    setTimeout(() => { if (rec.state !== 'inactive') rec.stop(); }, PROBE_MS);
  });
}

/** Which containers this device actually writes into.

    Two rounds of this were built on a guess and both were wrong, so here is what an
    iPhone SE actually did, one press at a time:

        audio/webm;codecs=opus    2055ms →     5 bytes
        audio/mp4;codecs=mp4a...  2055ms →     5 bytes
        (whatever was left)              →     worked

    Two unrelated encoders — Opus in WebM and AAC-LC in MP4 — returning the identical
    five-byte stub, then a third container recording properly through the same
    microphone seconds later. Whatever this is, it is not the two explanations the code
    was written around: the microphone was demonstrably working, and "that container is
    broken" cannot be true of two unrelated codecs at once while a third succeeds. What
    is certainly true is that this phone advertises three containers and implements
    fewer, and there is no way to know which except to try them.

    So all of them are tried — and on a stream with no microphone in it.

    An oscillator into a `MediaStreamAudioDestinationNode` is a real MediaStream
    carrying real samples, and it answers the only question the probe was ever asking:
    does this browser write bytes when told to use this container. Doing it that way
    buys three things the microphone could not. It needs no permission, so it can run
    before anyone has agreed to anything. It cannot disturb the capture session the
    first utterance is about to use — the original theory of this bug, and worth keeping
    ruled out rather than reintroduced. And because there is no capture session to
    conflict over, the containers can be measured *at the same time*: 2.2s for all
    three against 6.0s one after another, with the same verdicts.

    Which in turn means `begin()` no longer waits for it. The probe and the first press
    are on separate streams now, so the press starts recording immediately and the probe
    lands when it lands — at worst one press early. */
const PROBE_TONE_HZ = 440;

async function verifyCapture() {
  if (capabilities.verified()) return;            // already known on this device

  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return;
  const ctx = new Ctx();
  try {
    await ctx.resume();
    /* A context that will not start produces no samples, and every container would
       measure as empty — which is the one outcome that must never be recorded as
       evidence. Safari suspends until a gesture, which is why this runs from one. */
    if (ctx.state !== 'running') return;

    const dest = ctx.createMediaStreamDestination();
    const osc = ctx.createOscillator();
    osc.frequency.value = PROBE_TONE_HZ;
    osc.connect(dest);
    osc.start();

    const blocked = capabilities.blocked();
    const usable = capture.CANDIDATES.filter(
      (m) => supportsMime(m) && !blocked.includes(m));
    // -1 for a container the constructor refused outright, which is a real answer.
    const sizes = await Promise.all(
      usable.map((m) => recordBriefly(dest.stream, m).catch(() => -1)));
    osc.stop();

    const wrote = usable.filter((m, i) => sizes[i] >= PROBE_BYTES);
    const empty = usable.filter((m, i) => sizes[i] < capture.EMPTY_BYTES);

    /* Nothing wrote anything, on a stream that was certainly producing samples. That
       is not several broken encoders, it is a probe that cannot be trusted — so no
       container is condemned on it and the next real recording is metered instead. */
    if (!wrote.length) {
      if (empty.length) meterWanted = true;
      return;
    }

    capabilities.verify(capture.CANDIDATES.find((m) => wrote.includes(m)));
    for (const m of empty) capabilities.block(m);
  } catch {
    /* An AudioContext this browser will not give us says nothing about its encoders. */
  } finally {
    try { await ctx.close(); } catch { /* already gone */ }
  }
}

/** Whether the microphone had started delivering when the last recording began.
    `null` until something has been recorded. Read by `captureState()`. */
let lastMutedAtStart = null;

/** True from the moment a mic press is handled until its recording has been sent.
    Module-level because the two things that must not overlap live in different
    scopes: the format probe up here and the button's `begin()` down there. */
let recordingNow = false;

/** Warm the mic on the first interaction so the first recording isn't the slow one,
    and find out what this device can actually record while we are here. */
function prewarmMic() {
  /* Two independent things on the first gesture: open the microphone, so the first
     press does not pay for it, and find out what this browser encodes into, which
     needs no microphone at all. Neither waits for the other. */
  probing = verifyCapture().catch(() => {}).finally(() => { probing = null; });
  return ensureStream()
    .catch(() => { /* permission comes later, on first real use */ })
    .then(() => probing);
}

/* Opening the microphone costs 100-500ms, and until it is open a press records
   nothing. The gesture handlers below cope with that, but coping is not the same as
   not paying it — so where the browser will say that permission is already granted,
   the stream is opened at startup and the first press has nothing to wait for.

   Only where it will *say* so. `getUserMedia` without a gesture on a page that has
   not been granted the microphone is a prompt out of nowhere, which is both rude and
   likely to be denied; and `permissions.query` does not accept 'microphone'
   everywhere, so a browser that will not answer simply keeps the old behaviour of
   waiting for a first touch. */
async function prewarmIfAlreadyAllowed() {
  try {
    const status = await navigator.permissions.query({ name: 'microphone' });
    if (status.state === 'granted') await prewarmMic();
  } catch { /* not supported, or not answerable: the gesture path still warms it */ }
}

/* Recording, written for Safari on iOS rather than for the spec.

   Two guesses at the "Too short" reports were wrong, so this stops guessing:
   `stop()` now reports *why* it rejected a clip and the caller shows it, which
   turns a dead end into a number. The changes below are the causes worth ruling
   out at the same time.

   No timeslice. `start(200)` asks for periodic `dataavailable` events, and iOS
   delivers those as fragments whose first chunk holds the container header —
   reassembling them into one Blob can yield something the decoder reads as almost
   empty. One chunk at stop is what Safari does reliably.

   Wait for the data, not for `stop`. The spec fires `dataavailable` before
   `stop`, but Safari has shipped versions where the final chunk lands after, so
   resolving on `onstop` could read `chunks` while still empty — a full-length
   recording that measures as nothing. Both are awaited now.

   And a stream can go stale. iOS mutes or ends tracks when the page is
   backgrounded or another app takes the mic, and `stream.active` can still read
   true afterwards, so a live-looking stream produces silence. The tracks are
   checked before use, watched while held, and re-acquired if either says stop.

   The report is an attribution now, not a measurement. Recognition on the phone it
   was failing on worked *sometimes*, which rules out the simple explanations: a
   format that never encodes never works, and a permission that is refused never
   works either. Something intermittent was taking the microphone away, so a failed
   recording asks which of the two possible culprits it was — no sound reaching us,
   or sound we failed to encode — and does the matching thing. See capture.js. */

/** What this device has been seen to record, across sessions. */
const capabilities = capture.store();

const supportsMime = (m) => {
  try {
    return MediaRecorder.isTypeSupported(m);
  } catch {
    return false;
  }
};

const chosenMime = () => capture.pickMime({
  supported: supportsMime,
  verified: capabilities.verified(),
  blocked: capabilities.blocked(),
});

/** What this device claims and what has been learned about it, appended to a failure.

    Written because this bug has now been diagnosed wrongly twice from a screenshot.
    "5 bytes of audio/webm;codecs=opus" says which container failed and nothing about
    why that container was the one asked for — whether `audio/mp4` was offered and
    passed over, whether a previous attempt had already struck something off, whether
    the probe ever ran. All of that decides which fix is the right one, and all of it
    is one line on the screen. */
function captureState() {
  const claims = capture.CANDIDATES
    .map((m) => `${m.replace('audio/', '').replace(';codecs=opus', '/opus')}${
      supportsMime(m) ? '' : '✗'}`)
    .join(' ');
  const blocked = capabilities.blocked().map((m) => m.replace('audio/', '')).join(',');
  /* And what the track says about itself. `muted` is the one iOS sets when another app
     takes the microphone while `readyState` still reads `live`, and between them they
     separate "nothing reached the encoder" from "the encoder dropped it" — which is
     the distinction two rounds of this bug turned on. */
  const track = micTrack();
  const mic = track
    ? `${track.readyState}${track.muted ? '/muted' : ''}${track.enabled ? '' : '/disabled'}`
    : 'no track';
  // What it was when the recorder started, which is the state that decides whether
  // anything could have been captured. By failure time it has usually unmuted.
  const atStart = lastMutedAtStart === null ? '' : ` · at start ${
    lastMutedAtStart ? 'muted' : 'delivering'}`;
  return ` · offers ${claims}`
    + `${blocked ? ` · struck off ${blocked}` : ''}`
    + `${capabilities.verified() ? ` · known good ${capabilities.verified().replace('audio/', '')}` : ''}`
    + ` · mic ${mic}${atStart}`
    // Which stream the encoder was given. `raw` means the WebAudio graph would not
    // start and the old, broken-on-some-phones path was used.
    + ` · via ${audioCtx?.state === 'running' && micSource ? 'webaudio' : 'raw'}`;
}

class Recorder {
  constructor() { this.chunks = []; this.rec = null; }

  async start() {
    let stream = await ensureStream();
    /* Asked of the microphone, not of what the encoder is given — the re-routed stream
       has a track of its own that is always live and knows nothing about the capture
       session. A track that has ended is gone and needs replacing; one that is merely
       muted has not started yet, and replacing it produces another that has not
       started either, so that case waits. */
    if (micTrack() && micTrack().readyState !== 'live') {
      markStreamStale();
      stream = await ensureStream();
    }
    await whenDelivering(sharedStream);
    // Recorded before the recording, because by the time a failure is drawn the track
    // has usually unmuted and the evidence has gone with it.
    lastMutedAtStart = !!micTrack()?.muted;
    const mime = chosenMime();
    this.chunks = [];
    this.meter = levelMeter(stream);       // only after something has gone wrong
    this.rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    this.gotData = new Promise((resolve) => { this.resolveData = resolve; });
    this.rec.ondataavailable = (e) => {
      if (e.data && e.data.size) this.chunks.push(e.data);
      this.resolveData();
    };
    this.rec.start();                      // no timeslice — see above
    this.startedAt = performance.now();
  }

  /** Returns { blob } on success, or { reason } describing why not. */
  async stop() {
    if (!this.rec) return { reason: 'no recorder — the mic never started' };
    const rec = this.rec;
    this.rec = null;

    const stopped = new Promise((resolve) => { rec.onstop = resolve; });
    // Let the tail of the final word land before cutting the recorder.
    await new Promise((r) => setTimeout(r, 250));
    const ms = Math.round(performance.now() - this.startedAt);
    if (rec.state !== 'inactive') rec.stop();
    await stopped;
    // Safari can deliver the last chunk after `stop`; give it a moment either way.
    await Promise.race([this.gotData, new Promise((r) => setTimeout(r, 1200))]);

    const peak = this.meter ? this.meter.peak() : null;
    this.meter?.stop();
    this.meter = null;

    const blob = new Blob(this.chunks, { type: rec.mimeType || 'audio/webm' });
    const verdict = capture.diagnose({
      ms, bytes: blob.size, chunks: this.chunks.length, mime: rec.mimeType, peak,
    });
    if (verdict.ok) {
      // It worked: stop guessing at the format on this device.
      capabilities.verify(rec.mimeType);
      return { blob };
    }
    // Sound was there and the container dropped it: that format is the culprit.
    if (verdict.block) capabilities.block(rec.mimeType);
    // Nothing reached us, or nothing can be said about why: the likelier cause is a
    // capture session that has been taken away, so let go of it and reopen.
    if (verdict.stale) markStreamStale();
    // And measure the next one, so a second failure can be attributed rather than
    // guessed at.
    if (verdict.meter) meterWanted = true;
    return { reason: verdict.reason };
  }
}

/* How a spoken answer is graded.

   Not by an absolute confidence. That was tried, twice, and the recordings settled it: on a
   learner's own voice the target line scored 0.766 while its own near-misses scored 0.784,
   so no cut separates them — and the level moves with the speaker anyway, which is why the
   first threshold (0.9867, calibrated on synthetic speech) never once fired on a real one.

   What survives is the *ordering*. Score the line we asked for against a field of other
   things the learner could have said, all against the same audio and the same denominator,
   and accept if it wins. On the same recordings where a threshold accepted none of 16
   correct answers, ranking accepted 75% of them.

   Ranking alone is not enough, and the negative controls are what showed it. Against an
   all-blank posterior a *shorter* sequence is the likelier reading, so silence and room
   noise both "won" against a field of longer alternatives — 0.346 and 0.589, accepted on a
   technicality. `capture.js` already turns away a recording with no signal in it, but the
   floor here is the second line: below it, nothing is accepted however it ranks.

   The margin exists for the same reason. One wrong answer beat its field by 0.006, which
   is noise, not evidence. Requiring daylight costs two thin-margin accepts out of fifteen
   and removes the false one.

   The trade is deliberate and was chosen knowingly: ranking accepts near-misses — say
   `irid` for `irrid` and it passes, because the model's posteriors genuinely cannot resolve
   consonant length in a learner's speech. Saying a *different* line is still turned away.
   For somebody learning, being able to progress beats a phonetic precision that no
   available Maltese model can actually judge. */
/* Swept against 25 clean recordings in a learner's voice, 175 wrong-line pairings, and 64
   clips of silence and room noise at several levels:

     floor  ranking   correct     wrong-line      silence/noise
      0.65    yes      19/25      3/175   2%       0/64    0%
      0.55    yes      22/25      4/175   2%       6/64    9%
      0.24     no      25/25    131/175  75%      20/64   31%

   0.55 is the chosen point: three more of a learner's correct answers accepted for no
   measurable change in wrong-line acceptance. It does let some room noise through, which
   `capture.js` is the first line against.

   Accepting all 25 was asked for and costed. It requires dropping the ranking test
   altogether — three of those clips *lose* their rank, the model preferring a different
   deck line — and at that point 75% of wrong answers and a third of all silence are marked
   correct. That is not a lenient grader, it is no grader, and it would feed false correct
   answers into the FSRS scheduler and rot the review deck. Declined on those numbers.

   One speaker, 25 utterances — a real measurement, not a large one. Re-run
   `--clips voice` after adding recordings and move this if the split has moved.

   ── and then it came down to 0.35, because the disease was cured ──

   Every word above is still true about the code as it was. What it does not say is *why*
   ranking accepted silence, and the answer turns out to be the same fault that was
   costing the learner their rank: a short sequence has fewer obligatory emissions and
   more freedom about where to put them, so it explains a blank posterior beautifully and
   a long utterance respectably. Silence winning against a field of longer alternatives
   and `Grazzi ħafna` losing to `Bonġu!` are one bug seen from two sides, and the floor
   was a patch over one side of it.

   `nanostt.rankScore` now charges a hypothesis for claiming a length the audio cannot
   support — a duration prior fitted on the 29,860 TTS passes of the distillation corpus.
   With the cause addressed the patch can come off. Swept together, on the same 25 clips
   and on 90 negatives (digital silence, white noise at five levels, the learner's own
   clips at -30dB, and the learner's own clips reversed):

     prior  floor   learner accepted   silence   hiss   -30dB   reversed
      off    0.55         20/25            0%      0%     0%       8%     ← was
      off    0.30         20/25            0%      5%     0%       8%
      on     0.55         22/25            0%      0%     0%       8%
      on     0.45         23/25            0%      0%     0%       8%
      on     0.35         24/25            0%      0%     0%       8%     ← is
      on     0.20         24/25            5%      0%     0%      12%
      on     0.00         24/25           10%      5%     0%      12%

   Four more of the learner's correct answers accepted, for *identical* rejection of
   everything that is not speech. Not a trade: the row above the old one on both counts.
   Lowering the floor on its own buys nothing and starts admitting hiss — the two have to
   move together, which is what makes this a fix rather than a loosening.

   Reversed speech at 8% is unchanged and is not a regression; it has full speech energy
   and speech-like spectra, and nobody plays speech backwards at their phone. */
const MIN_CONFIDENCE = 0.35;

/** How far ahead of the runner-up the target has to be. It changes nothing at this floor,
    and it costs nothing: correct answers clear their field by 0.06-0.43, where the one
    false accept seen at a lower floor cleared it by 0.006. Kept as insurance against a
    thin win this sample did not happen to contain. */
const MIN_MARGIN = 0.02;

/** How many of the field's own standard deviations the target also has to clear.

    `MIN_MARGIN` is an absolute distance, and the recordings show why that is only half a
    rule: correct answers cleared their field by 0.06-0.43 and the one false accept by
    0.006, which 0.02 happens to separate on this speaker. It is the same mistake the
    absolute confidence made one level up — a distance without a scale, where the scale
    moves with the speaker. `nanostt` now reports the spread of the 24 alternatives on
    this very recording, which is that scale, measured where it applies and needing no
    history.

    Zero, so the rule is exactly `MIN_MARGIN` and nothing else. Turning it up can only
    refuse answers this floor currently accepts, and which ones is not something 25
    recordings of one speaker can settle — it needs the sweep in
    `constrained_ctc.py --clips voice` against those clips and the 90 negatives, the same
    way every other constant in this block was chosen. */
const MARGIN_SIGMAS = 0;

/** What share of the ranking field to draw from the scene being spoken.

    The rest comes from the whole script, as all of it does today. The case for it is that
    a plausible alternative is a better test than an implausible one, and the case for
    leaving it at zero is that it changes which answers are accepted: a harder field is
    stricter, so this trades accepts for false-accept resistance, and that trade is
    exactly what the floor sweep exists to price. Note also that the scene's field is
    small — a dozen or two lines — so a high share shrinks the effective field, and
    ranking against fewer alternatives is a different change wearing the same clothes. */
const FIELD_LOCAL = 0;

/** How many alternatives to rank against. Each is one CTC forward pass over posteriors
    already computed — about a millisecond — so this is bounded by patience, not cost. */
const RANK_AGAINST = 24;

/** A field of plausible other answers, sampled fresh so a learner cannot be unlucky in the
    same way twice. The target is excluded by the caller.

    Empty on the FastAPI dev build, which never loads the whole script — it asks the server
    per turn, where the static build ships all of it. With no field to rank against the
    acceptance falls back to the floor alone, which is weaker but not wrong; the deployed
    build is the static one. */
/* Every accepted line in the script, normalised once.

   This was rebuilt on every utterance: 113 nodes walked, 377 strings normalised
   through the Maltese rules, and the whole array shuffled — to keep 24 of it. That
   is a few milliseconds of regex immediately before the model runs, on the device
   least able to spare them, to produce an answer that cannot change: the script is
   loaded at boot and never edited. */
let answerPool = null;

/** Partial Fisher-Yates: draw `want` without replacement and stop, rather than
    shuffling the whole pool to read the front of it. */
function drawFrom(pool, want, target, taken) {
  const rest = pool.slice();
  const out = [];
  for (let n = rest.length; out.length < want && n > 0; n -= 1) {
    const j = Math.floor(Math.random() * n);
    const pick = rest[j];
    rest[j] = rest[n - 1];
    if (pick !== target && !taken.has(pick)) {
      taken.add(pick);
      out.push(pick);
    }
  }
  return out;
}

function distractorsFor(target, scene) {
  if (!answerPool) {
    answerPool = dialogueEngine.everyAnswer()
      .map((line) => mtext.normalise(line).toLowerCase().trim())
      .filter(Boolean);
  }
  const taken = new Set();
  const out = [];
  if (FIELD_LOCAL > 0 && scene) {
    const local = dialogueEngine.answersIn(scene)
      .map((line) => mtext.normalise(line).toLowerCase().trim())
      .filter(Boolean);
    out.push(...drawFrom(local, Math.round(FIELD_LOCAL * RANK_AGAINST), target, taken));
  }
  // Topped up from the whole script, so a thin scene never shrinks the field.
  out.push(...drawFrom(answerPool, RANK_AGAINST - out.length, target, taken));
  return out;
}

async function transcribe(blob, target, scene) {
  /* On-device only when there is nowhere better to send it. The model is the
     heaviest thing this app can do and the least reliable place to do it. */
  if (!remoteStt() && state.settings.local_stt && nanostt.isReady()) {
    try {
      /* Hand the recogniser the line we asked for. Deciding whether the audio *is* that
         line is a far easier question than transcribing it, and it is the only question
         the app has — so the transcript stops being the verdict and becomes the
         explanation for when the answer was wrong. */
      const flat = target ? mtext.normalise(target).toLowerCase().trim() : '';
      const r = await nanostt.transcribe(blob, {
        target: flat,
        distractors: flat ? distractorsFor(flat, scene) : [],
      });
      const need = Math.max(MIN_MARGIN, MARGIN_SIGMAS * (r.fieldSd || 0));
      const clear = r.runnerUp === null || r.rank > r.runnerUp + need;
      if (flat && clear && r.confidence >= MIN_CONFIDENCE) {
        /* The line we asked for explains this audio better than anything else it could
           have been. A garbled transcript here is the model failing at the harder task,
           not the learner failing at the easier one — so it is not shown back to them as
           though they had said it. */
        return { ...r, text: mtext.normalise(target), assessment: mtext.assess(target, target) };
      }
      /* Passed the floor, lost the field — and the sounds are the ones this line asks
         for. That is a learner who nearly said it, not audio that is not the line: the
         floor cannot tell those apart (it admits 67% of time-reversed clips) and the
         field cannot either (a near-miss is ranked behind an unrelated answer by a median
         of 0.648). Per-sound scoring can, so it decides this middle case, and the credit
         is the *near* one the app already has rather than a clean pass. */
      if (flat && r.confidence >= MIN_CONFIDENCE
          && Number.isFinite(r.gop) && r.gop >= nanostt.GOP_MIN) {
        return {
          ...r,
          text: mtext.normalise(target),
          assessment: mtext.assess(target, target),
          nearSound: true,
        };
      }
      if (r.text.trim()) {
        // Grading is a few microseconds of string comparison, but it lives on the
        // server with the Maltese rules, so it is one small stateless request.
        if (target) {
          r.assessment = STATIC.on
            ? mtext.assess(r.text, target)
            : await post('/api/attempt', { said: r.text, target }).catch(() => null);
        }
        return r;
      }
    } catch (err) {
      console.warn('local recogniser failed, using the server', err);
    }
  }
  if (STATIC.on && !remoteStt()) {
    /* Nowhere to send it: this build has no server and the device could not hold the
       model. Say so, rather than posting into a 404 and reporting whatever that
       returns as though the recording were at fault. */
    throw new Error('No recogniser available here yet — type your answer for now.');
  }
  const fd = new FormData();
  // Named for what it is: the server picks its decoder from the content type and the
  // extension, and an mp4 clip called speech.webm is a decode failure waiting to
  // happen — which is exactly what iOS records.
  fd.append('audio', blob, capture.fileNameFor(blob.type));
  if (target) fd.append('target', target);
  const r = await api(`${remoteStt()}/api/stt`, { method: 'POST', body: fd });
  /* Grading stays here. It is microseconds of string work against rules that are
     already in the page, and sending it away would add a round trip to a decision
     the browser can make itself. */
  if (target && STATIC.on && !r.assessment) r.assessment = mtext.assess(r.text, target);
  return r;
}

/** Turn local recognition on or off. Loading is the expensive half, so progress
    is reported rather than left as a frozen toggle. */
async function setLocalStt(on) {
  const box = $('localStt');
  const note = $('localSttNote');
  state.settings.local_stt = on;
  persistSettings();
  if (!on) {
    nanostt.unload();
    note.textContent = remoteStt() || !STATIC.on
      ? 'Speech is sent to the server for recognition.'
      : 'Speech recognition is off, and this build has no server to do it.';
    return;
  }
  if (!nanostt.supported()) {
    state.settings.local_stt = false;
    persistSettings();
    box.checked = false;
    note.textContent = 'This browser cannot run WebAssembly, so recognition has to '
      + 'happen on the server. Left off.';
    return;
  }
  box.disabled = true;
  try {
    await nanostt.load({
      onProgress: (f) => {
        note.textContent = f >= 1
          ? 'Loaded — recognition now runs on this device.'
          : `Downloading the recogniser… ${Math.round(f * 100)}%`;
      },
    });
    note.textContent = 'Loaded — recognition now runs on this device, offline.';
  } catch (err) {
    state.settings.local_stt = false;
    persistSettings();
    box.checked = false;
    note.textContent = `Could not load it: ${err.message}. `
      + (remoteStt() ? 'Using the server instead.' : 'Typing still works.');
  } finally {
    box.disabled = false;
  }
}

/** Wires push-to-talk (hold) and tap-to-toggle onto a mic button. */
function bindMic(button, { onResult, onStatus, target }) {
  const recorder = new Recorder();
  let active = false;
  let starting = false;
  let holdTimer = null;
  let isHold = false;

  const setBusy = (busy) => button.classList.toggle('is-busy', busy);

  /* The finger let go before the microphone was open. Remembered rather than
     dropped: `end()` used to return on `!active` and that is the state a press
     spends its first 100-500ms in, so on a cold page the whole first utterance —
     press, speak, release — went into a recorder that had not started yet and a
     release that nothing acted on. The button then quietly turned red on its own,
     and the *second* press was what sent anything. Reported as "fails the first
     time, works the second", and it happens on every page load, not once per
     device. */
  let releasedEarly = false;

  async function begin() {
    // `active` cannot be the only guard: it is set after the await below, so two
    // quick taps both got past it and started a second MediaRecorder over the top
    // of the first, which then ran on unreferenced and unstoppable.
    if (active || starting || button.classList.contains('is-busy')) return;
    starting = true;
    releasedEarly = false;
    recordingNow = true;
    // Feedback before the await, not after it. Half a second of a button that does
    // nothing is what teaches somebody to press it twice.
    button.classList.add('is-recording');
    onStatus?.('Opening the microphone…');
    /* No waiting on the probe. It used to record 300ms through this very microphone,
       so a press landing inside it put two MediaRecorders on one capture session — and
       the await that prevented that cost the first press up to two seconds of the
       utterance. The probe runs on a synthetic stream now and cannot collide with
       this, so the recording starts immediately and the probe lands when it lands. */
    try {
      await recorder.start();
      active = true;
      onStatus?.('Listening… (release to send)');
    } catch (err) {
      recordingNow = false;
      button.classList.remove('is-recording');
      onStatus?.('');
      toast(err.message);
      return;
    } finally {
      starting = false;
    }
    /* Released while we were still opening. Send what there is: the microphone came
       up partway through the utterance, so the tail of it is real audio and the
       recogniser is entitled to see it. If it caught nothing at all, `stop()` says
       so in the words of the failure rather than in silence. */
    if (releasedEarly) await end();
  }

  async function end() {
    if (starting) { releasedEarly = true; return; }   // begin() will call us
    if (!active) return;
    active = false;
    button.classList.remove('is-recording');
    setBusy(true);
    onStatus?.('Transcribing…');
    try {
      const { blob, reason } = await recorder.stop();
      // Say what actually went wrong. "Too short" was a guess the code made about
      // the learner, and twice it was wrong about itself instead.
      if (!blob) { onStatus?.(`Nothing recorded — ${reason}${captureState()}`); return; }
      const result = await transcribe(blob, typeof target === 'function' ? target() : target,
                                      drill.dialogue);
      onStatus?.('');
      await onResult(result);
    } catch (err) {
      onStatus?.('');
      toast(`Could not transcribe: ${err.message}`);
    } finally {
      setBusy(false);
      recordingNow = false;
      /* The probe stood aside for this recording, so if it never got to run — first
         press on a device nothing is known about — find out now, while nobody is
         holding the button. A successful recording has already answered the question
         by verifying its own container, so this is only for the ones that failed. */
      if (!capabilities.verified()) prewarmMic();
    }
  }

  let pendingStop = false;

  button.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    // Keep every later event for this finger aimed at the button. Without it a
    // few pixels of drift off a 54px target — which is most of what a thumb does
    // on a phone — fires pointerleave and cuts the recording mid-word.
    try { button.setPointerCapture(e.pointerId); } catch { /* older Safari */ }
    if (active) { pendingStop = true; return; }   // second tap of a toggle
    isHold = false;
    pendingStop = false;
    // Start capturing immediately. Waiting even 140ms here clipped the first
    // syllable off every utterance; hold-vs-tap is resolved on release instead,
    // which costs no audio.
    begin();
    holdTimer = setTimeout(() => { isHold = true; }, 140);
  });

  const release = () => {
    clearTimeout(holdTimer);
    if (pendingStop) { pendingStop = false; end(); return; }
    if (isHold) { end(); return; }   // held down: push-to-talk, release sends
    // Quick tap: keep recording until the next tap.
  };
  button.addEventListener('pointerup', (e) => { e.preventDefault(); release(); });
  // iOS fires this if it decides the gesture was a scroll after all. Send what we
  // have rather than dropping it silently.
  button.addEventListener('pointercancel', () => { if (active) end(); });
  // Only a mouse leaving the button means "stopped pressing". A touch that moves
  // is still a touch, and with pointer capture it will not fire this anyway.
  button.addEventListener('pointerleave', (e) => {
    if (e.pointerType === 'mouse' && active && isHold) end();
  });

  return { begin, end, isActive: () => active };
}

/* ── Boot ──────────────────────────────────────────────────────────────── */

async function boot() {
  let data;
  try {
    data = await splash.run({
      onDeck: (cards) => store.seedDeck(cards),
      onStatic: ({ boot, dialogues, audio }) => {
        STATIC.on = true;
        STATIC.audio = audio;
        STATIC.sttBase = boot.stt_base || '';
        dialogueEngine.load(dialogues);
      },
      // Whatever the startup screen learned that the learner should be told once
      // the app is open — a recogniser still waking, most of it.
      onNotice: (msg) => { state.sttNotice = msg; },
      /* Only reached when recognition is this device's job: a build pointed at
         `stt_base` has nothing to fetch — which is the whole point of pointing it
         there — and the startup screen wakes that host itself. */
      onModel: async (onProgress) => {
        // Only worth waiting on where it is the only recogniser there is.
        const settings = store.loadSettings();
        if (!settings.local_stt || !nanostt.supported()) return false;
        /* The device-memory guards that used to live here are gone with the model they
           were written for. A 2.1MB recogniser on the CPU cannot exhaust a page budget
           the way 200MB of weights on the GPU did, so there is no longer a class of
           phone that has to be refused, and nothing to remember about one that was.

           Bounded, and then left to finish on its own. The deck, the conversation and
           typing do not need it, and `isReady()` is checked at the moment something is
           said — so a late arrival simply starts working. */
        const loading = nanostt.load({ onProgress })
          .then(() => true)
          .catch(() => false);      // typing still works; the toggle explains why
        const waited = await Promise.race([
          loading,
          new Promise((done) => setTimeout(() => done('slow'), MODEL_WAIT_MS)),
        ]);
        if (waited !== 'slow') return waited;
        state.sttNotice = 'Still fetching the speaking recogniser — carry on, '
          + 'it will start working when it arrives.';
        return false;
      },
    });
  } catch (err) {
    splash.fail(err.message);
    return;
  }
  state.caps = data.capabilities;
  // Server defaults fill in only what the learner has never set. Their own
  // choices are on this device and must not be overwritten by a redeploy.
  state.settings = { ...data.defaults, ...store.loadSettings() };
  store.saveSettings(state.settings);

  applySettings();
  renderCaps();
  if (state.sttNotice) {
    toast(state.sttNotice, 9000);
    state.sttNotice = '';
  }
  await refreshCounts();

  await loadDrills();
  loadGrammar();

  // Opted in on a previous visit: warm it in the background so the first thing they
  // say is not the slow one. Static builds already loaded it during startup; a server
  // build has a recogniser of its own, so this warms rather than blocking.
  if (!remoteStt() && state.settings.local_stt && nanostt.supported()
      && !nanostt.isReady()) {
    nanostt.load().catch(() => { /* the server path still works */ });
  }
}

/** Counts and level come from the local database now, so they are recomputed
    rather than handed back by an endpoint. */
async function refreshCounts() {
  const c = await schedule.counts();
  updateCounts(c);
  $('levelChip').textContent = schedule.estimateLevel(c.learned);
}

function applySettings() {
  $('voiceSelect').value = state.settings.voice;
  $('rateRange').value = state.settings.rate;
  $('rateLabel').textContent = `${Number(state.settings.rate).toFixed(2)}×`;
  $('showEnglish').checked = state.settings.show_english;
  $('autoplay').checked = state.settings.autoplay;
  $('localStt').checked = !remoteStt() && !!state.settings.local_stt;
  /* Offered only where it would be used: `transcribe` sends the utterance to
     `stt_base` whenever there is one, so a switch here would be ignored. */
  $('localStt').disabled = !!remoteStt() || !nanostt.supported();
  $('localSttNote').textContent = localSttHint();
}

/** What the on-device toggle should say about itself. A toggle that explains none of
    the reasons it might be off is a toggle that reads as broken. */
function localSttHint() {
  if (remoteStt()) return 'Speech is recognised on the server for this deployment.';
  if (!nanostt.supported()) {
    return 'Needs WebAssembly — not available in this browser.';
  }
  return state.settings.local_stt
    ? 'On. Recognition runs here, offline, on a 2MB Maltese model.'
    : 'Off. Speech is sent to the server for recognition.';
}

function renderCaps() {
  const c = state.caps;
  const mark = (ok) => (ok ? '<span class="ok">✓</span>' : '<span class="off">✗</span>');
  $('capsBox').innerHTML = `
    <div>${mark(c.tts.length)} <b>Speech out</b> — ${c.tts.join(', ') || 'none'}</div>
    <div>${mark(c.stt.length)} <b>Speech in</b> — ${c.stt.join(', ') || 'none'}</div>`;
}

function persistSettings() {
  store.saveSettings(state.settings);
}

function updateCounts(counts) {
  if (!counts) return;
  const n = counts.due + Math.min(counts.new, 12);
  /* Three of them: the tab (wide screens), the hamburger and the sheet's Review
     row. On a phone the tab row is gone, so a count that only lived there would
     mean nothing told the learner there was anything to review. */
  for (const id of ['dueBadge', 'navBadge', 'sheetBadge']) {
    const badge = $(id);
    if (!badge) continue;
    badge.textContent = n;
    badge.hidden = n === 0;
  }
}

/* ── Drill: scripted conversation ──────────────────────────────────────────
   No model in the loop. The reply is chosen by phonetic match on the server in
   under a millisecond, and its audio is already cached, so the only wait is
   speech recognition. */

/* Per-run tally, so finishing a scene ends with something to look at rather than
   just stopping. Written once: it was spelled out at three call sites and the
   `peeked` bucket would have had to be added to all three — which is how the one
   that restores a saved conversation ends up a field behind the others. */
function freshRun() {
  return { first: 0, retried: 0, movedOn: 0, peeked: 0, learned: [], startedAt: Date.now() };
}

const drill = {
  dialogue: null, node: null, busy: false, attempts: 0, dialogues: [],
  // Whether the answer was on screen before it was given. See `toggleAnswer`.
  peeked: false,
  run: freshRun(),
  // What the node asked for, kept as the server or the engine gave it, so the
  // conversation can be restored without asking either of them again.
  present: null,
  // Every bubble on screen, in order. The DOM used to be the only copy.
  turns: [],
};

/** The conversation, so a reload does not restart the scene. */
const saved = session.store();

/** Write the conversation down. Called after anything that adds to the screen. */
function keepDrill() {
  if (!drill.dialogue || !drill.present) return;
  saved.save({
    dialogue: drill.dialogue,
    node: drill.node,
    attempts: drill.attempts,
    // Elapsed rather than the start time, so a scene picked up tomorrow does not
    // report the night as time spent answering.
    run: { ...drill.run, elapsedMs: Date.now() - drill.run.startedAt },
    present: drill.present,
    turns: drill.turns,
  });
  // Called after anything that adds to the screen, which is also exactly when the
  // header's "2/4" moves.
  updateDrillHead();
}

/** Scenes you have finished, so the path can show progress across sessions. */
const doneScenes = {
  key: 'sm.completedScenes',
  all() { try { return JSON.parse(localStorage.getItem(this.key)) || {}; } catch { return {}; } },
  mark(id) {
    const a = this.all();
    a[id] = (a[id] || 0) + 1;
    localStorage.setItem(this.key, JSON.stringify(a));
  },
};

async function loadDrills() {
  const { dialogues } = STATIC.on
    ? { dialogues: dialogueEngine.all().map(({ id, name, name_en, level, nodes }) =>
        ({ id, name, name_en, level, steps: Object.keys(nodes || {}).length })) }
    : await api('/api/drill/dialogues');
  drill.dialogues = dialogues;
  renderScenePath();
  if (drill.dialogue) return;
  if (restoreDrill(dialogues)) return;
  await startDrill(dialogues[0]?.id);
}

/** Put back the conversation from last time, if there is one to put back.

    A reload used to restart the scene from its first line — on a phone, where the
    browser reloads a backgrounded tab whenever it likes, that could happen without
    the learner doing anything at all. */
function restoreDrill(dialogues) {
  const s = saved.load();
  if (!s || !s.turns.length || !dialogues.some((d) => d.id === s.dialogue)) return false;

  /* Ask the engine for the prompt again rather than trusting the copy in storage.
     The stored one is from whichever build wrote it, and a saved conversation
     outlives a deploy — so replaying it would show the old build's prompt beside
     the new build's grading, which is the very mismatch the versioned shell cache
     was added to stop. It also answers whether the node still exists: a build can
     rename or drop one, and a conversation sitting on a node that is gone grades
     every answer as "unknown node" — a chat you can read and cannot continue.

     Only the engine can be asked. Against the server, the stored prompt is all
     there is, and a node that has gone is caught when an answer is sent. */
  const present = STATIC.on ? dialogueEngine.present(s.dialogue, s.node) : s.present;
  if (!present) return false;

  drill.dialogue = s.dialogue;
  drill.turns = s.turns;
  drill.run = { ...freshRun(), ...(s.run || {}) };
  // The clock counts time spent in the scene, not time the tab spent closed.
  drill.run.startedAt = Date.now() - (s.run?.elapsedMs || 0);
  $('drillChat').innerHTML = '';
  for (const t of s.turns) drillBubble(t.role, t.mt, t.en, t);
  // Silently: the tutor is not going to read the whole conversation back, and the
  // node is where the learner left off rather than a line just spoken.
  installDrillNode(present);
  // After installing, which starts a node's attempts at nought: two tries already
  // spent are two the learner does not have to spend again to be moved on.
  drill.attempts = s.attempts || 0;
  markCurrentScene();
  updateDrillHead();
  return true;
}

/** The learning path: every scene as a card, grouped by level, with its picture.

    This used to be a `<select>` sitting above the conversation. Thirty-five scenes
    in a flat list tells a learner nothing about where they are or what comes next,
    and it charged every conversation about 76px for the privilege — on an iPhone SE
    that is an eighth of the screen. As its own screen it can afford the pictures,
    which is where they were always worth showing: this is the point at which you
    are choosing between scenes, so the scenes should be recognisable.

    Ordered so the vocabulary builds, and the first unfinished scene is flagged as
    the obvious next thing. */
function renderScenePath() {
  const done = doneScenes.all();
  const nextUp = drill.dialogues.find((d) => !done[d.id]);
  const levels = [...new Set(drill.dialogues.map((d) => d.level))].sort();

  $('scenePath').innerHTML = levels.map((lvl) => {
    const cards = drill.dialogues.filter((d) => d.level === lvl).map((d) => {
      const state = done[d.id] ? 'is-done'
        : d.id === nextUp?.id ? 'is-next' : '';
      // Only where there is something to say. A badge on every card — the turn
      // count, say — reads as an unread count and makes the three that matter
      // invisible.
      const flag = done[d.id] ? '✓ Done' : d.id === nextUp?.id ? 'Next' : '';
      return `<button class="scene-card ${state}" data-scene="${escapeHtml(d.id)}">
          <img src="img/scene-${escapeHtml(d.id)}.webp" alt="" loading="lazy"
               decoding="async" width="320" height="160">
          ${flag ? `<span class="flag">${escapeHtml(flag)}</span>` : ''}
          <span class="body">
            <span class="mt">${escapeHtml(d.name)}</span>
            <!-- The English name and nothing else. A turn count appended here
                 wrapped half the cards onto a third line, and the conversation's
                 own header says how long the scene is once you are in it. -->
            <span class="en">${escapeHtml(d.name_en)}</span>
          </span>
        </button>`;
    }).join('');
    return `<h3 class="path-level">${escapeHtml(lvl)}</h3>
            <div class="path-grid">${cards}</div>`;
  }).join('');

  for (const card of $('scenePath').querySelectorAll('.scene-card')) {
    card.onclick = () => openScene(card.dataset.scene);
    // A card with no art is a card, not a broken-image icon.
    const img = card.querySelector('img');
    if (img) img.onerror = () => img.remove();
  }

  const finished = drill.dialogues.filter((d) => done[d.id]).length;
  $('pathProgress').textContent = nextUp
    ? `${finished} of ${drill.dialogues.length} done · next: ${nextUp.name_en}`
    : `All ${drill.dialogues.length} done — go round again, they get easier.`;

  markCurrentScene();
  updateDrillHead();
}

/** Which card is the conversation you are in the middle of. Separate from the rest
    of the path so switching scenes does not have to rebuild all thirty-five. */
function markCurrentScene() {
  for (const card of $('scenePath').querySelectorAll('.scene-card')) {
    const here = card.dataset.scene === drill.dialogue;
    card.classList.toggle('is-current', here);
    /* "Here" beats "Next" on the scene you are actually in, and a finished scene
       keeps its tick — going round again is the point of a finished one. Added as
       its own element rather than by overwriting the flag that is there: rewriting
       it lost the "Next" mark permanently as soon as you moved on. */
    const mark = card.querySelector('.flag.here');
    if (here && !card.classList.contains('is-done')) {
      if (!mark) {
        const el = document.createElement('span');
        el.className = 'flag here';
        el.textContent = 'Here';
        card.append(el);
      }
    } else if (mark) {
      mark.remove();
    }
  }
}

/** Tapping a card. Carrying on where you left off is the common case, so the same
    scene resumes rather than restarting — throwing away a conversation is what
    "Clear · start over" is for, and it should take saying so. */
function openScene(id) {
  switchView('drill');
  if (id === drill.dialogue && drill.turns.length) return;
  startDrill(id);
}

/** The conversation's header: the scene's name, and how far into it you are. */
function updateDrillHead() {
  const meta = drill.dialogues.find((d) => d.id === drill.dialogue);
  $('drillSceneName').textContent = meta ? meta.name_en : 'Scenes';
  const step = $('drillStep');
  // Every tutor line that is a prompt rather than a reply is one turn of the scene.
  const at = drill.turns.filter((t) => t.role === 'tutor' && !t.verdict).length;
  step.hidden = !(meta?.steps && at);
  if (!step.hidden) step.textContent = `${Math.min(at, meta.steps)}/${meta.steps}`;
}

/** Jump to the first scene not yet completed. */
function goToNextScene() {
  const done = doneScenes.all();
  const next = drill.dialogues.find((d) => !done[d.id]) || drill.dialogues[0];
  if (next) startDrill(next.id);
}

/** Start a scene from its first line, throwing away whatever was on screen. This
    is what "Start over" does, and the only way a saved conversation is discarded
    on purpose. */
async function startDrill(id) {
  if (!id) return;
  drill.dialogue = id;
  drill.run = freshRun();
  drill.turns = [];
  drill.present = null;
  saved.clear();
  $('drillChat').innerHTML = '';
  markCurrentScene();
  const node = STATIC.on
    ? dialogueEngine.start(id)
    : await post('/api/drill/start', { dialogue: id });
  presentDrillNode(node);
}

/** Point the composer at a node without saying anything: the prompt line, the
    frame being asked for, and where an answer will be graded. Shared by a fresh
    turn and by a conversation restored from a previous visit. */
function installDrillNode(node) {
  drill.node = node.node;
  drill.attempts = 0;
  drill.present = node;
  // An open question is scored on its Maltese frame, so the frame is on the screen
  // next to the English. Saying "say your name — anything goes" and then marking
  // `Jien …` asked the learner to guess which half was being looked at.
  // Each half stands on its own: a node with a frame and no English still shows the
  // frame, because the frame is the half that gets marked.
  const parts = [];
  if (node.expect_en) parts.push(escapeHtml(node.expect_en));
  if (node.frames?.length) parts.push(node.frames.map((f) => `<b>${escapeHtml(f)}</b>`).join(' / '));
  $('drillExpect').innerHTML = parts.length ? `→ ${parts.join(' · ')}` : '';
  hideAnswer();
}

/* ── Showing the answer before it is asked for ─────────────────────────────
   Somewhere to look when you have no idea. Until now the only way to find out what
   to say was to get it wrong twice and be shown it by the backstop — which teaches
   the shape of a wrong answer twice before the right one, and on a phone means two
   rounds of speaking into a microphone hoping.

   A peek is recorded, not punished. The answer still has to be produced, and the
   turn still advances — but it is not filed into the review deck as a phrase this
   learner *produced*, because they read it off the screen a moment earlier and FSRS
   would schedule it as known. That is the same reasoning `moved_on` already uses. */
function hideAnswer() {
  drill.peeked = false;
  $('drillAnswer').hidden = true;
  $('drillReveal').textContent = 'Show me';
  // Only offered where there is a line to show. An open question has one; a node
  // whose accepted answers are all open frames does not.
  $('drillReveal').hidden = !drill.present?.answer?.mt;
}

function toggleAnswer() {
  const answer = drill.present?.answer;
  if (!answer?.mt) return;
  const box = $('drillAnswer');
  if (!box.hidden) {                       // second press: put it away again
    box.hidden = true;
    $('drillReveal').textContent = 'Show me';
    return;
  }
  /* Their name, their town, their age: there is no single right answer, so what is on
     screen is a pattern and an example of filling it in — not a line to repeat.

     Saying that clearly matters more than it looks. The examples carry a name, and a
     learner who sees `Jisimni Silas` for one pattern and `Jien Sally` for another can
     reasonably conclude that `Jien` is the women's form. Maltese has plenty of
     masculine and feminine pairs, so the inference is a sensible one — it is simply
     wrong here: `Jien …` and `Jisimni …` are interchangeable, and the only thing that
     changes with the speaker is the name in the gap.

     So the patterns go on their own line, together, unlabelled and in the order the
     scene lists them, and the example below is marked as one. */
  const frames = drill.present.free ? (drill.present.frames || []) : [];
  const framesEl = $('drillAnswerFrames');
  framesEl.hidden = !frames.length;
  framesEl.textContent = frames.join('   ·   ');

  $('drillAnswerLead').textContent = !drill.present.free ? 'Say'
    : frames.length > 1 ? 'Either pattern — your own answer in the gap'
      : frames.length ? 'This pattern — your own answer in the gap'
        : 'Something like';
  $('drillAnswerMt').textContent = frames.length ? `e.g. ${answer.mt}` : answer.mt;
  $('drillAnswerEn').textContent = answer.en || '';
  box.hidden = false;
  $('drillReveal').textContent = 'Hide';
  drill.peeked = true;
  speak(answer.mt);
}

/** Install a node and say its line — a new turn, as opposed to a restored one.

    The picture is named after the turn rather than the scene, and stored on the turn
    rather than looked up when drawing: a conversation restored from storage has to
    come back with the same pictures beside the same questions, and only the turn
    knows which node it was. */
function presentDrillNode(node) {
  installDrillNode(node);
  const turn = {
    role: 'tutor', mt: node.say_mt, en: node.say_en,
    art: `${drill.dialogue}-${node.node}`,
  };
  drillBubble(turn.role, turn.mt, turn.en, turn);
  drill.turns.push(turn);
  keepDrill();
  if (state.settings.autoplay) speak(node.say_mt);
}

function showRunSummary() {
  const { first, retried, movedOn, peeked, learned, startedAt } = drill.run;
  const total = first + retried + movedOn + peeked;
  const secs = Math.max(1, Math.round((Date.now() - startedAt) / 1000));
  const scene = drill.dialogues.find((d) => d.id === drill.dialogue);
  const phrases = learned.slice(0, 8)
    .map((p) => `<li><span class="mt">${escapeHtml(p.mt)}</span>
                     <span class="en">${escapeHtml(p.en || '')}</span></li>`).join('');

  const el = document.createElement('div');
  el.className = 'turn tutor run-summary';
  el.innerHTML = `
    <div class="bubble">
      <p class="mt">Spiċċajna. Prosit!</p>
      <p class="en">That's the scene finished — well done.</p>
      <div class="summary-stats">
        <span><b>${first}</b> first try</span>
        <span><b>${retried}</b> on a retry</span>
        ${peeked ? `<span><b>${peeked}</b> after a look</span>` : ''}
        ${movedOn ? `<span><b>${movedOn}</b> waved through</span>` : ''}
        <span><b>${Math.round(secs / Math.max(1, total))}s</b> a turn</span>
      </div>
      ${phrases ? `<p class="summary-label">Added to your review deck</p>
                   <ul class="summary-phrases">${phrases}</ul>` : ''}
      <div class="summary-actions">
        <button class="ghost-btn" data-again>Again</button>
        <button class="primary-btn" data-next>Next scene</button>
      </div>
    </div>`;
  $('drillChat').append(el);
  $('drillChat').scrollTop = $('drillChat').scrollHeight;
  speak(scene ? 'Prosit! Spiċċajna.' : 'Prosit!');

  el.querySelector('[data-again]').onclick = () => startDrill(drill.dialogue);
  el.querySelector('[data-next]').onclick = goToNextScene;
}

/** One bubble. `verdict` and `target` are the marking above and the line to repeat
    below it; passing them here rather than bolting them on afterwards is what lets
    a restored conversation come back looking exactly like the one you left. */
function drillBubble(role, mt, en, { extraClass = '', verdict = null, target = null,
                                     art = '' } = {}) {
  const el = document.createElement('div');
  el.className = `turn ${role} ${extraClass}${art ? ' has-art' : ''}`;
  el.innerHTML = `
    <div class="bubble">
      ${verdict ? `<p class="drill-verdict ${escapeHtml(verdict.tone || '')}">${
        escapeHtml(verdict.mark || '')} ${escapeHtml(verdict.text || '')}</p>` : ''}
      <p class="mt">${escapeHtml(mt || '')}</p>
      ${en ? `<p class="en" ${state.settings.show_english ? '' : 'hidden'}>${escapeHtml(en)}</p>` : ''}
      ${target ? `<p class="drill-target">${escapeHtml(target.mt)}
           <em>${escapeHtml(target.en || '')}</em>
           <span class="target-tools">
             <button class="tool" data-target-play>🔊 Play</button>
             <button class="tool" data-target-slow>🐢 Slow</button>
           </span></p>` : ''}
      ${/* One pair of controls per bubble, under the line it speaks.

            Giving the target line its own Play left two pairs stacked at the bottom of
            a correction, and the lower one was the *reply's* — so the buttons directly
            beneath `Nitkellem ftit Malti.` played `Kważi. Għid: Nitkellem ftit Malti.`
            instead. Position is the only thing saying which control belongs to which
            line, and it was saying the wrong thing.

            The reply's pair is the one to drop. A correction reads `Kważi. Għid:` and
            then the target, so the target contains everything in it worth hearing
            again — and the reply has just been read aloud by autoplay. */
        role === 'tutor' && mt && !target ? `<div class="bubble-tools">
          <button class="tool" data-play>🔊 Play</button>
          <button class="tool" data-slow>🐢 Slow</button>
        </div>` : ''}
    </div>
    ${art ? `<img class="turn-art" src="img/turn-${escapeHtml(art)}.webp" alt=""
                  decoding="async" width="320" height="320">` : ''}`;
  if (role === 'tutor' && mt && !target) {
    el.querySelector('[data-play]').onclick = () => speak(mt);
    el.querySelector('[data-slow]').onclick = () => speak(mt, { rate: 0.7 });
  }
  /* The line the learner is being asked to say, on its own. The bubble's own Play
     speaks the *reply* — `Kważi. Għid: …` — which buries the target inside a sentence
     and after a word of Maltese the learner has just been told they got wrong. It was
     the one set response with no way to hear it by itself. */
  if (target?.mt) {
    el.querySelector('[data-target-play]').onclick = () => speak(target.mt);
    el.querySelector('[data-target-slow]').onclick = () => speak(target.mt, { rate: 0.7 });
  }
  /* A question with no picture is a question, not a broken-image icon. The class
     goes too, so the bubble takes the full width back rather than leaving a gap
     where the square was. */
  const img = el.querySelector('.turn-art');
  if (img) img.onerror = () => { img.remove(); el.classList.remove('has-art'); };
  const chat = $('drillChat');
  chat.append(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
}

async function answerDrill(said, opts = {}) {
  said = (said || '').trim();
  if (!said || drill.busy) return;
  drill.busy = true;
  $('drillInput').value = '';
  drillBubble('user', said, '');
  drill.turns.push({ role: 'user', mt: said });

  let holdBusy = false;
  try {
    const t0 = performance.now();
    const r = STATIC.on
      ? dialogueEngine.evaluate(drill.dialogue, drill.node, said, drill.attempts)
      : await post('/api/drill/answer', {
        dialogue: drill.dialogue, node: drill.node, said, attempts: drill.attempts,
      });
    // The engine reports an unknown node in the body where the server reports it as
    // a 404; raising here puts both on the same path out.
    if (r.error) throw new Error(r.error);
    /* An acoustic near-miss is credited the way `on_lead` is: the learner is not told
       they were wrong, and they are not told they were clean either. Reusing that state
       rather than adding a fourth keeps one meaning for the amber mark. */
    if (opts.nearSound) r.on_lead = true;
    if (!r.advance) drill.attempts += 1;
    const ms = Math.round(performance.now() - t0);

    // `on_lead` is a pass, and it is not a clean one: the transcript did not reach the
    // bar, and what carried it was being nearer to this answer than to anything else
    // the script accepts. Marked as the near tone with the line shown beside it, so the
    // learner can hear the difference between what they were credited with and what
    // came back — printing a plain ✓ over a mangled transcript teaches the mangling.
    const tone = (r.moved_on || r.on_lead) ? 'near'
      : { correct: 'ok', close: 'near', wrong: 'bad' }[r.verdict];
    const mark = r.moved_on ? '→' : { correct: '✓', close: '≈', wrong: '✗' }[r.verdict];
    // An open question is your name, your town, your age: accepted whatever you
    // say, because the app cannot know it. What is scored is the Maltese *frame*
    // around the slot — `Jien …`, `Għandi … sena` — and never the name or age in
    // it. Worth showing once the frame is there; below half it is reporting the
    // absence of one, and "correct · 15%" reads as an app that takes anything for
    // anything. Some answers to an open question are whole listed sentences rather
    // than the frame — `Dak sigriet!` for an age — and those are marked like any
    // other line, because that is what was measured.
    // `r.verdict` is no help here: on an open question it is "correct" whenever two
    // characters were said, so printing it would claim a judgement the app never
    // made. Either the frame was measured, or what was measured is how near they
    // came to one of the listed answers.
    const freeLabel = r.frame_scored ? 'frame right'
      : (r.score >= dialogueEngine.CORRECT ? 'answer right' : 'close to an answer');
    /* No per-sound hint here, and that is a measured decision rather than an omission.
       `nanostt.worstSound` names the weakest token and it is not accurate enough to say
       out loud: on twenty deliberate mispronunciations, where the sound that was changed is
       known, the worst-scoring token *is* that sound on 6 of the 15 where it could be named
       at all. Tightening the margin does not help — precision sits at 32-36% from a 0.5
       margin all the way to 3.0 — so there is no threshold at which this becomes advice
       rather than a guess, and being told to fix a sound you said correctly is worse than
       being told nothing. The same scores are good enough to decide *whether* the sounds
       are right, which is what the near-miss verdict above uses them for. */
    const verdictText = r.free
      ? (r.score >= 0.5
        ? `${freeLabel} · ${Math.round(r.score * 100)}%`
        : 'taken as given · not scored')
      : `${r.moved_on ? 'moving on' : r.on_lead ? 'close enough' : r.verdict}`
        + ` · ${Math.round(r.score * 100)}%`;

    const turn = {
      role: 'tutor',
      mt: r.reply_mt,
      en: r.reply_en,
      verdict: { tone, mark, text: `${verdictText} · ${ms}ms` },
      target: r.say_this_mt ? { mt: r.say_this_mt, en: r.say_this_en } : null,
    };
    drillBubble(turn.role, turn.mt, turn.en, turn);
    drill.turns.push(turn);
    keepDrill();

    if (state.settings.autoplay) await speak(r.reply_mt);

    // Tally before advancing: a first-time hit is worth more than a third attempt.
    if (r.verdict === 'correct') {
      if (r.moved_on) drill.run.movedOn += 1;
      else if (drill.peeked) drill.run.peeked += 1;
      else if (drill.attempts === 0) drill.run.first += 1;
      else drill.run.retried += 1;
      /* Never on a free node. There `matched_mt` is whichever example answer scored
         highest against a name or a town — 15% of nothing — and scheduling it filed
         a sentence the learner never said into their review deck as one they had
         produced correctly.
         And never after a peek, for the same reason one step removed: the line was
         on the screen a moment ago, so repeating it is evidence of reading rather
         than of recall, and FSRS would take it as a card learned. */
      if (r.matched_mt && !r.free && !r.moved_on && !drill.peeked) {
        drill.run.learned.push({ mt: r.matched_mt, en: r.matched_en });
        // The server used to do this; it has no database to do it in now.
        schedule.registerFromDrill([{ mt: r.matched_mt, en: r.matched_en }], state.settings)
          .catch(() => { /* bookkeeping must never break a turn */ });
      }
    }

    if (r.advance && r.next) {
      // Stay busy until the next node is actually installed. Clearing the flag in
      // `finally` reopened input 450ms early, and anything sent in that window was
      // graded against the turn that had just finished.
      holdBusy = true;
      setTimeout(() => { presentDrillNode(r.next); drill.busy = false; }, 450);
    } else if (r.finished) {
      $('drillExpect').textContent = '';
      /* And the cue with it. The composer belongs to a node, and there is no node
         any more: leaving it there ended the scene with a summary on screen and, in
         the panel below it, the answer to a question that had already been asked and
         a "Hide" button for it. Dropping `present` is what makes `hideAnswer` take
         the button away rather than re-offer it. */
      drill.present = null;
      hideAnswer();
      doneScenes.mark(drill.dialogue);
      renderScenePath();
      showRunSummary();
      // Nothing left to come back to: the scene is done and the summary offers
      // Again or the next scene. Keeping it would restore a conversation whose
      // composer has nowhere to send an answer.
      saved.clear();
      await refreshCounts();
    }
  } catch (err) {
    toast(err.message);
    /* The scene has moved under a restored conversation: the node it was sitting on
       is not there any more, so no answer will ever land. Say so once and start the
       scene, rather than leaving a chat that reads fine and cannot be continued —
       and forget it, or the next reload would restore the same dead end. */
    if (/unknown node/i.test(err.message || '')) {
      saved.clear();
      await startDrill(drill.dialogue);
    }
  } finally {
    if (!holdBusy) drill.busy = false;
  }
}

/* ── Review ────────────────────────────────────────────────────────────── */

async function loadQueue() {
  state.queue = await schedule.buildQueue(25, { settings: state.settings });
  state.qIndex = 0;
  await refreshCounts();
  showCard();
}

function showCard() {
  const empty = $('reviewEmpty');
  const card = $('reviewCard');
  if (state.qIndex >= state.queue.length) {
    state.card = null;
    card.hidden = true;
    empty.hidden = false;
    empty.querySelector('h2').textContent = state.queue.length
      ? 'Sew! Spiċċajt.' : "Xejn x'tirrepeti bħalissa";
    empty.querySelector('p').textContent = state.queue.length
      ? 'Session done — everything due has been reviewed. Come back later, or start a conversation.'
      : 'Nothing due right now. Have a conversation — new words you meet there get scheduled automatically — or pull in some new material below.';
    return;
  }

  empty.hidden = true;
  card.hidden = false;
  state.revealed = false;
  state.attempted = false;

  const c = state.queue[state.qIndex];
  state.card = c;

  $('cardMode').textContent = { produce: 'say it', recognise: 'recall', listen: 'listen' }[c.mode] || c.mode;
  $('cardTopic').textContent = c.topic || '';
  $('cardState').textContent = c.state === 'new' ? 'new' : c.state;
  $('cardProgress').textContent = `${state.qIndex + 1} / ${state.queue.length}`;

  const labels = {
    produce: 'Say this in Maltese',
    recognise: 'What does this mean?',
    listen: 'Listen, then repeat',
  };
  $('promptLabel').textContent = labels[c.mode] || '';

  // Front of the card depends on the retrieval direction being trained.
  if (c.mode === 'produce') {
    $('cardPrompt').textContent = c.en;
    $('cardSub').textContent = c.kind === 'phrase' ? 'whole phrase' : (c.pos || '');
    $('cardPlay').hidden = true;
  } else if (c.mode === 'recognise') {
    $('cardPrompt').textContent = c.mt;
    $('cardSub').textContent = '';
    $('cardPlay').hidden = false;
  } else {
    $('cardPrompt').textContent = c.mt;
    $('cardSub').textContent = c.literal ? `literally: ${c.literal}` : '';
    $('cardPlay').hidden = false;
    if (state.settings.autoplay) speak(c.mt);
  }

  $('cardAnswer').hidden = true;
  $('attemptBox').hidden = true;
  $('gradeRow').hidden = true;
  $('revealBtn').hidden = false;
  $('speakRow').hidden = c.mode === 'recognise';
  $('cardInput').value = '';

  for (const g of [1, 2, 3, 4]) $(`int${g}`).textContent = c.intervals?.[g] || '';
}

function reveal() {
  if (!state.card || state.revealed) return;
  state.revealed = true;
  const c = state.card;
  $('answerMt').textContent = c.mt;
  $('answerEn').textContent = c.en;
  $('answerNote').textContent = [c.literal ? `literally: ${c.literal}` : '', c.note || '']
    .filter(Boolean).join(' · ');
  $('answerExample').textContent = c.example_mt
    ? `${c.example_mt} — ${c.example_en || ''}` : '';
  $('cardAnswer').hidden = false;
  $('gradeRow').hidden = false;
  $('revealBtn').hidden = true;
  $('speakRow').hidden = true;
  if (state.settings.autoplay && !state.attempted) speak(c.mt);
}

async function gradeAttempt(said) {
  const c = state.card;
  if (!c || !said?.trim()) return;
  state.attempted = true;
  const a = STATIC.on
    ? mtext.assess(said, c.mt)
    : await post('/api/attempt', { said, target: c.mt });
  // The score has a phonetic floor but the diff is spelling-level, so the two can
  // disagree: `birrakisħa` for `Birra kiesħa` is a perfect answer that still diffs
  // as a substitution. Showing red words under "nothing to fix" reads as a bug, so
  // a perfect verdict shows no diff.
  const showDiff = a.verdict !== 'perfect';
  $('attemptDiff').hidden = !showDiff;
  // A chunk ending in "-" is a fused article (mill-, id-); keep it glued to the next.
  $('attemptDiff').innerHTML = a.diff.map((d, i) => {
    const prev = a.diff[i - 1];
    const glue = prev && (prev.target || prev.said).endsWith('-') ? '' : ' ';
    let html;
    if (d.op === 'equal') html = `<span class="eq">${escapeHtml(d.target)}</span>`;
    else if (d.op === 'sub') html = `<span class="bad">${escapeHtml(d.said)}</span> <span class="good">${escapeHtml(d.target)}</span>`;
    else if (d.op === 'del') html = `<span class="bad">${escapeHtml(d.said)}</span>`;
    else html = `<span class="miss">${escapeHtml(d.target)}</span>`;
    return (i ? glue : '') + html;
  }).join('');
  const pct = Math.round(a.score * 100);
  $('attemptVerdict').textContent = {
    perfect: `Perfetta! ${pct}% — nothing to fix.`,
    close: `Qrib ħafna — ${pct}%. Check the highlighted words.`,
    partial: `${pct}%. Listen again and have another go.`,
    off: `${pct}%. Let's hear the model version.`,
  }[a.verdict];
  $('attemptBox').hidden = false;
  reveal();
  if (a.score < 0.85) speak(c.mt, { rate: 0.75 });
  // Pre-select the auto-grade so a single click confirms it.
  document.querySelectorAll('.grade').forEach((b) => {
    b.style.outline = Number(b.dataset.grade) === a.grade ? `2px solid var(--sea)` : '';
  });
}

async function submitGrade(grade) {
  const c = state.card;
  if (!c) return;
  const said = $('cardInput').value.trim() || null;
  try {
    await schedule.recordReview(c.id, grade, c.mode, { said, settings: state.settings });
    await refreshCounts();
    // A lapsed card comes back later in the same session.
    if (grade === srs.AGAIN) state.queue.push({ ...c, intervals: c.intervals });
  } catch (err) {
    toast(err.message);
  }
  document.querySelectorAll('.grade').forEach((b) => { b.style.outline = ''; });
  state.qIndex += 1;
  showCard();
}

/* ── Progress ──────────────────────────────────────────────────────────── */

async function loadStats() {
  const s = await schedule.stats();
  const speakPct = s.speaking.attempts
    ? Math.round((s.speaking.correct / s.speaking.attempts) * 100) : 0;
  const cards = [
    ['Words learned', s.learned],
    ['Solid (3wk+)', s.solid],
    ['Due now', s.due],
    ['Not started', s.new],
    ['Day streak', s.streak],
    ['Spoken accuracy', `${speakPct}%`],
  ];
  $('statGrid').innerHTML = cards
    .map(([label, v]) => `<div class="stat"><b>${v}</b><span>${label}</span></div>`).join('');

  const max = Math.max(1, ...s.history.map((h) => h.n));
  $('historyChart').innerHTML = s.history
    .map((h) => `<i style="height:${Math.max(4, (h.n / max) * 100)}%" title="${h.d}: ${h.n} reviews, ${Math.round((h.retention || 0) * 100)}% recalled"></i>`)
    .join('') || '<span style="color:var(--text-dim);font-size:.85rem">No reviews yet.</span>';

  $('topicList').innerHTML = s.topics.map((t) => {
    const pct = t.total ? Math.round((t.learned / t.total) * 100) : 0;
    return `<div class="topic-row">
        <span>${escapeHtml(t.topic)}</span>
        <span class="topic-bar"><i style="width:${pct}%"></i></span>
        <span class="num">${t.learned}/${t.total}</span>
      </div>`;
  }).join('');

  $('weakList').innerHTML = s.weak.length
    ? s.weak.map((w) => `<li><span class="mt">${escapeHtml(w.mt)}</span><span class="n">${escapeHtml(w.en)} · ${w.lapses} slips</span></li>`).join('')
    : '<li class="n">Nothing sticky yet.</li>';

  $('errorList').innerHTML = s.weak.length
    ? '<li class="n">Practise the sticky words above — they carry the most value.</li>'
    : '<li class="n">No slips logged yet.</li>';
}

/* ── Vocabulary browser ────────────────────────────────────────────────────
   The deck is 470 curated items and until now none of it was visible: you met a
   word only when the scheduler chose to show it. This lets you look. */

let vocabTimer = null;

async function loadVocab() {
  const q = $('vocabSearch').value.trim();
  const topic = $('vocabTopic').value;
  const params = new URLSearchParams({ limit: '300' });
  if (q) params.set('q', q);
  if (topic) params.set('topic', topic);

  const cards = await localCards({ q, topic, limit: 300 });
  $('vocabCount').textContent = cards.length >= 300 ? '300+ shown' : `${cards.length} shown`;

  // Named `stage`, not `state`: the module-level `state` object holds the settings
  // every other render function reads, and shadowing it here is a trap.
  $('vocabList').innerHTML = cards.map((c) => {
    const stage = c.state === 'review' ? 'known' : c.state === 'new' ? 'new' : 'learning';
    return `<div class="vocab-row" data-say="${escapeHtml(c.mt)}">
        <button class="vocab-play" aria-label="Play ${escapeHtml(c.mt)}">🔊</button>
        <span class="vocab-mt">${escapeHtml(c.mt)}</span>
        <span class="vocab-en">${escapeHtml(c.en)}</span>
        <span class="vocab-tag ${stage}">${stage}</span>
      </div>`;
  }).join('') || '<p class="vocab-empty">Nothing matches.</p>';

  $('vocabList').querySelectorAll('.vocab-row').forEach((row) => {
    row.querySelector('.vocab-play').onclick = () => speak(row.dataset.say);
  });
}

/** The deck plus each card's local schedule state — the vocabulary browser's
    "known / learning / new" tag is the learner's progress, so it comes from
    IndexedDB rather than from the server. */
async function localCards({ q = '', topic = '', limit = 300 } = {}) {
  const [cards, states] = await Promise.all([store.getCards(), store.getStates()]);
  const byId = new Map(states.map((s) => [s.cardId, s]));
  const needle = q.toLowerCase();
  return cards
    .filter((c) => (!topic || c.topic === topic))
    .filter((c) => !needle
      || c.mt.toLowerCase().includes(needle) || c.en.toLowerCase().includes(needle))
    .map((c) => ({ ...c, state: byId.get(c.id)?.state || 'new' }))
    .sort((a, b) => (a.tier - b.tier) || String(a.id).localeCompare(String(b.id)))
    .slice(0, limit);
}

async function initVocab() {
  const cards = await store.getCards();
  const topics = [...new Set(cards.map((c) => c.topic).filter(Boolean))].sort();
  $('vocabTopic').innerHTML = '<option value="">All topics</option>'
    + topics.map((t) => `<option value="${t}">${t}</option>`).join('');
  $('vocabSearch').addEventListener('input', () => {
    clearTimeout(vocabTimer);
    vocabTimer = setTimeout(loadVocab, 180);
  });
  $('vocabTopic').addEventListener('change', loadVocab);
  await loadVocab();
}

/* ── Reference ─────────────────────────────────────────────────────────── */

async function loadGrammar() {
  try {
    const { markdown } = await api(STATIC.on ? 'api/grammar.json' : '/api/grammar');
    $('grammarPane').innerHTML = renderMarkdown(markdown);
  } catch {
    $('grammarPane').textContent = 'Reference unavailable.';
  }
}

/** Deliberately small markdown subset — enough for data/grammar_notes.md. */
function renderMarkdown(md) {
  const lines = md.split('\n');
  const out = [];
  let inTable = false;
  let inList = null;
  let inComment = false;

  const inline = (s) => escapeHtml(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[\s(])\*([^*]+)\*/g, '$1<em>$2</em>');

  const closeList = () => { if (inList) { out.push(`</${inList}>`); inList = null; } };
  const closeTable = () => { if (inTable) { out.push('</tbody></table>'); inTable = false; } };

  for (const raw of lines) {
    const line = raw.trimEnd();

    // Authoring notes live in HTML comments and are not for the learner.
    if (inComment) { if (line.includes('-->')) inComment = false; continue; }
    if (line.trimStart().startsWith('<!--')) {
      if (!line.includes('-->')) inComment = true;
      continue;
    }

    if (/^\|[\s:|-]+\|$/.test(line)) continue;                    // table separator
    if (line.startsWith('|')) {
      const cells = line.slice(1, -1).split('|').map((c) => c.trim());
      if (!inTable) {
        closeList();
        out.push(`<table><thead><tr>${cells.map((c) => `<th>${inline(c)}</th>`).join('')}</tr></thead><tbody>`);
        inTable = true;
      } else {
        out.push(`<tr>${cells.map((c) => `<td>${inline(c)}</td>`).join('')}</tr>`);
      }
      continue;
    }
    closeTable();

    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) { closeList(); out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); continue; }

    const ol = line.match(/^(\d+)\.\s+(.*)$/);
    if (ol) {
      if (inList !== 'ol') { closeList(); out.push('<ol>'); inList = 'ol'; }
      out.push(`<li>${inline(ol[2])}</li>`);
      continue;
    }
    const ul = line.match(/^[-*]\s+(.*)$/);
    if (ul) {
      if (inList !== 'ul') { closeList(); out.push('<ul>'); inList = 'ul'; }
      out.push(`<li>${inline(ul[1])}</li>`);
      continue;
    }

    if (!line.trim()) { closeList(); continue; }
    closeList();
    out.push(`<p>${inline(line)}</p>`);
  }
  closeList(); closeTable();
  return out.join('\n');
}

/* ── Mini-games ────────────────────────────────────────────────────────────
   Four activities the conversation and the deck between them do not cover, all of
   them recognition or assembly rather than production:

     build      an English sentence, its Maltese one word per tile, in order
     hearing    one of three near-identical words is played; pick it
     listening  ten seconds of continuous Maltese, then a question about it
     grammar    two contrasting correct sentences, then a gap to fill

   The items are derived at build time by `backend/games.py` — from the scripted
   dialogues and the deck, so their Maltese is sentences somebody already wrote — and
   `games.js` marks them. Nothing about how they are made lives here. */

const KIND_META = {
  build: { icon: '🧩', name: 'Ibni sentenza', en: 'Build a sentence',
           blurb: 'Put the Maltese in order. Some tiles do not belong.' },
  hearing: { icon: '👂', name: 'Liema smajt?', en: 'Which did you hear?',
             blurb: 'Three words that sound alike. One of them is played.' },
  listening: { icon: '🎧', name: 'Podcast qasir', en: 'Mini podcast',
               blurb: 'Ten seconds of Maltese, then a question about it.' },
  grammar: { icon: '📐', name: 'Regoli', en: 'Grammar',
             blurb: 'Two examples, then fill the gap.' },
};

const play = {
  payload: null,
  queue: [],
  at: 0,
  kind: null,
  answered: false,
  placed: [],            // build: the tiles put down, in order
  chosen: [],            // listening/words: the words ticked
  right: 0,
};

async function loadGames() {
  if (play.payload) return play.payload;
  play.payload = STATIC.on
    ? await api('api/games.json')
    : await api('/api/games');
  return play.payload;
}

async function showGameMenu() {
  const payload = await loadGames();
  $('gamePlay').hidden = true;
  $('gameMenu').hidden = false;

  const counts = games.KINDS ?? Object.keys(KIND_META);
  $('gameGrid').innerHTML = Object.keys(KIND_META).map((kind) => {
    const meta = KIND_META[kind];
    const n = (payload[kind] || []).length;
    return `<button class="game-card" data-kind="${kind}" ${n ? '' : 'disabled'}>
        <span class="game-icon" aria-hidden="true">${meta.icon}</span>
        <span class="game-body-text">
          <span class="mt">${escapeHtml(meta.name)}</span>
          <span class="en">${escapeHtml(meta.en)}</span>
          <span class="blurb">${escapeHtml(meta.blurb)}</span>
        </span>
        <span class="game-count">${n}</span>
      </button>`;
  }).join('') + `<button class="game-card is-mixed" data-kind="">
      <span class="game-icon" aria-hidden="true">🎲</span>
      <span class="game-body-text">
        <span class="mt">Taħlita</span>
        <span class="en">A bit of everything</span>
        <span class="blurb">All four, interleaved — which is how they work best.</span>
      </span>
    </button>`;

  for (const card of $('gameGrid').querySelectorAll('.game-card')) {
    card.onclick = () => startGames(card.dataset.kind || null);
  }
  const total = Object.keys(KIND_META).reduce((n, k) => n + (payload[k] || []).length, 0);
  $('gamesProgress').textContent = `${total} exercises · ${payload.session} to a round`;
}

async function startGames(kind) {
  const payload = await loadGames();
  play.kind = kind;
  play.queue = games.session(payload, {
    count: payload.session,
    kinds: kind ? [kind] : null,
  });
  play.at = 0;
  play.right = 0;
  if (!play.queue.length) { toast('Nothing to play here yet'); return; }

  $('gameMenu').hidden = true;
  $('gamePlay').hidden = false;
  $('gameName').textContent = kind ? KIND_META[kind].en : 'A bit of everything';
  showGameItem();
}

function showGameItem() {
  const item = play.queue[play.at];
  if (!item) { showGameSummary(); return; }

  play.answered = false;
  play.placed = [];
  play.chosen = [];
  $('gameVerdict').textContent = '';
  $('gameVerdict').className = 'game-verdict';
  $('gameNext').hidden = true;
  $('gameStep').hidden = false;
  $('gameStep').textContent = `${play.at + 1}/${play.queue.length}`;

  const draw = {
    build: drawBuild, hearing: drawHearing,
    listening: drawListening, grammar: drawGrammar,
  }[item.kind];
  $('gameBody').innerHTML = '';
  draw(item);
}

/* Each `draw*` writes the body and wires its own controls. They all end the same way:
   `settle()` marks the answer, says what was wrong where that is useful, and files
   anything the answer is evidence of. */

function settle(item, answer) {
  if (play.answered) return;
  play.answered = true;
  const correct = games.mark(item, answer);
  if (correct) play.right += 1;

  const verdict = $('gameVerdict');
  verdict.className = `game-verdict ${correct ? 'ok' : 'bad'}`;
  let say = correct ? 'Sewwa!' : (games.critique(item, answer) || 'Not quite.');
  if (!correct && item.kind === 'grammar') say = item.why || 'Not quite.';
  /* The explanations quote Maltese in backticks, the way the rest of this repo's prose
     does. Rendered rather than printed: a learner reading "`dar` is feminine" is
     reading punctuation, and the quoted part is the one word they need to look at. */
  verdict.innerHTML = escapeHtml(say)
    .replace(/`([^`]+)`/g, (_, mt) => `<b class="mt-inline">${mt}</b>`);
  $('gameBody').classList.add('is-answered');
  $('gameNext').hidden = false;
  $('gameNext').textContent = play.at + 1 >= play.queue.length ? 'Finish' : 'Next';

  // Only where the answer is evidence of producing something — see games.js.
  const learned = games.earned(item, correct);
  if (learned.length) {
    schedule.registerFromDrill(learned, state.settings, 'games')
      .catch(() => { /* bookkeeping must never break a turn */ });
  }
  if (correct) speak(games.audioFor(item));
}

function playButton(item, label = '🔊 Play') {
  const line = games.audioFor(item);
  if (!line) return '';
  return `<div class="game-audio">
      <button class="cue-btn" data-say>${label}</button>
      <button class="cue-btn" data-slow>🐢 Slow</button>
    </div>`;
}

function wirePlay(item) {
  const line = games.audioFor(item);
  const say = $('gameBody').querySelector('[data-say]');
  const slow = $('gameBody').querySelector('[data-slow]');
  if (say) say.onclick = () => speak(line);
  if (slow) slow.onclick = () => speak(line, { rate: 0.7 });
}

/* 1. Build the sentence ─────────────────────────────────────────────────── */

function drawBuild(item) {
  $('gameBody').innerHTML = `
    <p class="game-ask">${escapeHtml(item.prompt_en)}</p>
    <div class="tile-answer" id="tileAnswer" aria-label="Your sentence"></div>
    <div class="tile-pool" id="tilePool"></div>
    <div class="game-actions">
      <button class="ghost-btn" data-clear>Clear</button>
      <button class="primary-btn" data-check>Check</button>
    </div>`;

  const pool = $('gameBody').querySelector('#tilePool');
  const answer = $('gameBody').querySelector('#tileAnswer');

  const render = () => {
    answer.innerHTML = play.placed.map((w, i) =>
      `<button class="tile" data-take="${i}">${escapeHtml(w)}</button>`).join('');
    // A tile that has been placed leaves a gap rather than vanishing, so the pool does
    // not reflow under the finger mid-sentence.
    pool.innerHTML = item.tiles.map((w, i) =>
      `<button class="tile ${play.used?.has(i) ? 'is-spent' : ''}" data-put="${i}"
        ${play.used?.has(i) ? 'disabled' : ''}>${escapeHtml(w)}</button>`).join('');

    for (const b of pool.querySelectorAll('[data-put]')) {
      b.onclick = () => {
        const i = Number(b.dataset.put);
        play.used.add(i);
        play.placed.push(item.tiles[i]);
        render();
      };
    }
    for (const b of answer.querySelectorAll('[data-take]')) {
      b.onclick = () => {
        const at = Number(b.dataset.take);
        const word = play.placed[at];
        play.placed.splice(at, 1);
        // Put back the *first* spent tile with that word, which is the one the learner
        // will expect to light up again.
        for (const i of [...play.used]) {
          if (item.tiles[i] === word) { play.used.delete(i); break; }
        }
        render();
      };
    }
  };

  play.used = new Set();
  render();
  $('gameBody').querySelector('[data-clear]').onclick = () => {
    play.placed = []; play.used = new Set(); render();
  };
  $('gameBody').querySelector('[data-check]').onclick = () => {
    settle(item, play.placed);
    // Show the sentence as it should have read, which is the correction.
    const shown = document.createElement('p');
    shown.className = 'game-shown';
    shown.textContent = item.mt;
    $('gameBody').append(shown);
  };
}

/* 2. Which did you hear ─────────────────────────────────────────────────── */

function drawHearing(item) {
  $('gameBody').innerHTML = `
    <p class="game-ask">Which word was that?</p>
    ${playButton(item, '🔊 Play it again')}
    <div class="game-options">
      ${item.options.map((o, i) => `<button class="game-option" data-pick="${i}">
          <b>${escapeHtml(o.mt)}</b><em>${escapeHtml(o.en)}</em>
        </button>`).join('')}
    </div>`;
  wirePlay(item);
  wireOptions(item);
  speak(games.audioFor(item));
}

/* 3. Mini podcast ───────────────────────────────────────────────────────── */

function drawListening(item) {
  if (item.ask === 'which') {
    $('gameBody').innerHTML = `
      <p class="game-ask">${escapeHtml(item.question_en)}</p>
      <p class="game-source">${escapeHtml(item.name_en)}</p>
      ${playButton(item, '🔊 Play the clip')}
      <div class="game-options">
        ${item.options.map((o, i) => `<button class="game-option wide" data-pick="${i}">
            <b>${escapeHtml(o.en)}</b>
          </button>`).join('')}
      </div>`;
    wirePlay(item);
    wireOptions(item);
    return;
  }

  $('gameBody').innerHTML = `
    <p class="game-ask">${escapeHtml(item.question_en)}</p>
    <p class="game-source">${escapeHtml(item.name_en)}</p>
    ${playButton(item, '🔊 Play the clip')}
    <div class="word-pool" id="wordPool">
      ${item.pool.map((w) => `<button class="tile" data-word="${escapeHtml(w)}">${escapeHtml(w)}</button>`).join('')}
    </div>
    <div class="game-actions">
      <button class="primary-btn" data-check disabled>Check</button>
    </div>`;
  wirePlay(item);

  const check = $('gameBody').querySelector('[data-check]');
  for (const b of $('gameBody').querySelectorAll('[data-word]')) {
    b.onclick = () => {
      const word = b.dataset.word;
      const at = play.chosen.indexOf(word);
      if (at >= 0) play.chosen.splice(at, 1);
      else if (play.chosen.length < 3) play.chosen.push(word);
      b.classList.toggle('is-picked', play.chosen.includes(word));
      check.disabled = play.chosen.length !== 3;
    };
  }
  check.onclick = () => {
    settle(item, play.chosen);
    for (const b of $('gameBody').querySelectorAll('[data-word]')) {
      if (item.answer.includes(b.dataset.word)) b.classList.add('is-right');
    }
  };
}

/* 4. Grammar ────────────────────────────────────────────────────────────── */

function drawGrammar(item) {
  $('gameBody').innerHTML = `
    <p class="game-rule">${escapeHtml(item.rule)}</p>
    <div class="grammar-shown">
      ${item.show.map((ex) => `<p class="grammar-ex">
          <b>${escapeHtml(ex.mt)}</b><em>${escapeHtml(ex.en)}</em>
        </p>`).join('')}
    </div>
    <p class="game-ask gap">${escapeHtml(item.ask_mt).replace('___',
      '<span class="gap-slot" id="gapSlot">?</span>')}</p>
    <p class="game-source">${escapeHtml(item.ask_en)}</p>
    <div class="game-options">
      ${item.options.map((o, i) => `<button class="game-option" data-pick="${i}">
          <b>${escapeHtml(o)}</b>
        </button>`).join('')}
    </div>`;
  wireOptions(item, (i) => {
    const slot = $('gameBody').querySelector('#gapSlot');
    if (slot) slot.textContent = item.options[i];
  });
}

/* Shared: a row of options, one press, then the marking is shown on the buttons. */

function wireOptions(item, onPick) {
  for (const b of $('gameBody').querySelectorAll('[data-pick]')) {
    b.onclick = () => {
      if (play.answered) return;
      const picked = Number(b.dataset.pick);
      onPick?.(picked);
      settle(item, picked);
      for (const other of $('gameBody').querySelectorAll('[data-pick]')) {
        const at = Number(other.dataset.pick);
        if (at === item.answer) other.classList.add('is-right');
        else if (at === picked) other.classList.add('is-wrong');
      }
    };
  }
}

function showGameSummary() {
  const total = play.queue.length;
  $('gameStep').hidden = true;
  $('gameVerdict').textContent = '';
  $('gameNext').hidden = true;
  $('gameBody').classList.remove('is-answered');
  $('gameBody').innerHTML = `
    <div class="game-summary">
      <p class="mt">Spiċċajna!</p>
      <p class="score"><b>${play.right}</b> of ${total}</p>
      <div class="summary-actions">
        <button class="ghost-btn" data-again>Again</button>
        <button class="primary-btn" data-menu>Other games</button>
      </div>
    </div>`;
  $('gameBody').querySelector('[data-again]').onclick = () => startGames(play.kind);
  $('gameBody').querySelector('[data-menu]').onclick = () => showGameMenu();
  refreshCounts().catch(() => {});
}

$('gameBack').addEventListener('click', () => showGameMenu().catch((e) => toast(e.message)));
$('gameNext').addEventListener('click', () => { play.at += 1; showGameItem(); });

/* ── Wiring ────────────────────────────────────────────────────────────── */

function switchView(name) {
  for (const t of document.querySelectorAll('.tab, .sheet-item')) {
    t.classList.toggle('is-active', t.dataset.view === name);
  }
  document.querySelectorAll('.view').forEach((v) => v.classList.toggle('is-active', v.id === `view-${name}`));
  if (name === 'review' && !state.card) loadQueue().catch((e) => toast(e.message));
  if (name === 'progress') loadStats().catch((e) => toast(e.message));
  if (name === 'drill' && !drill.dialogue) loadDrills().catch((e) => toast(e.message));
  // The path is built with the dialogue list, which the Talk view is what loads.
  if (name === 'scenes' && !drill.dialogues.length) {
    loadDrills().catch((e) => toast(e.message));
  }
  if (name === 'reference' && !$('vocabList').children.length) {
    initVocab().catch((e) => toast(e.message));
  }
  if (name === 'games' && !$('gameGrid').children.length) {
    showGameMenu().catch((e) => toast(e.message));
  }
  // The chat has no height to scroll while it is display:none, so it is put at the
  // bottom once it has one again.
  if (name === 'drill') {
    const chat = $('drillChat');
    chat.scrollTop = chat.scrollHeight;
  }
}

document.querySelectorAll('.tab').forEach((t) => {
  t.addEventListener('click', () => switchView(t.dataset.view));
});

/* ── Navigation sheet ──────────────────────────────────────────────────────
   Everything the top bar used to hold on a phone, plus the two buttons that used
   to sit above the conversation. Below 640px this is the only way to the other
   views, so it has to be complete. */

function openSheet() {
  $('sheetConversation').hidden = !$('view-drill').classList.contains('is-active');
  $('navBtn').setAttribute('aria-expanded', 'true');
  $('navSheet').showModal();
}

function closeSheet() {
  $('navSheet').close();
}

$('navSheet').addEventListener('close', () => {
  $('navBtn').setAttribute('aria-expanded', 'false');
});

// Tapping the backdrop. A sheet you can only leave by finding a button is a trap on
// a phone, and `dialog` gives the click to the element itself when it is the scrim.
$('navSheet').addEventListener('click', (e) => {
  if (e.target === $('navSheet')) closeSheet();
});

$('navBtn').addEventListener('click', openSheet);
$('sheetClose').addEventListener('click', closeSheet);

document.querySelectorAll('.sheet-item').forEach((b) => {
  b.addEventListener('click', () => { closeSheet(); switchView(b.dataset.view); });
});

$('sheetSettings').addEventListener('click', () => {
  closeSheet();
  openSettings().catch((e) => toast(e.message));
});
$('sheetRestart').addEventListener('click', () => {
  closeSheet();
  startDrill(drill.dialogue);
});
$('sheetNext').addEventListener('click', () => { closeSheet(); goToNextScene(); });
/* The conversation's own header. `‹ scene name` is the way back to the path — the
   same gesture as any list-then-detail screen — and `⋯` is the actions. */
$('drillScene').addEventListener('click', () => switchView('scenes'));
$('drillMore').addEventListener('click', openSheet);

$('drillSend').addEventListener('click', () => answerDrill($('drillInput').value));
$('drillInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') answerDrill($('drillInput').value);
});

$('drillReveal').addEventListener('click', toggleAnswer);
$('drillAnswerPlay').addEventListener('click', () => speak(drill.present?.answer?.mt));
$('drillAnswerSlow').addEventListener('click',
  () => speak(drill.present?.answer?.mt, { rate: 0.7 }));

bindMic($('drillMic'), {
  onStatus: (s) => { $('drillStatus').textContent = s || 'Hold the mic and answer'; },
  onResult: async (res) => {
    if (!res.text) { toast('Nothing heard — try again'); return; }
    await answerDrill(res.text, {
      nearSound: res.nearSound, worst: res.worstSound, gop: res.gop,
    });
  },
});

$('cardPlay').addEventListener('click', () => state.card && speak(state.card.mt));
$('revealBtn').addEventListener('click', reveal);
$('skipBtn').addEventListener('click', reveal);
$('startNewBtn').addEventListener('click', () => loadQueue().catch((e) => toast(e.message)));

$('cardInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') gradeAttempt($('cardInput').value);
});

bindMic($('cardMic'), {
  target: () => state.card?.mt,
  onStatus: () => {},
  onResult: async (res) => {
    $('cardInput').value = res.text;
    await gradeAttempt(res.text);
  },
});

document.querySelectorAll('.grade').forEach((b) => {
  b.addEventListener('click', () => submitGrade(Number(b.dataset.grade)));
});

/* Settings. Reached from the gear on a wide screen and from the sheet on a phone,
   where the gear is one of the things that had to leave the top bar. */
async function openSettings() {
  $('settingsDialog').showModal();
  const c = await schedule.counts();
  $('progressSummary').textContent =
    `${c.learned} learned · ${c.total - c.new} started · ${c.today} reviews in the last day`;
}

$('settingsBtn').addEventListener('click', () => { openSettings().catch(() => {}); });

/* Progress lives on this device, so the learner needs a way to carry it. */
$('exportProgress').addEventListener('click', async () => {
  try {
    const dump = await store.exportAll();
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(dump)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `speak-maltese-progress-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (err) {
    toast(`Export failed: ${err.message}`);
  }
});

$('importProgress').addEventListener('click', () => $('importFile').click());

$('importFile').addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  e.target.value = '';                       // so re-picking the same file fires again
  if (!confirm('Importing replaces the progress on this device. Continue?')) return;
  try {
    await store.importAll(JSON.parse(await file.text()));
    toast('Progress imported.');
    await refreshCounts();
    await loadQueue();
  } catch (err) {
    toast(`Import failed: ${err.message}`, 6000);
  }
});

$('resetProgress').addEventListener('click', async () => {
  if (!confirm('Delete all your progress on this device? This cannot be undone.')) return;
  await store.reset();
  const { cards } = await api(STATIC.on ? 'api/deck.json' : '/api/deck');
  await store.seedDeck(cards);
  toast('Starting over.');
  await refreshCounts();
  await loadQueue();
});
$('rateRange').addEventListener('input', (e) => {
  state.settings.rate = Number(e.target.value);
  $('rateLabel').textContent = `${state.settings.rate.toFixed(2)}×`;
});
$('localStt').addEventListener('change', (e) => setLocalStt(e.target.checked));

$('settingsDialog').addEventListener('close', () => {
  state.settings.voice = $('voiceSelect').value;
  state.settings.show_english = $('showEnglish').checked;
  state.settings.autoplay = $('autoplay').checked;
  persistSettings();
});

/* Keyboard shortcuts */
document.addEventListener('keydown', (e) => {
  const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName);
  const reviewing = $('view-review').classList.contains('is-active');

  if (reviewing && !typing) {
    if (e.code === 'Space') { e.preventDefault(); state.revealed ? submitGrade(3) : reveal(); return; }
    if (['1', '2', '3', '4'].includes(e.key) && state.revealed) { submitGrade(Number(e.key)); return; }
    if (e.key === 'r' && state.card) { speak(state.card.mt); return; }
  }
});

/* Acquire the mic on the first gesture, so the first recording does not pay the
   getUserMedia cost mid-utterance — and before any gesture at all where the browser
   will confirm the microphone is already ours to open.

   `capture: true` matters. These were bubble-phase, so when the first gesture *was*
   the mic button they ran after the button's own handler had already started
   recording — which is the whole reason `begin()`'s `if (probing) await probing` could
   never fire on the press that needed it. In the capture phase they run first, which
   is what makes that await mean something. */
window.addEventListener('pointerdown', prewarmMic, { once: true, capture: true });
window.addEventListener('keydown', prewarmMic, { once: true, capture: true });
prewarmIfAlreadyAllowed();

// Offline support. Registered last so a failure here can never stop the app
// booting — it is an enhancement, not a dependency.
if ('serviceWorker' in navigator) {
  // Was this page already being served by a worker? If not, the first activation
  // is this app starting up, not a new build arriving.
  const renewal = !!navigator.serviceWorker.controller;
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('sw.js').catch(() => {});
  });

  /* A new build has taken over the page. Until it reloads, the code in memory is
     the old build talking to the new one's files — which is how a prompt came to
     be missing the Maltese frame that its own freshly-fetched data was asking for.
     So reload, once, and not mid-turn: an answer being typed or graded is the one
     moment when losing the page costs the learner something. */
  let reloading = false;
  navigator.serviceWorker.addEventListener('controllerchange', () => {
    if (!renewal || reloading) return;
    reloading = true;
    const whenIdle = () => {
      if (drill.busy || $('drillInput').value.trim()
        || document.querySelector('.is-recording')) {
        setTimeout(whenIdle, 1500);
        return;
      }
      location.reload();
    };
    whenIdle();
  });
}

boot();

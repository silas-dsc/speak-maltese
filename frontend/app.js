/* Nitkellmu — Speak Maltese
   Single-page client. No build step, no dependencies.

   The learner's progress lives here, in IndexedDB, not on the server: see
   store.js for the database and schedule.js for the FSRS scheduling that used to
   run in Python. The server is stateless — decks, dialogue, recognition, speech. */

import * as dialogueEngine from './dialogue.js';
import * as localstt from './localstt.js';
import * as splash from './splash.js';
import * as store from './store.js';
import * as schedule from './schedule.js';
import * as srs from './srs.js';
import * as mtext from './text.js';
import * as session from './session.js';

const $ = (id) => document.getElementById(id);

const state = {
  caps: null,
  settings: store.loadSettings(),
  queue: [],
  qIndex: 0,
  card: null,
  revealed: false,
  attempted: false,
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

const STATIC = { on: false, audio: null, modelsBase: '/models/' };

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
  if (currentAudio) { currentAudio.pause(); currentAudio = null; }
  const audio = new Audio(url);
  if (STATIC.on && rate && rate !== state.settings.rate) {
    audio.playbackRate = Math.max(0.5, Math.min(1.5, rate / state.settings.rate));
  }
  currentAudio = audio;
  return audio.play().catch((err) => {
    // Autoplay policies block the first sound until the user interacts.
    if (err.name !== 'NotAllowedError') toast('Audio unavailable — check TTS setup');
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

async function ensureStream() {
  if (sharedStream?.active) return sharedStream;
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
  return sharedStream;
}

/** Warm the mic on the first interaction so the first recording isn't the slow one. */
function prewarmMic() {
  ensureStream().catch(() => { /* permission comes later, on first real use */ });
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
   checked before use and re-acquired if they are not ready. */

const MIN_MS = 250;
const MIN_BYTES = 600;

class Recorder {
  constructor() { this.chunks = []; this.rec = null; }

  async start() {
    let stream = await ensureStream();
    if (!stream.getAudioTracks().some((t) => t.readyState === 'live' && !t.muted)) {
      sharedStream = null;                 // stale: drop it and ask again
      stream = await ensureStream();
    }
    const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4']
      .find((m) => MediaRecorder.isTypeSupported(m)) || '';
    this.chunks = [];
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

    const blob = new Blob(this.chunks, { type: rec.mimeType || 'audio/webm' });
    if (ms < MIN_MS) return { reason: `only ${(ms / 1000).toFixed(1)}s of audio` };
    if (blob.size < MIN_BYTES) {
      return { reason: `${ms}ms recorded but only ${blob.size} bytes captured `
        + `(${this.chunks.length} chunk${this.chunks.length === 1 ? '' : 's'}, `
        + `${rec.mimeType || 'no mime'})` };
    }
    return { blob };
  }
}

async function transcribe(blob, target) {
  if (state.settings.local_stt && localstt.isReady()) {
    try {
      const r = await localstt.transcribe(blob);
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
  const fd = new FormData();
  fd.append('audio', blob, 'speech.webm');
  if (target) fd.append('target', target);
  return api('/api/stt', { method: 'POST', body: fd });
}

/** Turn local recognition on or off. Loading is the expensive half, so progress
    is reported rather than left as a frozen toggle. */
async function setLocalStt(on) {
  const box = $('localStt');
  const note = $('localSttNote');
  state.settings.local_stt = on;
  persistSettings();
  if (!on) {
    localstt.unload();
    note.textContent = 'Speech is sent to the server for recognition.';
    return;
  }
  if (!localstt.supported()) {
    state.settings.local_stt = false;
    persistSettings();
    box.checked = false;
    note.textContent = 'This browser has no WebGPU, so on-device recognition '
      + 'would be far slower than the server. Left off.';
    return;
  }
  box.disabled = true;
  try {
    await localstt.load({
      base: STATIC.modelsBase,
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
    note.textContent = `Could not load it: ${err.message}. Still using the server.`;
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

  async function begin() {
    // `active` cannot be the only guard: it is set after the await below, so two
    // quick taps both got past it and started a second MediaRecorder over the top
    // of the first, which then ran on unreferenced and unstoppable.
    if (active || starting || button.classList.contains('is-busy')) return;
    starting = true;
    try {
      await recorder.start();
      active = true;
      button.classList.add('is-recording');
      onStatus?.('Listening… (release to send)');
    } catch (err) {
      toast(err.message);
    } finally {
      starting = false;
    }
  }

  async function end() {
    if (!active) return;
    active = false;
    button.classList.remove('is-recording');
    setBusy(true);
    onStatus?.('Transcribing…');
    try {
      const { blob, reason } = await recorder.stop();
      // Say what actually went wrong. "Too short" was a guess the code made about
      // the learner, and twice it was wrong about itself instead.
      if (!blob) { onStatus?.(`Nothing recorded — ${reason}`); return; }
      const result = await transcribe(blob, typeof target === 'function' ? target() : target);
      onStatus?.('');
      await onResult(result);
    } catch (err) {
      onStatus?.('');
      toast(`Could not transcribe: ${err.message}`);
    } finally {
      setBusy(false);
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
        STATIC.modelsBase = boot.models_base || STATIC.modelsBase;
        dialogueEngine.load(dialogues);
      },
      onModel: async (onProgress) => {
        // Only worth waiting on where it is the only recogniser there is.
        if (!store.loadSettings().local_stt || !localstt.supported()) return false;
        try {
          await localstt.load({ base: STATIC.modelsBase, onProgress });
          return true;
        } catch {
          return false;      // typing still works; the toggle explains why
        }
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
  await refreshCounts();

  await loadDrills();
  loadGrammar();

  // Opted in on a previous visit: warm it in the background so the first thing
  // they say is not the slow one. The model is in the HTTP cache by now.
  // Static builds already loaded it during startup. A server build has a working
  // recogniser of its own, so this warms in the background instead of blocking.
  if (state.settings.local_stt && localstt.supported() && !localstt.isReady()) {
    localstt.load({ base: STATIC.modelsBase })
      .catch(() => { /* the server path still works */ });
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
  $('localStt').checked = !!state.settings.local_stt;
  $('localStt').disabled = !localstt.supported();
  $('localSttNote').textContent = localstt.supported()
    ? (state.settings.local_stt
      ? 'Loads when you next speak.'
      : 'Speech is sent to the server for recognition.')
    : 'Needs WebGPU — not available in this browser.';
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
  const badge = $('dueBadge');
  const n = counts.due + Math.min(counts.new, 12);
  badge.textContent = n;
  badge.hidden = n === 0;
}

/* ── Drill: scripted conversation ──────────────────────────────────────────
   No model in the loop. The reply is chosen by phonetic match on the server in
   under a millisecond, and its audio is already cached, so the only wait is
   speech recognition. */

const drill = {
  dialogue: null, node: null, busy: false, attempts: 0, dialogues: [],
  // Per-run tally, so finishing a scene ends with something to look at rather
  // than just stopping.
  run: { first: 0, retried: 0, movedOn: 0, learned: [], startedAt: 0 },
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
}

/** Scenes you have finished, so the picker can show progress across sessions. */
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
    ? { dialogues: dialogueEngine.all().map(({ id, name, name_en, level }) =>
        ({ id, name, name_en, level })) }
    : await api('/api/drill/dialogues');
  drill.dialogues = dialogues;
  const sel = $('drillSelect');
  renderDrillOptions();
  sel.onchange = () => startDrill(sel.value);
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
  drill.run = s.run || { first: 0, retried: 0, movedOn: 0, learned: [], startedAt: Date.now() };
  // The clock counts time spent in the scene, not time the tab spent closed.
  drill.run.startedAt = Date.now() - (s.run?.elapsedMs || 0);
  $('drillSelect').value = s.dialogue;
  showSceneImage(s.dialogue);
  $('drillChat').innerHTML = '';
  for (const t of s.turns) drillBubble(t.role, t.mt, t.en, t);
  // Silently: the tutor is not going to read the whole conversation back, and the
  // node is where the learner left off rather than a line just spoken.
  installDrillNode(present);
  // After installing, which starts a node's attempts at nought: two tries already
  // spent are two the learner does not have to spend again to be moved on.
  drill.attempts = s.attempts || 0;
  return true;
}

/** Groups the picker by level and ticks what you've finished.

    Thirty-odd scenes in a flat list tells a learner nothing about where to start,
    so they are grouped by level in the order the vocabulary builds, and the first
    unfinished scene is offered as the obvious next thing. */
function renderDrillOptions() {
  const done = doneScenes.all();
  const sel = $('drillSelect');
  const current = sel.value;
  const levels = [...new Set(drill.dialogues.map((d) => d.level))].sort();

  sel.innerHTML = levels.map((lvl) => {
    const opts = drill.dialogues.filter((d) => d.level === lvl)
      .map((d) => `<option value="${d.id}">${done[d.id] ? '✓ ' : ''}${d.name} — ${d.name_en}</option>`)
      .join('');
    return `<optgroup label="${lvl}">${opts}</optgroup>`;
  }).join('');
  if (current) sel.value = current;

  const nextUp = drill.dialogues.find((d) => !done[d.id]);
  const finished = drill.dialogues.filter((d) => done[d.id]).length;
  $('drillProgress').textContent = nextUp
    ? `${finished} of ${drill.dialogues.length} scenes done · next up: ${nextUp.name_en}`
    : `All ${drill.dialogues.length} scenes done — go round again, they get easier.`;
}

/** Jump to the first scene not yet completed. */
function goToNextScene() {
  const done = doneScenes.all();
  const next = drill.dialogues.find((d) => !done[d.id]) || drill.dialogues[0];
  $('drillSelect').value = next.id;
  startDrill(next.id);
}

/** Start a scene from its first line, throwing away whatever was on screen. This
    is what "Start over" does, and the only way a saved conversation is discarded
    on purpose. */
async function startDrill(id) {
  if (!id) return;
  drill.dialogue = id;
  drill.run = { first: 0, retried: 0, movedOn: 0, learned: [], startedAt: Date.now() };
  drill.turns = [];
  drill.present = null;
  saved.clear();
  $('drillChat').innerHTML = '';
  showSceneImage(id);
  const node = STATIC.on
    ? dialogueEngine.start(id)
    : await post('/api/drill/start', { dialogue: id });
  presentDrillNode(node);
}

/* Scene art is decoration: if an image is missing the header simply stays hidden,
   because a broken-image icon above a conversation is worse than no picture. */
function showSceneImage(id) {
  const hero = $('sceneHero');
  const img = $('sceneImg');
  const meta = drill.dialogues?.find((d) => d.id === id);
  img.onerror = () => { hero.hidden = true; };
  img.onload = () => { hero.hidden = false; };
  img.alt = meta ? `${meta.name_en}` : '';
  $('sceneCaption').textContent = meta ? `${meta.name} · ${meta.name_en}` : '';
  hero.hidden = true;
  img.src = `img/scene-${id}.webp`;
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
  const frames = (node.frames || []).map((f) => `<b>${escapeHtml(f)}</b>`).join(' / ');
  $('drillExpect').innerHTML = node.expect_en
    ? `→ ${escapeHtml(node.expect_en)}${frames ? ` · ${frames}` : ''}`
    : '';
}

/** Install a node and say its line — a new turn, as opposed to a restored one. */
function presentDrillNode(node) {
  installDrillNode(node);
  drillBubble('tutor', node.say_mt, node.say_en);
  drill.turns.push({ role: 'tutor', mt: node.say_mt, en: node.say_en });
  keepDrill();
  if (state.settings.autoplay) speak(node.say_mt);
}

function showRunSummary() {
  const { first, retried, movedOn, learned, startedAt } = drill.run;
  const total = first + retried + movedOn;
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
function drillBubble(role, mt, en, { extraClass = '', verdict = null, target = null } = {}) {
  const el = document.createElement('div');
  el.className = `turn ${role} ${extraClass}`;
  el.innerHTML = `
    <div class="bubble">
      ${verdict ? `<p class="drill-verdict ${verdict.tone}">${verdict.mark} ${escapeHtml(verdict.text)}</p>` : ''}
      <p class="mt">${escapeHtml(mt || '')}</p>
      ${en ? `<p class="en" ${state.settings.show_english ? '' : 'hidden'}>${escapeHtml(en)}</p>` : ''}
      ${target ? `<p class="drill-target">${escapeHtml(target.mt)}
           <em>${escapeHtml(target.en || '')}</em></p>` : ''}
      ${role === 'tutor' && mt ? `<div class="bubble-tools">
          <button class="tool" data-play>🔊 Play</button>
          <button class="tool" data-slow>🐢 Slow</button>
        </div>` : ''}
    </div>`;
  if (role === 'tutor' && mt) {
    el.querySelector('[data-play]').onclick = () => speak(mt);
    el.querySelector('[data-slow]').onclick = () => speak(mt, { rate: 0.7 });
  }
  $('drillChat').append(el);
  $('drillChat').scrollTop = $('drillChat').scrollHeight;
  return el;
}

async function answerDrill(said) {
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
    if (!r.advance) drill.attempts += 1;
    const ms = Math.round(performance.now() - t0);

    const tone = r.moved_on ? 'near' : { correct: 'ok', close: 'near', wrong: 'bad' }[r.verdict];
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
    const verdictText = r.free
      ? (r.score >= 0.5
        ? `${freeLabel} · ${Math.round(r.score * 100)}%`
        : 'taken as given · not scored')
      : `${r.moved_on ? 'moving on' : r.verdict} · ${Math.round(r.score * 100)}%`;

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
      else if (drill.attempts === 0) drill.run.first += 1;
      else drill.run.retried += 1;
      // Never on a free node. There `matched_mt` is whichever example answer
      // scored highest against a name or a town — 15% of nothing — and scheduling
      // it filed a sentence the learner never said into their review deck as one
      // they had produced correctly.
      if (r.matched_mt && !r.free && !r.moved_on) {
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
      doneScenes.mark(drill.dialogue);
      renderDrillOptions();
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

/* ── Wiring ────────────────────────────────────────────────────────────── */

function switchView(name) {
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('is-active', t.dataset.view === name));
  document.querySelectorAll('.view').forEach((v) => v.classList.toggle('is-active', v.id === `view-${name}`));
  if (name === 'review' && !state.card) loadQueue().catch((e) => toast(e.message));
  if (name === 'progress') loadStats().catch((e) => toast(e.message));
  if (name === 'drill' && !drill.dialogue) loadDrills().catch((e) => toast(e.message));
  if (name === 'reference' && !$('vocabList').children.length) {
    initVocab().catch((e) => toast(e.message));
  }
}

document.querySelectorAll('.tab').forEach((t) => {
  t.addEventListener('click', () => switchView(t.dataset.view));
});


$('drillSend').addEventListener('click', () => answerDrill($('drillInput').value));
$('drillInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') answerDrill($('drillInput').value);
});
$('drillRestart').addEventListener('click', () => startDrill(drill.dialogue));
$('drillNext').addEventListener('click', goToNextScene);

bindMic($('drillMic'), {
  onStatus: (s) => { $('drillStatus').textContent = s || 'Hold the mic and answer'; },
  onResult: async (res) => {
    if (!res.text) { toast('Nothing heard — try again'); return; }
    await answerDrill(res.text);
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

/* Settings */
$('settingsBtn').addEventListener('click', async () => {
  $('settingsDialog').showModal();
  const c = await schedule.counts();
  $('progressSummary').textContent =
    `${c.learned} learned · ${c.total - c.new} started · ${c.today} reviews in the last day`;
});

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

// Acquire the mic on the first gesture, so the first recording does not pay the
// getUserMedia cost mid-utterance.
window.addEventListener('pointerdown', prewarmMic, { once: true });
window.addEventListener('keydown', prewarmMic, { once: true });

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

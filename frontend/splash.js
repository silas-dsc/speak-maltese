/* The first thirty seconds.

   On free hosting the container is cold: the process starts fast, but the Maltese
   wav2vec2 model is ~1.2GB of weights and loading it takes tens of seconds. Until
   then the app looks finished and behaves broken — you hold the mic, speak, and
   nothing comes back. A blank wait with no explanation reads as a bug, and the
   learner leaves before the app has started.

   So the door is held shut and the wait is shown. Three real pieces of work, in
   the order they can be done:

     1. reach the server at all (also the thing that wakes a sleeping Space)
     2. fetch and seed the deck into IndexedDB — genuinely a few hundred rows
     3. wait for the recogniser to report itself loaded

   The bar under them is honest about 1 and 2, which finish quickly and measurably.
   Step 3 is the long one and the server cannot say how far through it is, so that
   segment creeps towards — but never reaches — the end on a decaying curve, and
   snaps to full the moment /api/health says ready. Faking completion would be
   worse than showing nothing; creeping is the honest shape of "still working, no
   idea how long". */

const STEPS = [
  ['Waking the server', 0.15],
  ['Loading your deck', 0.35],
  ['Warming the Maltese recogniser', 1.0],
];

// The recogniser load is the unknown. This is roughly how long a cold container
// takes, and is used only to shape the creep — never to declare success.
const EXPECTED_WARMUP_MS = 45000;

/* How long a static build will hold the door for a recogniser that lives somewhere
   else. Long enough to cover a Space that is merely loading its weights, nowhere
   near long enough to sit through a full cold boot: reading, listening and typing
   never needed it, and the container carries on waking whether or not this page is
   watching. */
const REMOTE_WARM_MS = 30000;

let el = null;

function ui() {
  if (el) return el;
  el = document.createElement('div');
  el.className = 'splash';
  el.innerHTML = `
    <div class="splash-inner">
      <h1>Nitkellmu</h1>
      <p class="splash-sub">Speak Maltese</p>
      <div class="splash-bar"><i></i></div>
      <p class="splash-step">Starting…</p>
      <p class="splash-note" hidden></p>
    </div>`;
  document.body.append(el);
  return el;
}

function paint(fraction, label, note) {
  const root = ui();
  root.querySelector('.splash-bar i').style.width = `${Math.round(fraction * 100)}%`;
  root.querySelector('.splash-step').textContent = label;
  const noteEl = root.querySelector('.splash-note');
  noteEl.textContent = note || '';
  noteEl.hidden = !note;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Wake the server, retrying while it boots.

    A sleeping Hugging Face Space answers the first request only after the
    container is up, which can be a minute; a single fetch would simply fail. */
async function wake(attempts = 40) {
  let lastErr;
  for (let i = 0; i < attempts; i += 1) {
    try {
      return await (await fetchOk('/api/bootstrap')).json();
    } catch (err) {
      lastErr = err;
      paint(0.05 + Math.min(0.09, i * 0.01), 'Waking the server',
        i > 2 ? 'The host sleeps when idle — this only happens on the first visit.' : '');
      await sleep(Math.min(3000, 600 + i * 300));
    }
  }
  throw lastErr || new Error('Server did not respond');
}

async function fetchOk(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res;
}

/** Poll until the recogniser is loaded, creeping the bar meanwhile. */
async function waitForModel(from, to, giveUpAfterMs = 4 * 60 * 1000) {
  const startedAt = Date.now();
  for (;;) {
    // Reading, listening and typing never needed the recogniser. If it is taking
    // implausibly long — or the server died partway through loading it, which a
    // free tier does — let the learner in rather than holding the door forever.
    if (Date.now() - startedAt > giveUpAfterMs) return null;
    let health;
    try {
      health = await (await fetchOk('/api/health')).json();
    } catch {
      health = null;                       // a restart mid-wait: keep waiting
    }
    if (health && (health.ready || !health.warming)) return health;

    // Asymptotic: 63% of the remaining span after one expected warmup, 86% after
    // two, never 100%. It cannot overshoot into a lie about being finished.
    const elapsed = Date.now() - startedAt;
    const progress = 1 - Math.exp(-elapsed / EXPECTED_WARMUP_MS);
    paint(from + (to - from) * progress, 'Warming the Maltese recogniser',
      elapsed > 12000
        ? 'Loading the speech model — a few hundred megabytes, once per restart.'
        : 'You can read and listen already; speaking needs this.');
    await sleep(1000);
  }
}

/* Wake the recogniser that lives somewhere else.

   A static build pointed at a deployment never fetches the model — a phone cannot
   hold it — so its wait is not a download, it is a container waking up. A free
   Space sleeps when idle and then reads ~1.2GB of weights, and whoever asks first
   pays for all of it. Paying it here is the same wait moved somewhere it can be
   seen and explained, instead of arriving as a mic button that does nothing for a
   minute the first time somebody speaks.

   There is no percentage to report — the server cannot say how far through the load
   it is — so this creeps like `waitForModel` and snaps to full only on a real
   answer. Returns whether it got one. */
async function warmRemote(base, from, to, onNotice) {
  const startedAt = Date.now();
  for (;;) {
    let health = null;
    try {
      health = await (await fetchOk(`${base}/api/health`)).json();
    } catch {
      /* A sleeping Space answers nothing until its container is up, and the holding
         pages it serves meanwhile carry no CORS headers — so a boot in progress
         arrives here as a network failure rather than a status. Either way it is
         not ready, and the request itself is what started it waking. */
    }
    if (health && (health.ready || !health.warming)) return true;

    const elapsed = Date.now() - startedAt;
    if (elapsed > REMOTE_WARM_MS) {
      onNotice?.('The recogniser is still waking up — carry on; speaking will '
        + 'start working in a moment.');
      return false;
    }
    paint(from + (to - from) * (1 - Math.exp(-elapsed / EXPECTED_WARMUP_MS)),
      'Waking the Maltese recogniser',
      elapsed > 8000
        ? 'It runs on a host that sleeps when idle, so the first visit waits for it.'
        : 'You can read and listen already; speaking needs this.');
    await sleep(1000);
  }
}

/** Run the whole startup. Resolves with the bootstrap payload. */
export async function run({ onDeck, onStatic, onModel, onNotice } = {}) {
  paint(0.03, 'Starting');

  // A static build ships its answers as files. Try that first: it is one cheap
  // request, it tells us there is no server to wait for, and on Pages there never
  // was one to wake.
  let boot = null;
  try {
    const res = await fetch('api/bootstrap.json');
    if (res.ok) boot = await res.json();
  } catch { /* not a static build */ }

  if (boot?.static) {
    paint(STEPS[0][1], 'Loading your deck');
    const [deck, dialogues, audio] = await Promise.all([
      (await fetchOk('api/deck.json')).json(),
      (await fetchOk('api/dialogues.json')).json(),
      (await fetchOk('audio/index.json')).json(),
    ]);
    await onStatic?.({ boot, dialogues, audio });
    await onDeck?.(deck.cards);

    /* The recogniser is the only large thing here and the only one with nothing to
       fall back to, so it gets the rest of the bar either way. Which wait it is
       depends on where it runs: a download on this device, with a real percentage,
       or a container waking up at `stt_base`, with none to be had. */
    if (boot.stt_base) {
      paint(STEPS[1][1], 'Waking the Maltese recogniser');
      await warmRemote(boot.stt_base, STEPS[1][1], STEPS[2][1], onNotice);
    } else if (onModel) {
      const done = await onModel((f) => paint(
        STEPS[1][1] + (STEPS[2][1] - STEPS[1][1]) * f,
        'Downloading the Maltese recogniser',
        f > 0.02 ? 'About 200MB, once — it is cached after this, and everything '
          + 'except speaking works already.' : ''));
      if (done === false) paint(1, 'Ready — speaking needs WebGPU, so it is off');
    }
    paint(1, 'Mela — ejja nibdew!');
    await sleep(280);
    dismiss();
    return boot;
  }

  paint(0.05, STEPS[0][0]);
  boot = await wake();
  paint(STEPS[0][1], STEPS[1][0]);

  const { cards } = await (await fetchOk('/api/deck')).json();
  await onDeck?.(cards);
  paint(STEPS[1][1], STEPS[2][0]);

  // Speech recognition is the only thing that needs the model. If this
  // deployment has no local recogniser there is nothing to wait for.
  if (boot.capabilities?.stt?.some((p) => p === 'wav2vec2' || p === 'faster_whisper')) {
    const health = await waitForModel(STEPS[1][1], STEPS[2][1]);
    if (!health) {
      paint(1, 'Starting without speech input');
      await sleep(1200);
    }
  }

  paint(1, 'Mela — ejja nibdew!');
  await sleep(280);
  dismiss();
  return boot;
}

export function fail(message) {
  const root = ui();
  root.classList.add('is-failed');
  root.querySelector('.splash-step').textContent = 'Could not start';
  const note = root.querySelector('.splash-note');
  note.textContent = `${message} — reload to try again.`;
  note.hidden = false;
}

export function dismiss() {
  if (!el) return;
  el.classList.add('is-done');
  setTimeout(() => { el?.remove(); el = null; }, 320);
}

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

/** Run the whole startup. Resolves with the bootstrap payload. */
export async function run({ onDeck, onStatic } = {}) {
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

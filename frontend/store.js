/* Learner state, on the device.

   Everything here used to live in a SQLite file next to the server. That was fine
   for one person on one laptop and wrong for anything else: a public deployment
   gave every visitor the same review history, and free hosting throws the disk
   away on each restart — so the one irreplaceable thing in the app was also the
   least durable.

   IndexedDB instead. Three stores:

     cards   — the deck, seeded from /api/deck and refreshed when it changes
     states  — per-card schedule (FSRS), the part that is genuinely yours
     reviews — the log, for the progress charts and the streak

   Settings stay in localStorage: they are tiny, synchronous access keeps the
   first paint simple, and losing them costs nothing. */

import * as srs from './srs.js';

const DB_NAME = 'speak-maltese';
const DB_VERSION = 1;

let dbp = null;

function open() {
  if (dbp) return dbp;
  dbp = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('cards')) {
        const cards = db.createObjectStore('cards', { keyPath: 'id' });
        cards.createIndex('tier', 'tier');
        cards.createIndex('topic', 'topic');
      }
      if (!db.objectStoreNames.contains('states')) {
        const states = db.createObjectStore('states', { keyPath: 'cardId' });
        states.createIndex('due', 'due');
        states.createIndex('state', 'state');
      }
      if (!db.objectStoreNames.contains('reviews')) {
        const reviews = db.createObjectStore('reviews', { keyPath: 'id', autoIncrement: true });
        reviews.createIndex('at', 'at');
        reviews.createIndex('cardId', 'cardId');
      }
    };
    req.onsuccess = () => {
      const db = req.result;
      // If another tab (or a delete) needs a version change, hold the door open
      // for it. A connection nobody closes blocks the other side indefinitely,
      // and the symptom is this tab's *next* open hanging with no error at all.
      db.onversionchange = () => { db.close(); dbp = null; };
      resolve(db);
    };
    req.onerror = () => reject(req.error || new Error('IndexedDB unavailable'));
    req.onblocked = () => reject(new Error('Another tab is holding an older database open'));

    /* An open request that is queued behind a pending delete or upgrade never
       fires success, error *or* blocked — it simply waits. Without a deadline the
       app sits on "Loading your deck" forever showing nothing, which is what a
       second tab or an interrupted reset actually produces. Fail loudly instead;
       the splash turns it into a message and a reload fixes it. */
    setTimeout(() => reject(new Error(
      'The local database did not open — close other tabs with this app and reload')),
      8000);
  });
  // Do not let one failure poison the page. `onblocked` clears as soon as the
  // other tab closes, and a quota hiccup can pass — caching the rejected promise
  // meant every later read failed with a stale error until a reload.
  dbp.catch(() => { dbp = null; });
  return dbp;
}

async function tx(names, mode, fn) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const t = db.transaction(names, mode);
    // Resolve on `complete`, not on the last request's success: in a readwrite
    // transaction the writes are not durable until the transaction commits, and
    // resolving early let a caller read back its own write and miss it.
    let result;
    t.oncomplete = () => resolve(result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error || new Error('transaction aborted'));
    result = fn(names.length === 1 ? t.objectStore(names[0]) : t);
    if (result && typeof result.then === 'function') {
      reject(new Error('store callbacks must be synchronous'));
    }
  });
}

const asPromise = (req) => new Promise((resolve, reject) => {
  req.onsuccess = () => resolve(req.result);
  req.onerror = () => reject(req.error);
});

/* ── Deck ────────────────────────────────────────────────────────────────── */

/** Refresh the shipped deck, leaving everything the learner has added alone.

    The deck is content and ships with the app; the schedule is the learner's.
    A card that disappears from the deck keeps its state row — harmless, and it
    means removing a word from a TSV by accident does not destroy its history.

    This used to `clear()` the card store first, which quietly deleted every
    phrase earned in conversation on the next page load: drill cards are written
    by `addCards` and are not in `/api/deck`, so they were wiped while their state
    rows survived as orphans that `buildQueue` then filtered out. Cards the server
    did not send are now left where they are.

    Runs on every boot, behind the startup screen, so what it costs is what the
    learner waits. It asked the schedule store whether each of the 470 cards had a
    row — 470 separate `get`s, each a request object with its own callback, to
    answer a question one `getAllKeys()` answers in a single read. That is the whole
    reason it is a keyed store. */
export async function seedDeck(cards) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const t = db.transaction(['cards', 'states'], 'readwrite');
    const cardStore = t.objectStore('cards');
    const stateStore = t.objectStore('states');
    const shipped = new Set(cards.map((c) => c.id));
    let added = 0;

    // Drop only cards that came from a previous deck and are no longer in it —
    // never anything the learner produced.
    const existing = cardStore.getAll();
    existing.onsuccess = () => {
      for (const old of existing.result) {
        if (old.source !== 'drill' && !shipped.has(old.id)) cardStore.delete(old.id);
      }
    };

    // Keys only: which cards already have a schedule. The rows themselves are the
    // learner's and are not read here, let alone rewritten — overwriting one would
    // reset that card's schedule on every boot.
    const scheduled = stateStore.getAllKeys();
    scheduled.onsuccess = () => {
      const have = new Set(scheduled.result);
      for (const c of cards) {
        cardStore.put(c);
        if (!have.has(c.id)) {
          stateStore.put({ cardId: c.id, ...blankState() });
          added += 1;
        }
      }
    };

    t.oncomplete = () => resolve({ cards: cards.length, fresh: added });
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

/* One definition of a fresh schedule, owned by the scheduler. It was written out
   twice — here and as `srs.newState` — and had already drifted: only this copy
   carried `suspended`, and the next field added would have gone to one of them. */
function blankState() {
  return { ...srs.newState(), suspended: 0 };
}

/* Reads resolve on the request rather than on transaction completion: a readonly
   transaction has nothing to commit, so there is no window in which the answer
   could still change. Writes go through `tx`, which waits for `complete`. */

async function readAll(store, indexName, range) {
  const db = await open();
  const s = db.transaction([store], 'readonly').objectStore(store);
  return asPromise((indexName ? s.index(indexName) : s).getAll(range));
}

async function readOne(store, key) {
  const db = await open();
  return asPromise(db.transaction([store], 'readonly').objectStore(store).get(key));
}

export const getCards = () => readAll('cards');
export const getCard = (id) => readOne('cards', id);
export const getStates = () => readAll('states');
export const getState = (id) => readOne('states', id);

export async function putState(cardId, st) {
  return tx(['states'], 'readwrite', (s) => { s.put({ cardId, ...st }); });
}

export async function addCards(cards) {
  return tx(['cards', 'states'], 'readwrite', (t) => {
    const cardStore = t.objectStore('cards');
    const stateStore = t.objectStore('states');
    for (const c of cards) {
      cardStore.put(c);
      const get = stateStore.get(c.id);
      get.onsuccess = () => { if (!get.result) stateStore.put({ cardId: c.id, ...blankState() }); };
    }
  });
}

export async function logReview(entry) {
  return tx(['reviews'], 'readwrite', (s) => { s.add(entry); });
}

export const getReviews = () => readAll('reviews');

export async function suspend(cardId, on = true) {
  const st = await getState(cardId);
  if (st) await putState(cardId, { ...st, suspended: on ? 1 : 0 });
}

/** Wipe everything the learner has done. Used by the "start over" control. */
export async function reset() {
  return tx(['cards', 'states', 'reviews'], 'readwrite', (t) => {
    for (const name of ['cards', 'states', 'reviews']) t.objectStore(name).clear();
  });
}

/** Whole-database export, so progress can move between devices or be kept. */
export async function exportAll() {
  const [cards, states, reviews] = await Promise.all([getCards(), getStates(), getReviews()]);
  return { version: DB_VERSION, exportedAt: new Date().toISOString(), cards, states, reviews };
}

export async function importAll(dump) {
  if (!dump || !Array.isArray(dump.states)) throw new Error('Not a progress export');
  // `reset()` clears the deck too, so a dump without cards would leave the app
  // with schedules pointing at nothing. Refuse rather than half-wipe.
  if (!Array.isArray(dump.cards) || !dump.cards.length) {
    throw new Error('Export contains no cards — nothing was changed');
  }
  await reset();
  return tx(['cards', 'states', 'reviews'], 'readwrite', (t) => {
    for (const c of dump.cards || []) t.objectStore('cards').put(c);
    for (const s of dump.states || []) t.objectStore('states').put(s);
    // Drop the old autoincrement keys so the imported log appends cleanly.
    for (const r of dump.reviews || []) t.objectStore('reviews').add({ ...r, id: undefined });
  });
}

/* ── Settings ────────────────────────────────────────────────────────────── */

const SETTINGS_KEY = 'sm.settings';

export const DEFAULT_SETTINGS = {
  voice: 'mt-MT-GraceNeural',
  rate: 0.95,
  show_english: true,
  autoplay: true,
  daily_new: 15,
  daily_review: 120,
  target_retention: 0.9,
  /* On by default. On the static build there is no server recogniser, so this is
     the difference between an app you can speak to and one you can only type at.
     It costs a 2.1MB download on first use — it was ~200MB and needed a WebGPU
     adapter, which is what the startup screen's progress bar was built for and why
     the settings dialog still says how big it is. Now it turns itself off only
     where there is no WebAssembly. */
  local_stt: true,
};

export function loadSettings() {
  try {
    return { ...DEFAULT_SETTINGS, ...(JSON.parse(localStorage.getItem(SETTINGS_KEY)) || {}) };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(s) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
}

/* ── Two markers from a model this app no longer ships ─────────────────────

   The recogniser used to be able to take the tab down with it, and both of these
   remembered that. `sm.sttLoading` was written before a load and cleared after it,
   so a marker still present at the next boot meant the tab had died mid-load and the
   load should not be walked into again. `sm.modelTooBig` recorded a device where
   that had happened, and recognition was refused there for good — a fact about the
   hardware rather than a preference of the learner's.

   Both were about ~200MB of weights instantiated on the GPU against the 250-350MB
   WebKit gives a page on an iPhone SE. The recogniser is 2.1MB on the CPU now and
   has no such ceiling to reach.

   So the code is gone and the keys are cleared rather than read. `sm.modelTooBig`
   because a device still carrying it is carrying a verdict about a model it no
   longer runs, and nothing left in the app would ever lift it. `sm.sttLoading`
   because nothing had read it since the model shrank — it was a flag written on
   every boot that no code consulted, and reconnecting it would have meant switching
   speech off for anyone whose tab was interrupted for an ordinary reason.

   A no-op on every device that never had them. */
for (const stale of ['sm.modelTooBig', 'sm.sttLoading']) {
  try {
    localStorage.removeItem(stale);
  } catch { /* private mode: nothing was stored, so there is nothing to clear */ }
}

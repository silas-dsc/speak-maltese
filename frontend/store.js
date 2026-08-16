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
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error('IndexedDB unavailable'));
    req.onblocked = () => reject(new Error('Another tab is holding an older database open'));
  });
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

/** Replace the deck, keeping every schedule whose card survives.

    The deck is content and ships with the app; the schedule is the learner's.
    A card that disappears from the deck keeps its state row — harmless, and it
    means removing a word from a TSV by accident does not destroy its history. */
export async function seedDeck(cards) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const t = db.transaction(['cards', 'states'], 'readwrite');
    const cardStore = t.objectStore('cards');
    const stateStore = t.objectStore('states');
    let added = 0;
    cardStore.clear();
    for (const c of cards) {
      cardStore.put(c);
      // Only create a state row if there isn't one; overwriting would reset the
      // schedule of every card on every boot.
      const get = stateStore.get(c.id);
      get.onsuccess = () => {
        if (!get.result) {
          stateStore.put({ cardId: c.id, ...blankState() });
          added += 1;
        }
      };
    }
    t.oncomplete = () => resolve({ cards: cards.length, fresh: added });
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

function blankState() {
  return {
    stability: 0, difficulty: 0, reps: 0, lapses: 0,
    state: 'new', step: 0, due: null, lastReview: null,
    prodReps: 0, prodCorrect: 0, suspended: 0,
  };
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

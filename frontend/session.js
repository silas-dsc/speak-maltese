/* The conversation you are in the middle of, kept across a reload.

   Everything else the app knows about you already survives — the deck and its
   schedule in IndexedDB, the settings and the finished scenes in localStorage —
   but the conversation itself lived only in the DOM. Reload the page and the
   scene restarted from its first line, which is a small loss on a laptop and a
   real one on a phone, where a backgrounded tab is reloaded by the browser
   whenever it feels like it. It also made the app's own reload after a deploy
   cost something, and that reload exists to stop a worse problem.

   localStorage, not IndexedDB: a conversation is a few kilobytes, restoring it has
   to happen before the first paint to avoid a flash of empty chat, and losing one
   costs a scene rather than a month of review history.

   Turns are stored as what they *said*, not as the markup that showed it — role,
   Maltese, English, the verdict line and any correction — so a change to how a
   bubble is built cannot resurrect stale HTML from someone's phone. */

const KEY = 'sm.drillSession';
const VERSION = 1;

/* A scene is at most 35 nodes, but a learner can retry a line as often as they
   like, so the log is bounded. The cap is the number of *turns*, and the oldest
   go first: what you said five minutes ago matters less than being able to save
   at all, and a full localStorage throws on write. */
const LIMIT = 300;

/** A store over any Storage-shaped thing — the real localStorage in the app, a
    stub in the tests. Every method is safe to call when storage is unavailable or
    full, because a conversation is never worth breaking a turn over. */
export function store(storage = globalThis.localStorage) {
  const read = () => {
    try {
      return JSON.parse(storage.getItem(KEY));
    } catch {
      return null;
    }
  };

  return {
    /** The saved conversation, or null if there is nothing usable.

        A shape from an older build is dropped rather than migrated: it is one
        scene, and guessing at half-familiar data is how you resurrect a bug that
        was already fixed. */
    load() {
      const s = read();
      if (!s || s.v !== VERSION || !s.dialogue || !s.present) return null;
      return { ...s, turns: Array.isArray(s.turns) ? s.turns : [] };
    },

    /** Replace the saved conversation. Returns what was saved, for chaining. */
    save(session) {
      const trimmed = {
        ...session,
        v: VERSION,
        turns: (session.turns || []).slice(-LIMIT),
      };
      try {
        storage.setItem(KEY, JSON.stringify(trimmed));
      } catch {
        // Full, or private mode with storage denied. The conversation on screen is
        // unaffected; only its survival across a reload is lost.
      }
      return trimmed;
    },

    clear() {
      try {
        storage.removeItem(KEY);
      } catch { /* see save() */ }
    },
  };
}

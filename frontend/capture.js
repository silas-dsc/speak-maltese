/* Which container to record in — decided by what this device actually produces,
   not by what it claims it can.

   `MediaRecorder.isTypeSupported` is a claim, and on iOS it is sometimes false. It
   answered yes to `audio/webm;codecs=opus` on a real iPhone and then returned five
   bytes — a container header with no audio in it — for a four-and-a-half-second
   recording. Nothing in the API says so: the recording simply comes back empty, the
   app says "nothing recorded", and trying again does exactly the same thing, because
   the next attempt asks for the same container that has already been shown not to
   work. `audio/mp4` is the one iOS records for real.

   So the choice is remembered instead of recomputed. A container that produced
   nothing here is struck off and the next one is used, and once one has demonstrably
   worked it is the one used from then on. The knowledge is per device, because that
   is where the fault is — no user-agent sniffing, which is how you end up with a
   list of browsers to maintain and the same bug on the next one. */

const KEY = 'sm.capture';
const VERSION = 1;

/* Opus in WebM first: it is the smallest, every recogniser in the chain reads it,
   and on the platforms where it works it works well. `audio/mp4` (AAC) last, which
   in practice means first on Apple hardware, since the two above it get struck off
   the moment they come back empty. */
export const CANDIDATES = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'];

/** The best container to ask for: what is known to work here, else the first the
    browser admits to that has not already failed. `''` means "let the browser
    choose", which is all that is left when nothing is supported. */
export function pickMime({ supported = () => false, verified = '', blocked = [] } = {}) {
  if (verified && supported(verified) && !blocked.includes(verified)) return verified;
  return CANDIDATES.find((m) => !blocked.includes(m) && supported(m)) || '';
}

/** A file name for an upload, so the server sees the format it is being given. */
export function fileNameFor(mime = '') {
  const ext = mime.includes('mp4') ? 'mp4' : mime.includes('ogg') ? 'ogg' : 'webm';
  return `speech.${ext}`;
}

/** What this device has been seen to do, across sessions — over any Storage-shaped
    thing, so the tests can hand it one that misbehaves. Every method tolerates
    storage that is full or refuses to answer: the recorder still records. */
export function store(storage = globalThis.localStorage) {
  const read = () => {
    try {
      const s = JSON.parse(storage.getItem(KEY));
      return s && s.v === VERSION ? s : { v: VERSION, verified: '', blocked: [] };
    } catch {
      return { v: VERSION, verified: '', blocked: [] };
    }
  };
  const write = (s) => {
    try {
      storage.setItem(KEY, JSON.stringify(s));
    } catch { /* nothing here is worth failing a recording over */ }
  };

  return {
    verified: () => read().verified,
    blocked: () => read().blocked,

    /** This container produced audio. Remember it and stop guessing. */
    verify(mime) {
      const s = read();
      write({ ...s, verified: mime, blocked: s.blocked.filter((m) => m !== mime) });
    },

    /** This container produced nothing. Never ask for it on this device again. */
    block(mime) {
      if (!mime) return;
      const s = read();
      if (s.blocked.includes(mime)) return;
      write({
        v: VERSION,
        verified: s.verified === mime ? '' : s.verified,
        blocked: [...s.blocked, mime],
      });
    },
  };
}

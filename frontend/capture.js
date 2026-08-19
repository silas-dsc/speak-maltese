/* Which container to record in — decided by what this device actually produces,
   not by what it claims it can.

   `MediaRecorder.isTypeSupported` is a claim, and it is not always true. An iPhone
   answered yes to `audio/webm;codecs=opus` and returned five bytes — a container
   header with no audio in it — for a four-and-a-half-second recording, twice. What
   it will not explain on its own is that recognition on the same phone worked
   *sometimes*: a format that never encodes never works. So there are two candidate
   faults, they look identical from here, and only one of them is the format's.

   This module holds the format half: the order to try, what this device has been
   seen to produce, and the reasoning about which of the two faults a failed
   recording actually was (see `diagnose`). A container is struck off only with
   evidence that sound reached it — never on a single failure that a muted
   microphone explains just as well. Kept per device rather than per browser name,
   because that is where the fault is: a list of user-agent strings to maintain is
   how you end up with the same bug on the next browser. */

const KEY = 'sm.capture';
const VERSION = 1;

/* `audio/mp4` first. It used to be last, on the reasoning that Opus in WebM is
   smaller and universally readable, and that Apple hardware would reach mp4 anyway
   "since the two above it get struck off the moment they come back empty".

   That reasoning had two holes. Getting struck off costs a *turn*: an iPhone SE
   answers yes to `audio/webm;codecs=opus`, records 2444ms into it and hands back
   five bytes, and the utterance is gone — no reordering afterwards brings it back.
   And the size argument expired when recognition moved onto the device: the 2.1MB
   model decodes through `decodeAudioData`, which reads AAC natively and cares
   nothing for the container, and the only deployment that still uploads audio is one
   that names a remote host.

   So the order is now "what the platform actually implements first". Every browser
   that records mp4 does so correctly; the ones that do not — Chrome, Firefox — say
   so through `isTypeSupported` and fall through to Opus, which they implement
   correctly. Neither has to be recognised by name, which is the point: a list of
   user-agent strings to maintain is how the same bug arrives on the next browser. */
export const CANDIDATES = ['audio/mp4', 'audio/webm;codecs=opus', 'audio/webm'];

/** The best container to ask for: what is known to work here, else the first the
    browser admits to that has not already failed. `''` means "let the browser
    choose", which is all that is left when nothing is supported. */
export function pickMime({ supported = () => false, verified = '', blocked = [] } = {}) {
  if (verified && supported(verified) && !blocked.includes(verified)) return verified;
  return CANDIDATES.find((m) => !blocked.includes(m) && supported(m)) || '';
}

/* Recording is rejected below these, and the numbers are here so the reasoning
   about *why* is in one place with them. */
export const MIN_MS = 250;
export const MIN_BYTES = 600;
/* Peak amplitude, 0..1, from a level meter over the recording. Room tone through a
   phone mic sits well above this; a capture session that has been taken away reads
   as exactly zero. */
export const SILENT_PEAK = 0.005;

/** Why a recording came back with nothing in it, and what to do about it.

    The first version of this blamed the learner ("too short"). The second reported
    the numbers, which was honest and left the app with nothing to act on. This one
    attributes the failure, because the two causes are opposites and the fixes point
    in different directions:

    * the microphone produced no sound — iOS mutes a capture track when another app
      takes the mic or the page is backgrounded, and `readyState` still reads live,
      so a stream that looks fine records silence. The stream has to be thrown away
      and re-acquired; the format is innocent.
    * the microphone produced sound and the encoder dropped it — the container was
      accepted and never written into. That one is the format's fault and it gets
      struck off.

    Without a level meter the two are indistinguishable, so an unmetered failure
    concludes nothing: it re-acquires the stream, which is the cheaper and more
    likely fix, and asks for metering on the next attempt rather than condemning a
    format that may well have been working a minute ago. */
export function diagnose({ ms = 0, bytes = 0, chunks = 0, mime = '', peak = null } = {}) {
  if (ms < MIN_MS) {
    return { ok: false, blame: 'short', block: false, stale: false, meter: false,
             reason: `only ${(ms / 1000).toFixed(1)}s of audio` };
  }
  if (bytes >= MIN_BYTES) return { ok: true, blame: '', block: false, stale: false, meter: false };

  const measured = `${ms}ms recorded but only ${bytes} bytes captured `
    + `(${chunks} chunk${chunks === 1 ? '' : 's'}, ${mime || 'no mime'})`;

  if (peak === null) {
    return { ok: false, blame: 'unknown', block: false, stale: true, meter: true,
             reason: `${measured} — reopening the microphone, please try again` };
  }
  if (peak < SILENT_PEAK) {
    return { ok: false, blame: 'silence', block: false, stale: true, meter: true,
             reason: 'the microphone went silent — another app may have taken it. '
                     + 'Reopened it, please try again' };
  }
  return { ok: false, blame: 'encoder', block: true, stale: true, meter: true,
           reason: `${measured} — sound was there, ${mime || 'that format'} dropped it, `
                   + 'so the next attempt uses another format' };
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

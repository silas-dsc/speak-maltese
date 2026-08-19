/* Service worker: make the app usable with no network at all.

   Everything the app *says* is pre-rendered audio and everything it shows is a
   fixed shell, so both cache cleanly. Speech recognition runs on the backend, so
   answering by voice still needs the server — but reading, listening and typing
   all keep working on a plane or a bad connection.

   Two strategies, chosen per request:
   * shell (HTML/CSS/JS/images) — stale-while-revalidate, so a cached copy paints
     instantly and the next load picks up any change.
   * audio (/api/tts) — cache-first and permanent, because a given sentence at a
     given voice and rate never changes, and re-fetching it is pure latency. */

/* Stamped with a hash of the built shell by scripts/build_static.py, and left as
   `dev` when the app is served from the repo.

   Without it a deploy reached the device in pieces. Stale-while-revalidate serves
   the cached copy and refreshes in the background, so a page open across a deploy
   ran new `api/dialogues.json` against the previous `app.js` — which is how a
   prompt that had been given a Maltese frame to show appeared with the frame
   missing. A new hash is a new cache, dropped and refetched whole on the next
   load, so the shell can never be half of one build and half of another. */
const BUILD = 'dev';
const SHELL = `shell-${BUILD}`;
/* Not per build. A sentence at a given voice and rate is the same MP3 forever, and
   23MB of them should not be re-downloaded because a stylesheet changed. */
const AUDIO = 'audio-v6';

/* Relative to the worker's scope, not to the origin. This app is served from the
   root by the FastAPI build and from /speak-maltese/ by GitHub Pages, and an
   absolute '/app.js' would cache the wrong thing — or nothing — on the second. */
const SHELL_ASSETS = [
  './', './index.html', './style.css', './manifest.webmanifest',
  // The client is ES modules. Missing one of these offline would not degrade the
  // app, it would fail to boot at all.
  './app.js', './srs.js', './store.js', './schedule.js', './splash.js',
  './nanostt.js', './text.js', './dialogue.js', './session.js', './capture.js',
  './games.js',
  /* The recogniser itself. 2.1MB, and precaching it is the difference between an app
     that listens on a plane and one that only reads there — the 200MB model it
     replaces could never have been in here. */
  './stt/model.onnx', './stt/vocab.txt',
  /* The static build's data, and the manifest that maps a line to its MP3. Absent
     on the FastAPI build, where these are live endpoints — `add` failures are
     tolerated below, so listing them costs nothing there. Without them a device
     that installs a new build and then goes offline has a fresh shell and no deck
     to boot it with, because the per-build cache is empty of everything it did not
     precache. */
  './api/bootstrap.json', './api/deck.json', './api/dialogues.json',
  './api/grammar.json', './api/games.json', './audio/index.json',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(SHELL)
      // addAll is atomic: one 404 would leave the app with no cache at all, so
      // each asset is added independently and a miss is tolerated.
      .then((c) => Promise.allSettled(SHELL_ASSETS.map((a) => c.add(a))))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => k !== SHELL && k !== AUDIO).map((k) => caches.delete(k)),
      ))
      .then(() => self.clients.claim()),
  );
});

/* Which strategy a request gets. Named, and written as one expression per case,
   so the routing can be tested by calling it rather than by grepping for it —
   tests/test_api.py does exactly that. */
function routeFor(url) {
  /* Synthesised speech is immutable for a given (text, voice, rate): the server
     path asks for it by those, and the static build ships it as audio/<32 hex>.mp3
     named after them. Both belong in the audio cache, which is not named after the
     build — 23MB re-downloaded on every deploy is not a cache. `audio/index.json`,
     the line→file manifest, is deliberately not matched here: it changes when the
     audio does, so it belongs with the shell.

     Written as one self-contained expression per case, with nothing from the
     module around it, so tests/test_api.py can lift this function out and run the
     real thing rather than grepping for it. */
  if (url.pathname.endsWith('/api/tts')
      || /\/audio\/[0-9a-f]{32}\.mp3$/.test(url.pathname)) return 'audio';
  // Live server state must never be served stale. The static build has no live
  // state: its api/*.json are immutable files and cache like any other asset.
  if (url.pathname.includes('/api/') && !url.pathname.endsWith('.json')) return 'network';
  return 'shell';
}

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  const route = routeFor(url);

  if (route === 'audio') {
    event.respondWith(
      caches.open(AUDIO).then(async (cache) => {
        const hit = await cache.match(request);
        if (hit) return hit;
        const res = await fetch(request);
        if (res.ok) cache.put(request, res.clone());
        return res;
      }),
    );
    return;
  }

  if (route === 'network') return;

  event.respondWith(
    caches.open(SHELL).then(async (cache) => {
      const hit = await cache.match(request);
      const fresh = fetch(request)
        .then((res) => {
          if (res.ok) cache.put(request, res.clone());
          return res;
        })
        // Offline with nothing cached: `hit` is undefined, and respondWith rejects
        // on undefined rather than failing the request cleanly. Answer with an
        // explicit offline response instead.
        .catch(() => hit || new Response('', { status: 504, statusText: 'Offline' }));
      return hit || fresh;
    }),
  );
});

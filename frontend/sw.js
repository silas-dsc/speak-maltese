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

const VERSION = 'v3';
const SHELL = `shell-${VERSION}`;
const AUDIO = `audio-${VERSION}`;

const SHELL_ASSETS = ['/', '/index.html', '/style.css', '/app.js', '/manifest.webmanifest'];

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

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  // Synthesised speech is immutable for a given (text, voice, rate).
  if (url.pathname === '/api/tts') {
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

  // Everything else under /api is live state — never serve it stale.
  if (url.pathname.startsWith('/api/')) return;

  event.respondWith(
    caches.open(SHELL).then(async (cache) => {
      const hit = await cache.match(request);
      const fresh = fetch(request)
        .then((res) => {
          if (res.ok) cache.put(request, res.clone());
          return res;
        })
        .catch(() => hit);
      return hit || fresh;
    }),
  );
});

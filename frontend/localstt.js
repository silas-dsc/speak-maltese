/* Speech recognition on the device.

   Recognition is the last thing in this app that needs a server. Everything else
   — the deck, the schedule, the scripted dialogue, the pre-rendered audio — is
   either shipped with the page or lives in IndexedDB. Run the recogniser here too
   and the whole thing works with no backend at all: no cold starts, no sleeping
   container, no round trip per utterance, and it keeps working on a plane.

   The trade is a one-off download of about 200MB, so this is opt-in and off by
   default. Nobody should spend that without being asked.

   Two hard requirements, both measured rather than assumed:

   * **WebGPU.** On WASM the same model runs at 0.22x realtime — a two-second
     utterance takes nine seconds, which is worse than any network. On WebGPU it
     is ~30x realtime, faster than the server round trip. If there is no WebGPU
     there is no local recognition; the server path stays.
   * **Own the CTC decode.** transformers.js's tokenizer path collapses repeated
     frames *after* removing blanks, which eats every Maltese geminate: grazzi →
     grazi, kollox → kolox, irrid → irid. Doing argmax here and merging repeats
     before dropping blanks is the difference between usable and quietly wrong. */

const CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6';

/** Where the ONNX weights live. Same-origin by default — the Space serves them —
    but a static deployment can point this at any host (the Hugging Face Hub is
    the obvious one, since GitHub refuses files over 100MB). */
export const DEFAULT_MODEL_BASE = '/models/';
const MODEL_ID = 'mt-w2v2';

let ready = null;          // in-flight or resolved load
let parts = null;          // { processor, model, idToTok, blankId }

/** Can this browser do it at all? WASM is capable and far too slow to use. */
export function supported() {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

export function isReady() {
  return parts !== null;
}

/** Load the model. Safe to call repeatedly; the first call does the work.
    `onProgress` receives 0..1 — the download dominates, so it is worth showing. */
export function load({ base = DEFAULT_MODEL_BASE, dtype = 'q4f16', onProgress } = {}) {
  if (ready) return ready;
  ready = (async () => {
    if (!supported()) throw new Error('This browser has no WebGPU');

    const { AutoModelForCTC, AutoProcessor, env } = await import(/* @vite-ignore */ CDN);
    env.allowRemoteModels = false;   // never silently fetch a different model
    env.allowLocalModels = true;
    env.localModelPath = base;

    const vocab = await (await fetch(`${base}${MODEL_ID}/vocab.json`)).json();
    const idToTok = [];
    for (const [tok, id] of Object.entries(vocab)) idToTok[id] = tok;

    const processor = await AutoProcessor.from_pretrained(MODEL_ID);
    const model = await AutoModelForCTC.from_pretrained(MODEL_ID, {
      dtype,
      device: 'webgpu',
      progress_callback: (p) => {
        if (onProgress && p.status === 'progress' && p.total) {
          onProgress(Math.min(1, p.loaded / p.total));
        }
      },
    });

    parts = { processor, model, idToTok, blankId: vocab['[PAD]'] ?? vocab['<pad>'] };
    onProgress?.(1);
    return parts;
  })();
  // A failed load must not poison the toggle for the rest of the session.
  ready.catch(() => { ready = null; });
  return ready;
}

export function unload() {
  parts = null;
  ready = null;
}

/* Merge repeated frames, *then* drop the blanks. The other order looks identical
   and silently degeminates — see the note at the top of this file. */
function ctcDecode(ids, idToTok, blankId, delimiter = '|') {
  const out = [];
  let prev = -1;
  for (const id of ids) {
    if (id !== prev) {
      if (id !== blankId) out.push(idToTok[id]);
      prev = id;
    }
  }
  return out.join('').replaceAll(delimiter, ' ').replace(/\s+/g, ' ').trim();
}

function argmax(logits) {
  const [, frames, size] = logits.dims;
  const d = logits.data;
  const ids = new Array(frames);
  for (let f = 0; f < frames; f += 1) {
    let best = 0;
    let bestVal = -Infinity;
    for (let v = 0; v < size; v += 1) {
      const x = d[f * size + v];
      if (x > bestVal) { bestVal = x; best = v; }
    }
    ids[f] = best;
  }
  return ids;
}

/** Whatever the MediaRecorder produced → mono 16kHz, which is what wav2vec2 wants. */
export async function decode(blob) {
  const buf = await blob.arrayBuffer();
  const probe = new OfflineAudioContext(1, 1, 16000);
  const decoded = await probe.decodeAudioData(buf);
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  return (await off.startRendering()).getChannelData(0);
}

/** Transcribe a recording. Same shape as the /api/stt response, minus the
    assessment — the caller asks the server for that, or grades locally. */
export async function transcribe(blob) {
  if (!parts) throw new Error('Local recogniser is not loaded');
  const audio = await decode(blob);
  const inputs = await parts.processor(audio);
  const t0 = performance.now();
  const { logits } = await parts.model(inputs);
  const ms = Math.round(performance.now() - t0);
  return {
    text: ctcDecode(argmax(logits), parts.idToTok, parts.blankId),
    provider: 'wav2vec2-webgpu',
    ms,
    seconds: audio.length / 16000,
  };
}

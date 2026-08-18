/* Speech recognition on the device, small enough to actually be there.

   This replaces a 200MB WebGPU model with a 2.1MB one on the CPU. The old path needed
   both things a phone does not have: 200MB of page memory, against WebKit's 250-350MB
   budget on an iPhone SE, and a WebGPU adapter — so it killed the tab, and the app
   learned to stop offering it. The model here is a distillation of that same 315M
   recogniser into 0.53M parameters (`scripts/distill_stt.py`), convolution-only, so it
   runs on WASM at a fraction of realtime and needs no GPU at all.

   Measured on held-out synthetic speech: 0.95 app score and 88% of answers marked
   correct, against the teacher's 0.98 and 96%. The gap is real and the trade is the
   whole point — a recogniser that is present on every device beats a better one that
   is present on none.

   What has to be right, and is tested rather than assumed:

   * **The features.** The model was trained on NeMo's 64-bin log-mel: pre-emphasis,
     reflect-padded STFT with a 320-sample Hann window inside a 512-point frame, a
     Slaney mel filterbank, and per-bin mean/variance normalisation. Every constant
     came out of the checkpoint's own config, and getting one wrong does not fail — it
     quietly returns worse Maltese. `tests/test_client_nanostt.py` checks this
     implementation against the Python one that produced the training data.
   * **Own the CTC decode.** Merge repeated frames *before* dropping blanks. The other
     order eats every Maltese geminate: grazzi → grazi, kollox → kolox, irrid → irid. */

const ORT_VERSION = '1.23.0';
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

/** Where the model lives. 2.1MB fits in the repository, so unlike the 200MB build this
    is served from our own origin on both the FastAPI app and GitHub Pages — no CDN for
    the weights, and nothing to configure. */
export const DEFAULT_MODEL_BASE = 'stt/';

const SR = 16000;
const N_FFT = 512;
const WIN = 320;          // window_size 0.02s at 16kHz
const HOP = 160;          // window_stride 0.01s
const PREEMPH = 0.97;
const LOG_GUARD = 2 ** -24;
const N_MELS = 64;

let ready = null;
let parts = null;         // { session, idToTok, blankId, mel, window }

/** No WebGPU requirement, and WASM is fast enough here — so the only question is
    whether this is a browser at all. */
export function supported() {
  return typeof WebAssembly === 'object' && typeof OfflineAudioContext !== 'undefined';
}

export function isReady() {
  return parts !== null;
}

/* ── Mel filterbank ──────────────────────────────────────────────────────── */

const F_SP = 200 / 3;
const BREAK_HZ = 1000;
const BREAK_MEL = BREAK_HZ / F_SP;
const LOGSTEP = 0.0690875477931522;    // log(6.4) / 27

const toMel = (f) => (f < BREAK_HZ ? f / F_SP
  : BREAK_MEL + Math.log(Math.max(f, 1e-9) / BREAK_HZ) / LOGSTEP);
const toHz = (m) => (m < BREAK_MEL ? m * F_SP
  : BREAK_HZ * Math.exp(LOGSTEP * (m - BREAK_MEL)));

/** Triangular filters with Slaney normalisation — librosa's `htk=False`, NeMo's
    default. Returned as `nMels` rows of `nFreqs`, which is the order the matmul below
    wants. */
export function melFilters(nFreqs = N_FFT / 2 + 1, nMels = N_MELS, sampleRate = SR) {
  const lo = toMel(0);
  const hi = toMel(sampleRate / 2);
  const pts = new Float64Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i += 1) {
    pts[i] = toHz(lo + ((hi - lo) * i) / (nMels + 1));
  }
  const bank = [];
  for (let m = 0; m < nMels; m += 1) {
    const row = new Float32Array(nFreqs);
    const [a, b, c] = [pts[m], pts[m + 1], pts[m + 2]];
    const enorm = 2 / (pts[m + 2] - pts[m]);
    for (let k = 0; k < nFreqs; k += 1) {
      const f = (k * sampleRate) / 2 / (nFreqs - 1);
      row[k] = Math.max(0, Math.min((f - a) / (b - a), (c - f) / (c - b))) * enorm;
    }
    bank.push(row);
  }
  return bank;
}

/* ── FFT ─────────────────────────────────────────────────────────────────── */

/** In-place iterative radix-2. Two hundred of these per utterance, so it does not need
    to be clever, only correct. */
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < half; k += 1) {
        const ur = re[i + k];
        const ui = im[i + k];
        const xr = re[i + k + half];
        const xi = im[i + k + half];
        const vr = xr * cr - xi * ci;
        const vi = xr * ci + xi * cr;
        re[i + k] = ur + vr;
        im[i + k] = ui + vi;
        re[i + k + half] = ur - vr;
        im[i + k + half] = ui - vi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

/** Hann window of `WIN`, symmetric (`periodic=false`), zero-padded to `N_FFT` and
    centred — which is what torch.stft does internally when win_length < n_fft. */
export function analysisWindow() {
  const w = new Float32Array(N_FFT);
  const pad = (N_FFT - WIN) >> 1;
  for (let i = 0; i < WIN; i += 1) {
    w[pad + i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (WIN - 1));
  }
  return w;
}

/* ── Features ────────────────────────────────────────────────────────────── */

/** Mono 16kHz float samples → (nMels, frames) log-mel, normalised per bin.

    Exported so the parity test can call exactly what the recogniser calls. */
export function features(wave, mel = melFilters(), window = analysisWindow()) {
  const n = wave.length;

  // Pre-emphasis, then reflect padding — NeMo's defaults, and on a two-word answer the
  // edges are a real share of the utterance rather than a rounding detail.
  const emph = new Float32Array(n);
  emph[0] = wave[0];
  for (let i = 1; i < n; i += 1) emph[i] = wave[i] - PREEMPH * wave[i - 1];

  const half = N_FFT >> 1;
  const padded = new Float32Array(n + N_FFT);
  for (let i = 0; i < half; i += 1) padded[i] = emph[Math.min(half - i, n - 1)];
  padded.set(emph, half);
  for (let i = 0; i < half; i += 1) {
    padded[half + n + i] = emph[Math.max(n - 2 - i, 0)];
  }

  const frames = Math.floor((padded.length - N_FFT) / HOP) + 1;
  const nMels = mel.length;
  const out = new Float32Array(nMels * frames);       // row-major (nMels, frames)
  const re = new Float64Array(N_FFT);
  const im = new Float64Array(N_FFT);
  const power = new Float64Array(N_FFT / 2 + 1);

  for (let t = 0; t < frames; t += 1) {
    const off = t * HOP;
    for (let i = 0; i < N_FFT; i += 1) {
      re[i] = padded[off + i] * window[i];
      im[i] = 0;
    }
    fft(re, im);
    for (let k = 0; k <= N_FFT / 2; k += 1) power[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < nMels; m += 1) {
      const row = mel[m];
      let acc = 0;
      for (let k = 0; k < row.length; k += 1) acc += power[k] * row[k];
      out[m * frames + t] = Math.log(acc + LOG_GUARD);
    }
  }

  // `normalize: per_feature` — per mel bin across time, sample variance (n-1), which
  // is the divisor NeMo uses.
  for (let m = 0; m < nMels; m += 1) {
    const base = m * frames;
    let sum = 0;
    for (let t = 0; t < frames; t += 1) sum += out[base + t];
    const mean = sum / frames;
    let sq = 0;
    for (let t = 0; t < frames; t += 1) {
      const d = out[base + t] - mean;
      sq += d * d;
    }
    const std = Math.sqrt(sq / Math.max(1, frames - 1));
    const scale = 1 / (std + 1e-5);
    for (let t = 0; t < frames; t += 1) out[base + t] = (out[base + t] - mean) * scale;
  }
  return { data: out, nMels, frames };
}

/* ── Load ────────────────────────────────────────────────────────────────── */

export function load({ base = DEFAULT_MODEL_BASE, onProgress } = {}) {
  if (ready) return ready;
  ready = (async () => {
    if (!supported()) throw new Error('This browser cannot run WebAssembly');
    onProgress?.(0.05);

    const ort = await import(/* @vite-ignore */ `${ORT_CDN}ort.min.mjs`);
    // The runtime's own .wasm files. Vendoring these is the follow-up that makes the
    // recogniser work offline like the rest of the app; the weights already do.
    ort.env.wasm.wasmPaths = ORT_CDN;
    ort.env.wasm.numThreads = 1;      // no cross-origin isolation on Pages
    onProgress?.(0.3);

    const [bytes, vocabText] = await Promise.all([
      fetch(`${base}model.onnx`).then((r) => {
        if (!r.ok) throw new Error(`model.onnx → ${r.status}`);
        return r.arrayBuffer();
      }),
      fetch(`${base}vocab.txt`).then((r) => {
        if (!r.ok) throw new Error(`vocab.txt → ${r.status}`);
        return r.text();
      }),
    ]);
    onProgress?.(0.75);

    const idToTok = [];
    let blankId = null;
    for (const line of vocabText.split('\n')) {
      if (!line.trim()) continue;
      const cut = line.lastIndexOf(' ');
      const tok = line.slice(0, cut);
      const id = Number(line.slice(cut + 1));
      idToTok[id] = tok;
      if (tok === '<blk>') blankId = id;
    }

    const session = await ort.InferenceSession.create(bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    const tokToId = new Map();
    idToTok.forEach((tok, id) => { if (tok !== undefined) tokToId.set(tok, id); });

    parts = { ort, session, idToTok, tokToId, blankId,
              mel: melFilters(), window: analysisWindow() };
    onProgress?.(1);
    return parts;
  })();
  ready.catch(() => { ready = null; });
  return ready;
}

export function unload() {
  parts = null;
  ready = null;
}

/* ── Scoring the line we asked for ───────────────────────────────────────── */

/* The app never has to transcribe. It shows a line, the learner says it, and the only
   question is whether what came back was that line — which is a strictly easier
   question than "what did they say", and the one this model is much better at. Free
   decoding of the same audio marks 88% of correct answers correct; scoring the known
   target with the CTC forward algorithm gets that to ~100% at a threshold that still
   turns away 95% of near-misses.

   The score is a *ratio* — how well the target explains the audio against how well the
   model's own best path does — so it does not depend on the model being good in
   absolute terms. A weaker model has a weaker greedy path too, and the comparison
   survives. That is what makes one threshold usable across devices and voices. */

const NEG = -1e30;

function logaddexp(a, b) {
  if (a <= NEG) return b;
  if (b <= NEG) return a;
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

/** Target text → token ids, dropping anything the model has no token for: it is a
    character CTC with no punctuation, so `Mingħajr zokkor.` must lose its full stop
    before it can be aligned to anything. Expects text already normalised and
    lower-cased by the caller, which owns the Maltese rules. */
export function encodeTarget(flat, tokToId, space = '▁') {
  const ids = [];
  for (const ch of flat) {
    const tok = ch === ' ' ? space : ch;
    const id = tokToId.get(tok);
    if (id !== undefined) ids.push(id);
  }
  return ids;
}

/** log P(ids | audio), summed over every alignment — the CTC forward algorithm.

    The extended sequence interleaves blanks (`b s1 b s2 … b`) so a path may or may not
    emit one between tokens, and a repeated token *must* have one between its copies.
    That rule is why `irrid` and `irid` are different hypotheses here rather than the
    same one — the distinction greedy decoding throws away. */
export function ctcLogp(logprobs, frames, vocab, ids, blank) {
  if (!ids.length) return NEG;
  const size = 2 * ids.length + 1;
  const ext = new Int32Array(size).fill(blank);
  for (let i = 0; i < ids.length; i += 1) ext[2 * i + 1] = ids[i];

  // A path may skip from s-2 to s only where that does not merge two identical
  // tokens, and never onto a blank.
  const skip = new Uint8Array(size);
  for (let s = 2; s < size; s += 1) {
    skip[s] = ext[s] !== blank && ext[s] !== ext[s - 2] ? 1 : 0;
  }

  let cur = new Float64Array(size).fill(NEG);
  let next = new Float64Array(size);
  cur[0] = logprobs[ext[0]];
  if (size > 1) cur[1] = logprobs[ext[1]];

  for (let t = 1; t < frames; t += 1) {
    const base = t * vocab;
    for (let s = 0; s < size; s += 1) {
      let acc = cur[s];
      if (s >= 1) acc = logaddexp(acc, cur[s - 1]);
      if (s >= 2 && skip[s]) acc = logaddexp(acc, cur[s - 2]);
      next[s] = acc + logprobs[base + ext[s]];
    }
    const swap = cur; cur = next; next = swap;
  }
  return size > 1 ? logaddexp(cur[size - 1], cur[size - 2]) : cur[size - 1];
}

/** The best single path — the most any sequence could have scored on this audio. */
export function greedyLogp(logprobs, frames, vocab) {
  let total = 0;
  for (let t = 0; t < frames; t += 1) {
    let best = -Infinity;
    const base = t * vocab;
    for (let v = 0; v < vocab; v += 1) {
      const x = logprobs[base + v];
      if (x > best) best = x;
    }
    total += best;
  }
  return total;
}

/** 0..~1. Per *frame*, because a long sentence accumulates more log-probability than a
    short one and an unnormalised total would rank `Bonġu` above every full sentence in
    the deck regardless of what was said. Can slightly exceed 1: the numerator sums over
    every alignment while the denominator is one path. */
export function targetConfidence(logprobs, frames, vocab, ids, blank) {
  const gap = (ctcLogp(logprobs, frames, vocab, ids, blank)
    - greedyLogp(logprobs, frames, vocab)) / Math.max(1, frames);
  return Math.exp(gap);
}

/* ── Decode ──────────────────────────────────────────────────────────────── */

/** Merge repeated frames, *then* drop the blanks. The other order looks identical and
    silently degeminates — see the note at the top of this file. */
export function ctcDecode(ids, idToTok, blankId) {
  const out = [];
  let prev = -1;
  for (const id of ids) {
    if (id !== prev) {
      if (id !== blankId) out.push(idToTok[id] ?? '');
      prev = id;
    }
  }
  return out.join('').replaceAll('▁', ' ').replace(/\s+/g, ' ').trim();
}

/** Whatever MediaRecorder produced → mono 16kHz, which is what the model was fed. */
export async function decode(blob) {
  const buf = await blob.arrayBuffer();
  const probe = new OfflineAudioContext(1, 1, SR);
  const decoded = await probe.decodeAudioData(buf);
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * SR), SR);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  return (await off.startRendering()).getChannelData(0);
}

/** Transcribe a recording. Same shape as the /api/stt response, minus the assessment —
    the caller grades locally.

    Pass `target` — already normalised and lower-cased by the caller, which owns the
    Maltese rules — and the result carries `confidence`: how well that exact line explains
    the audio, against the model's own best guess.

    Pass `distractors` too and it also reports whether the target *beat* them, which is
    the number to grade on. Measured on a learner's own recordings, an absolute threshold
    cannot work: the target scored 0.766 where near-misses scored 0.784, so no cut
    separates them, and the value drifts with the speaker anyway. Ranking is scale-free —
    the target won against 24 alternatives 75% of the time on the same clips, where the
    threshold accepted none of them. The transcript is what to *show* when nothing wins. */
export async function transcribe(blob, { target = '', distractors = [] } = {}) {
  if (!parts) throw new Error('Local recogniser is not loaded');
  const audio = await decode(blob);
  const { data, nMels, frames } = features(audio, parts.mel, parts.window);
  const t0 = performance.now();
  const input = new parts.ort.Tensor('float32', data, [1, nMels, frames]);
  const { logprobs } = await parts.session.run({ audio_signal: input });
  const ms = Math.round(performance.now() - t0);

  const [, outFrames, vocab] = logprobs.dims;
  const d = logprobs.data;
  const ids = new Array(outFrames);
  for (let t = 0; t < outFrames; t += 1) {
    let best = 0;
    let bestVal = -Infinity;
    for (let v = 0; v < vocab; v += 1) {
      const x = d[t * vocab + v];
      if (x > bestVal) { bestVal = x; best = v; }
    }
    ids[t] = best;
  }
  const result = {
    text: ctcDecode(ids, parts.idToTok, parts.blankId),
    provider: 'mt-nano-wasm',
    ms,
    seconds: audio.length / SR,
  };

  if (target) {
    const score = (line) => {
      const seq = encodeTarget(line, parts.tokToId);
      return seq.length ? targetConfidence(d, outFrames, vocab, seq, parts.blankId) : 0;
    };
    result.confidence = score(target);

    /* Every alternative is scored against the same audio and the same denominator, so
       only the ordering is being trusted — not the absolute value, which is the part that
       does not survive a change of speaker. */
    let best = -Infinity;
    let bestLine = '';
    for (const line of distractors) {
      if (!line || line === target) continue;
      const c = score(line);
      if (c > best) { best = c; bestLine = line; }
    }
    result.runnerUp = Number.isFinite(best) ? best : null;
    result.runnerUpLine = bestLine;
    result.wins = result.runnerUp === null || result.confidence > result.runnerUp;
  }
  return result;
}

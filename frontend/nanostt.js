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
const DEFAULT_MODEL_BASE = 'stt/';

const SR = 16000;
const N_FFT = 512;
const WIN = 320;          // window_size 0.02s at 16kHz
const HOP = 160;          // window_stride 0.01s
const PREEMPH = 0.97;
const LOG_GUARD = 2 ** -24;
const N_MELS = 64;

/* How far below the loudest frame still counts as speech, and which frame count the
   duration prior is fed. Both inert at these values: `total` is every frame the
   recogniser was handed, which is what the deployed constants were fitted against.
   See the note above `DUR_INTERCEPT` for why neither moves on its own. */
const SPEECH_DROP_DB = 30;
const DUR_FRAMES = 'total';

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
  /* Where each triangle is actually nonzero. A mel filter covers 8 of the 257
     frequency bins on average (2 at the bottom of the range, 23 at the top) and is
     zero across the rest, so projecting through the whole row multiplied by zero
     about 249 times per mel, 64 mels a frame, ~300 frames an utterance.

     Summing only the covered bins is bit-identical — tests/test_client_nanostt.py
     compares the whole feature matrix against NeMo's — and takes a three-second
     utterance from 10.4ms to 4.2ms. Small against the model run, but it is the part
     that happens on the main thread while the learner waits.

     Carried beside the bank rather than on the rows, so a row is still a plain
     Float32Array to anything that indexes or copies it. */
  bank.from = new Int32Array(nMels);
  bank.to = new Int32Array(nMels);

  for (let m = 0; m < nMels; m += 1) {
    const row = new Float32Array(nFreqs);
    const [a, b, c] = [pts[m], pts[m + 1], pts[m + 2]];
    const enorm = 2 / (pts[m + 2] - pts[m]);
    let from = nFreqs;
    let to = 0;
    for (let k = 0; k < nFreqs; k += 1) {
      const f = (k * sampleRate) / 2 / (nFreqs - 1);
      row[k] = Math.max(0, Math.min((f - a) / (b - a), (c - f) / (c - b))) * enorm;
      if (row[k] > 0) {
        if (k < from) from = k;
        to = k + 1;
      }
    }
    bank.from[m] = Math.min(from, to);
    bank.to[m] = to;
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
  const energy = new Float32Array(frames);            // pre-normalisation, see below
  const re = new Float64Array(N_FFT);
  const im = new Float64Array(N_FFT);
  const power = new Float64Array(N_FFT / 2 + 1);

  for (let t = 0; t < frames; t += 1) {
    const off = t * HOP;
    let eAcc = 0;
    for (let i = 0; i < N_FFT; i += 1) {
      re[i] = padded[off + i] * window[i];
      im[i] = 0;
    }
    fft(re, im);
    for (let k = 0; k <= N_FFT / 2; k += 1) power[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < nMels; m += 1) {
      const row = mel[m];
      // Only the bins the triangle covers — see `melFilters`. The fallback is for a
      // bank built by something other than that function, where the whole row is
      // the honest answer.
      const from = mel.from ? mel.from[m] : 0;
      const to = mel.to ? mel.to[m] : row.length;
      let acc = 0;
      for (let k = from; k < to; k += 1) acc += power[k] * row[k];
      out[m * frames + t] = Math.log(acc + LOG_GUARD);
      eAcc += acc;
    }
    /* Kept here because it cannot be recovered later: normalising each mel bin over
       time is precisely the step that throws absolute level away, so the features the
       model sees carry no notion of loud or quiet. */
    energy[t] = eAcc;
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
  return { data: out, nMels, frames, energy };
}

/** `[start, end)` over analysis frames, dropping the quiet head and tail.

    Relative to the loudest frame, so it needs no absolute level and survives a quiet
    microphone. Interior pauses stay: they are part of how long the line took to say. */
export function speechSpan(energy, dropDb = SPEECH_DROP_DB) {
  if (!energy || !energy.length) return [0, 0];
  let peak = 0;
  for (let t = 0; t < energy.length; t += 1) if (energy[t] > peak) peak = energy[t];
  const floor = 10 * Math.log10(peak + 1e-10) - dropDb;
  let start = -1;
  let end = 0;
  for (let t = 0; t < energy.length; t += 1) {
    if (10 * Math.log10(energy[t] + 1e-10) >= floor) {
      if (start < 0) start = t;
      end = t + 1;
    }
  }
  return start < 0 ? [0, energy.length] : [start, end];
}

/** How many *output* frames of the recording are speech.

    The student strides its 100fps mel by 2, so the prior counts in output frames and
    this halves to match. Deliberately not an endpointer for acceptance — the energy
    gate tried for that was dropped for overfitting 25 clips. This only decides what
    the prior is told the length of the audio was. */
export function speechFrames(energy, dropDb = SPEECH_DROP_DB) {
  const [start, end] = speechSpan(energy, dropDb);
  return Math.max(0, (end >> 1) - (start >> 1));
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
/** Ids *and* the readable tokens, mirroring `scripts/gop_encode.py`.

    `encodeTarget` returns ids, which is all the ranking needs. Saying *which* sound was
    wrong needs the grapheme to survive the encoding rather than be guessed back out of the
    vocabulary afterwards. Deliberately the same mapping as `encodeTarget` — including
    `space` being a vocabulary key — because a second, subtly different encoder would put
    the two scores on different token sequences and nothing would say so. */
export function encodeTargetTokens(flat, tokToId, space = '▁') {
  const ids = [];
  const toks = [];
  for (const ch of flat) {
    const tok = ch === ' ' ? space : ch;
    if (tokToId.has(tok)) {
      ids.push(tokToId.get(tok));
      toks.push(ch === ' ' ? '␣' : ch);
    }
  }
  return { ids, toks };
}

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

/* ── Which sound was wrong ─────────────────────────────────────────────────── */
/* Port of `scripts/gop.py`. Goodness of Pronunciation, per target token, from the
   alignment CTC already implies rather than from a forced one — a learner's timing is
   exactly what a forced alignment gets wrong, and a soft occupancy does not have to
   choose. See the module note there for the formula.

   These constants are duplicated from `scripts/gop.py` on purpose, the same way the
   duration constants are: the frontend cannot import Python, and a test pins the two
   together so they cannot drift apart silently. */

/* Graphemes whose score says nothing about the learner. Measured on 75 recordings that
   are *correct*: `q` because the model cannot hear a glottal stop at all, and `g`/`h`/`'`
   because `għ` is silent, so there is no sound to find and the model is right to fail.
   Charging a learner for these is charging them for the model's blind spots. */
export const GOP_IGNORE = new Set(["'", 'd', 'g', 'h', 'j', 'q', 'r', 'ż']);

/* Below this, the sounds are not the ones the line asks for. Set where it holds 93% of
   the learner's own recordings while refusing every time-reversed clip — the negative
   the confidence floor is worst at, admitting 67% of them. */
export const GOP_MIN = -2.29;

/** Per-token frame occupancy, `(L × frames)` row-major. Null when the target cannot be
    aligned at all. */
export function occupancy(logprobs, frames, vocab, ids, blank) {
  const L = ids.length;
  if (!L || L > frames) return null;
  const size = 2 * L + 1;
  const ext = new Int32Array(size).fill(blank);
  for (let i = 0; i < L; i += 1) ext[2 * i + 1] = ids[i];
  const skip = new Uint8Array(size);
  for (let s = 2; s < size; s += 1) {
    skip[s] = ext[s] !== blank && ext[s] !== ext[s - 2] ? 1 : 0;
  }

  // Both matrices in full rather than a rolling row: the occupancy needs alpha and beta
  // at the same frame, which a rolling forward pass has already thrown away.
  const alpha = new Float64Array(frames * size).fill(NEG);
  alpha[0] = logprobs[ext[0]];
  if (size > 1) alpha[1] = logprobs[ext[1]];
  for (let t = 1; t < frames; t += 1) {
    const row = t * size, prev = row - size, base = t * vocab;
    for (let s = 0; s < size; s += 1) {
      let acc = alpha[prev + s];
      if (s >= 1) acc = logaddexp(acc, alpha[prev + s - 1]);
      if (s >= 2 && skip[s]) acc = logaddexp(acc, alpha[prev + s - 2]);
      alpha[row + s] = acc + logprobs[base + ext[s]];
    }
  }

  const beta = new Float64Array(frames * size).fill(NEG);
  const last = (frames - 1) * size;
  beta[last + size - 1] = 0;
  if (size > 1) beta[last + size - 2] = 0;
  for (let t = frames - 2; t >= 0; t -= 1) {
    const row = t * size, nxt = row + size, base = (t + 1) * vocab;
    for (let s = 0; s < size; s += 1) {
      let acc = beta[nxt + s] + logprobs[base + ext[s]];
      if (s + 1 < size) {
        acc = logaddexp(acc, beta[nxt + s + 1] + logprobs[base + ext[s + 1]]);
      }
      if (s + 2 < size && skip[s + 2]) {
        acc = logaddexp(acc, beta[nxt + s + 2] + logprobs[base + ext[s + 2]]);
      }
      beta[row + s] = acc;
    }
  }

  const total = size > 1
    ? logaddexp(alpha[last + size - 1], alpha[last + size - 2])
    : alpha[last + size - 1];
  const out = new Float64Array(L * frames);
  if (!(total > NEG) || !Number.isFinite(total)) return out;
  for (let i = 0; i < L; i += 1) {
    const s = 2 * i + 1;
    for (let t = 0; t < frames; t += 1) {
      const v = alpha[t * size + s] + beta[t * size + s] - total;
      out[i * frames + t] = Math.exp(Math.min(0, Math.max(-700, v)));
    }
  }
  return out;
}

/** GOP per target token: the occupancy-weighted margin against the model's own choice.
    Zero means "wherever this token sits, it was also what the model wanted". */
export function tokenGop(logprobs, frames, vocab, ids, blank) {
  const gam = occupancy(logprobs, frames, vocab, ids, blank);
  const out = new Float64Array(ids.length);
  if (!gam) return out;
  const best = new Float64Array(frames);
  let bestMax = -Infinity;
  for (let t = 0; t < frames; t += 1) {
    let m = -Infinity;
    const base = t * vocab;
    for (let v = 0; v < vocab; v += 1) {
      const x = logprobs[base + v];
      if (x > m) m = x;
    }
    best[t] = m;
    if (m > bestMax) bestMax = m;
  }
  for (let i = 0; i < ids.length; i += 1) {
    const tok = ids[i];
    let mass = 0, acc = 0, tokMax = -Infinity;
    for (let t = 0; t < frames; t += 1) {
      const w = gam[i * frames + t];
      const x = logprobs[t * vocab + tok];
      mass += w;
      acc += w * (x - best[t]);
      if (x > tokMax) tokMax = x;
    }
    // Never aligned anywhere: score it on its best frame instead of dividing by zero.
    out[i] = mass <= 1e-9 ? tokMax - bestMax : acc / mass;
  }
  return out;
}

/** One number for the attempt: the mean GOP over the tokens the model can be trusted on.
    NaN when the line is made entirely of tokens in `GOP_IGNORE`, which is not a verdict —
    callers must treat it as "no opinion" rather than as a failure. */
export function gopScore(logprobs, frames, vocab, ids, toks, blank) {
  const g = tokenGop(logprobs, frames, vocab, ids, blank);
  let sum = 0, n = 0;
  for (let i = 0; i < g.length; i += 1) {
    if (toks && GOP_IGNORE.has(toks[i])) continue;
    sum += g[i];
    n += 1;
  }
  return n ? sum / n : NaN;
}

/** The worst-scoring token the learner can be told about, or null when there is nothing
    worth saying. Skips the model's blind spots for the same reason the score does. */
export function worstSound(logprobs, frames, vocab, ids, toks, blank) {
  const g = tokenGop(logprobs, frames, vocab, ids, blank);
  let worst = null, at = -1;
  for (let i = 0; i < g.length; i += 1) {
    if (toks && GOP_IGNORE.has(toks[i])) continue;
    if (worst === null || g[i] < worst) { worst = g[i]; at = i; }
  }
  return worst === null ? null : { token: toks ? toks[at] : null, index: at, gop: worst };
}

/* ── How long should that have taken? ─────────────────────────────────────── */
/* Port of `constrained_ctc.duration_prior` / `rank_score`.

   `targetConfidence` divides by frames, which stops an unnormalised total ranking
   `Bonġu` above every sentence in the deck. It does not stop the other direction, and
   the other direction is where a learner loses: a short sequence has fewer obligatory
   emissions and more freedom about where to put them, so it can explain a long
   utterance respectably by ignoring most of it.

   On 25 recordings of a learner's voice, every one of the five that lost its rank lost
   it to the same line — `Bonġu!`, five tokens, the shortest thing in the field. Not one
   was a confusion; all five were the same artefact.

   So charge a hypothesis for claiming a length the audio cannot support. Fitted on the
   29,860 TTS passes of the distillation corpus, where the line is known and was actually
   synthesised: frames ≈ 28.28 + 1.8794 × tokens, sd 13.27, at the student's 50fps. That
   is 38ms a character, which is what speech does. */
const DUR_INTERCEPT = 28.28;
const DUR_SLOPE = 1.8794;
const DUR_SD = 13.27;

/* Swept on the learner's recordings against the deployed field of 24 lines drawn from
   the 377 the script accepts, and checked against 25 synthetic clips it must not damage
   and 90 negatives it must not admit. 0.1 is the peak for three independently-trained
   students, including one that ranks 29% without it — a property of the method, not of
   one model or of 25 clips. See the table in `constrained_ctc.py`. */
const DUR_WEIGHT = 0.1;

/* Slope of the spread against hypothesis length. 0 is the single constant above, which
   is what ships.

   `scripts/fit_duration.py` re-measured all of this against 1,335 cached renders and
   found the residual is nothing like homoscedastic — binned by length its sd runs 4.6
   frames at 7 tokens, 13.2 at 15, 37.2 at 26 — so one constant is provably the wrong
   shape. It is still what ships, because the constants and `DUR_WEIGHT` are one joint
   fit and neither survives moving alone. What ranking consumes is not the fit but the
   prior's *differential* between two hypothesis lengths on fixed audio, and on those
   same clips the charge laid on a five-token rival over the truth — the `Bonġu!`
   artefact all of this exists to kill — is +0.834 in confidence units as deployed,
   against the +0.155 needed to reverse the five documented failures. Refitting the mean
   and keeping one sd drops it to +0.115 and puts the bug back; refitting with a sloped
   sd raises it to +2.6, which is λ = 0.3 wearing a different hat, and λ = 0.3 measured
   61% learner accept and 32% synthetic. Both are regressions, so both wait for a joint
   sweep against the learner clips and the negatives. */
const DUR_SD_SLOPE = 0;

/** Spread of the length a hypothesis of this many tokens implies. */
export function durationSd(tokens) {
  return Math.max(1e-6, DUR_SD + DUR_SD_SLOPE * tokens);
}

/** How surprising this hypothesis's length is for this much audio. Always ≤ 0. */
export function durationPrior(tokens, frames) {
  const expected = DUR_INTERCEPT + DUR_SLOPE * tokens;
  const z = (frames - expected) / durationSd(tokens);
  return -0.5 * z * z;
}

/** What to *compare* hypotheses on: the acoustic fit plus what their length costs.

    Kept separate from `targetConfidence` on purpose. "Which of these lines is it" is a
    comparison and the prior belongs in it; "is there anything here at all" is a floor,
    and a floor that moved with the length of whichever line was asked for would not be
    a floor. `app.js` uses the first for ranking and the second for `MIN_CONFIDENCE`.

    `speech` overrides what counts as the length of the audio — pass `speechFrames` to
    charge the hypothesis against the speech alone. Left out, every frame counts, which
    is what the deployed constants were fitted against. */
export function rankScore(logprobs, frames, vocab, ids, blank, speech) {
  return targetConfidence(logprobs, frames, vocab, ids, blank)
    + DUR_WEIGHT * durationPrior(ids.length, speech === undefined ? frames : speech);
}

/** How spread out the field's scores are, as a sample standard deviation.

    Reported because a margin is a distance, and a distance means nothing without a
    scale: 0.006 of daylight is noise in a field spanning 0.4 and decisive in one
    spanning 0.01 — and the scale moves with the speaker exactly as the absolute
    confidence does, which is the thing ranking was introduced to escape.

    Taken per utterance rather than accumulated across turns, so it needs no history and
    has no cold start. The alternatives were all scored on this recording, which is
    precisely the scale wanted. Two passes rather than the sum-of-squares shortcut: the
    scores cluster tightly, and there `Σx² - (Σx)²/n` is a subtraction of two nearly
    equal numbers and can come out negative. `app.js` decides what to do with the
    result — see `MARGIN_SIGMAS`. */
export function spread(values) {
  if (!values || values.length < 2) return 0;
  let sum = 0;
  for (const v of values) sum += v;
  const mean = sum / values.length;
  let sq = 0;
  for (const v of values) sq += (v - mean) * (v - mean);
  return Math.sqrt(sq / (values.length - 1));
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
  const { data, nMels, frames, energy } = features(audio, parts.mel, parts.window);
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
    /* Two numbers, two jobs. `confidence` is the acoustic fit alone and is what the
       floor in app.js looks at. `rank` adds what the hypothesis's length costs and is
       what the field is compared on — see `rankScore`. */
    const acoustic = (line) => {
      const seq = encodeTarget(line, parts.tokToId);
      return seq.length ? targetConfidence(d, outFrames, vocab, seq, parts.blankId) : 0;
    };
    /* What the prior is told the audio's length was. `DUR_FRAMES` is 'total' — every
       frame — because that is the definition the constants were fitted against; feeding
       it the speech span without refitting them charges a five-token rival correctly on
       only 37% of clips against the deployed 91%, which is the whole trap. */
    const spoken = DUR_FRAMES === 'speech' ? speechFrames(energy) : outFrames;
    const ranked = (line) => {
      const seq = encodeTarget(line, parts.tokToId);
      return seq.length
        ? rankScore(d, outFrames, vocab, seq, parts.blankId, spoken) : -Infinity;
    };
    result.confidence = acoustic(target);
    result.rank = ranked(target);

    /* Every alternative is scored against the same audio and the same denominator, so
       only the ordering is being trusted — not the absolute value, which is the part that
       does not survive a change of speaker. */
    let best = -Infinity;
    let bestLine = '';
    const field = [];
    for (const line of distractors) {
      if (!line || line === target) continue;
      const c = ranked(line);
      if (c > best) { best = c; bestLine = line; }
      if (Number.isFinite(c)) field.push(c);
    }
    /* Per-sound score, for the verdict between "correct" and "wrong". The floor and the
       field both answer "is this the line"; this one answers "are these the sounds", which
       is the question that separates a learner who nearly said it from audio that is not
       the line at all. The floor cannot: it admits 67% of time-reversed speech. */
    const { ids: gopIds, toks: gopToks } = encodeTargetTokens(target, parts.tokToId);
    result.gop = gopIds.length
      ? gopScore(d, outFrames, vocab, gopIds, gopToks, parts.blankId) : NaN;
    result.worstSound = gopIds.length
      ? worstSound(d, outFrames, vocab, gopIds, gopToks, parts.blankId) : null;

    result.runnerUp = Number.isFinite(best) ? best : null;
    result.runnerUpLine = bestLine;
    result.wins = result.runnerUp === null || result.rank > result.runnerUp;
    result.fieldSd = spread(field);
  }
  return result;
}

/* Maltese text comparison and phonetic keying, ported from backend/text.py and
   backend/phonetics.py so the app can grade without a server.

   This is a translation, not a reimplementation. Every rule, every threshold and
   the order they are applied in mirror the Python, because the Python is what the
   334 recorded transcripts in tests/fixtures were measured against.
   `tests/test_text_parity.py` runs both over the same strings and compares, for
   the same reason the FSRS port has a parity test: two implementations of one
   algorithm drift silently, and the symptom here would be a learner marked wrong
   for a sentence they said correctly.

   Maltese orthography is close to phonemic, so a small set of rewrites absorbs
   almost all of the variation a recogniser introduces:

       ie → long i          għ → silent
       x  → ʃ               ħ, h → h (h is silent word-finally)
       ż  → z               z → ts
       ċ  → tʃ              ġ → dʒ, j → j (as in yes)
       q  → glottal stop, dropped in the soft key */

/* ── Normalisation and folding ───────────────────────────────────────────── */

const FOLD = {
  'ġ': 'g', 'Ġ': 'g', 'ħ': 'h', 'Ħ': 'h',
  'ż': 'z', 'Ż': 'z', 'ċ': 'c', 'Ċ': 'c',
  'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
  '’': "'", '‘': "'", '`': "'", '´': "'",
};

// Python's `[^\w\s'\-]` with re.UNICODE keeps letters, digits and underscore.
const PUNCT = /[^\p{L}\p{N}_\s'-]/gu;

export function normalise(s) {
  s = (s || '').normalize('NFC').trim().replace(/[’‘]/g, "'");
  return s.replace(/\s+/g, ' ');
}

export function fold(s) {
  s = normalise(s).toLowerCase().replace(/[ġĠħĦżŻċĊàèìòù’‘`´]/g, (c) => FOLD[c] ?? c);
  s = s.replaceAll('gh', '');            // silent għ, and its de-diacriticised form
  s = s.replace(PUNCT, ' ');
  s = s.replaceAll('-', ' ').replaceAll("'", '');
  // The article is frequently dropped or misheard; treat it as optional.
  s = s.replace(/\b(il|l|ic|id|in|ir|is|it|ix|iz)\b\s*/g, '');
  return s.replace(/\s+/g, ' ').trim();
}

export const tokens = (s) => normalise(s).split(' ').filter(Boolean);

/** Split into [display, comparison-key] pairs, breaking at hyphens.

    Maltese writes the article and fused prepositions onto the next word
    (`mill-Awstralja`, `id-dar`) and recognisers split them about half the time.
    Splitting here makes both spellings tokenise identically. */
export function units(s) {
  const out = [];
  for (const word of tokens(s)) {
    const parts = word.split('-');
    parts.forEach((part, i) => {
      if (!part) return;
      out.push([part + (i < parts.length - 1 ? '-' : ''), fold(part).replaceAll(' ', '')]);
    });
  }
  return out;
}

/* ── difflib.SequenceMatcher.ratio(), faithfully ─────────────────────────────

   Not edit distance. Python's ratio is 2*M/T where M is the total size of the
   matching blocks found by its recursive longest-match algorithm — which is not
   the same number an LCS or a Levenshtein ratio gives, and the thresholds in this
   app (0.86 to accept, 0.8 inside word matching) were tuned against *this*
   function. Substituting a different similarity would silently move every
   decision boundary. */

function matchingBlocks(a, b) {
  const b2j = new Map();
  for (let i = 0; i < b.length; i += 1) {
    const ch = b[i];
    if (!b2j.has(ch)) b2j.set(ch, []);
    b2j.get(ch).push(i);
  }

  function longestMatch(alo, ahi, blo, bhi) {
    let besti = alo, bestj = blo, bestsize = 0;
    let j2len = new Map();
    for (let i = alo; i < ahi; i += 1) {
      const newj2len = new Map();
      for (const j of b2j.get(a[i]) || []) {
        if (j < blo) continue;
        if (j >= bhi) break;
        const k = (j2len.get(j - 1) || 0) + 1;
        newj2len.set(j, k);
        if (k > bestsize) { besti = i - k + 1; bestj = j - k + 1; bestsize = k; }
      }
      j2len = newj2len;
    }
    return [besti, bestj, bestsize];
  }

  const queue = [[0, a.length, 0, b.length]];
  const blocks = [];
  while (queue.length) {
    const [alo, ahi, blo, bhi] = queue.pop();
    const [i, j, k] = longestMatch(alo, ahi, blo, bhi);
    if (!k) continue;
    blocks.push([i, j, k]);
    if (alo < i && blo < j) queue.push([alo, i, blo, j]);
    if (i + k < ahi && j + k < bhi) queue.push([i + k, ahi, j + k, bhi]);
  }
  return blocks;
}

export function ratio(a, b) {
  if (!a.length && !b.length) return 1.0;
  const matches = matchingBlocks(a, b).reduce((n, [, , k]) => n + k, 0);
  const total = a.length + b.length;
  return total ? (2.0 * matches) / total : 1.0;
}

/* ── Scoring ─────────────────────────────────────────────────────────────── */

export function similarity(a, b) {
  const fa = fold(a);
  const fb = fold(b);
  if (!fa && !fb) return 1.0;
  if (!fa || !fb) return 0.0;
  return ratio(fa, fb);
}

/** Token-level F1 on folded words — more forgiving of word order than raw ratio. */
export function wordSimilarity(a, b) {
  const ta = units(a).map(([, k]) => k).filter(Boolean);
  const tb = units(b).map(([, k]) => k).filter(Boolean);
  if (!ta.length || !tb.length) return 0.0;
  const pool = [...tb];
  let overlap = 0;
  for (const t of ta) {
    const idx = pool.findIndex((x) => ratio(t, x) >= 0.8);
    if (idx !== -1) { pool.splice(idx, 1); overlap += 1; }
  }
  const precision = overlap / ta.length;
  const recall = overlap / tb.length;
  if (precision + recall === 0) return 0.0;
  return (2 * precision * recall) / (precision + recall);
}

/* `Math.round(x * 10000) / 10000` is the obvious way to round to four places and
   it disagrees with Python: multiplying by 10000 rounds first, so a value just
   below 0.66875 becomes exactly 6687.5 and then rounds up, where Python — working
   from the exact double — correctly gives 0.6687. The parity test caught it on
   `qagħad id-dar`. `toFixed` rounds the exact value, as Python's round() does. */
const round4 = (x) => Number(x.toFixed(4));

export function score(said, target) {
  return round4(0.45 * similarity(said, target) + 0.55 * wordSimilarity(said, target));
}

/** Word-level diff for the correction card. `del` = extra, `ins` = omitted. */
export function diffWords(said, target) {
  const ua = units(said);
  const ub = units(target);
  const a = ua.map(([d]) => d);
  const fa = ua.map(([, k]) => k);
  const b = ub.map(([d]) => d);
  const fb = ub.map(([, k]) => k);
  const out = [];
  for (const [op, i1, i2, j1, j2] of opcodes(fa, fb)) {
    if (op === 'equal') {
      for (let k = 0; k < i2 - i1; k += 1) {
        out.push({ op: 'equal', said: a[i1 + k], target: b[j1 + k] });
      }
    } else if (op === 'replace') {
      const span = Math.max(i2 - i1, j2 - j1);
      for (let k = 0; k < span; k += 1) {
        out.push({ op: 'sub',
          said: i1 + k < i2 ? a[i1 + k] : '',
          target: j1 + k < j2 ? b[j1 + k] : '' });
      }
    } else if (op === 'delete') {
      for (let k = i1; k < i2; k += 1) out.push({ op: 'del', said: a[k], target: '' });
    } else {
      for (let k = j1; k < j2; k += 1) out.push({ op: 'ins', said: '', target: b[k] });
    }
  }
  return out;
}

function opcodes(a, b) {
  const blocks = matchingBlocks(a, b).sort((x, y) => x[0] - y[0] || x[1] - y[1]);
  blocks.push([a.length, b.length, 0]);
  const out = [];
  let i = 0;
  let j = 0;
  for (const [ai, bj, size] of blocks) {
    if (i < ai && j < bj) out.push(['replace', i, ai, j, bj]);
    else if (i < ai) out.push(['delete', i, ai, j, bj]);
    else if (j < bj) out.push(['insert', i, ai, j, bj]);
    if (size) out.push(['equal', ai, ai + size, bj, bj + size]);
    i = ai + size;
    j = bj + size;
  }
  return out;
}

/* ── Phonetic key ────────────────────────────────────────────────────────── */

// Order matters: digraphs are consumed before the single letters inside them.
const RULES = [
  ['għ', ''], ['gh', ''],
  ['ie', 'I'],
  ['ċ', 'C'], ['ch', 'C'], ['c', 'C'],
  ['ġ', 'J'], ['g', 'J'],
  ['x', 'S'],
  ['ż', 'z'], ['ts', 'z'],
  ['q', 'Q'],
  ['ħ', 'h'],
  ['j', 'y'],
];

export function phoneticKey(s) {
  let t = (s || '').trim().toLowerCase().normalize('NFC')
    .replace(/[’`]/g, "'")
    .replace(/[àèìòù]/g, (c) => FOLD[c]);
  for (const [src, dst] of RULES) t = t.replaceAll(src, dst);
  t = t.replaceAll("'", '');
  t = t.replace(/h\b/g, '');                 // h is silent word-finally
  t = t.replace(/[^a-zA-Z\s]/g, ' ').toLowerCase();
  t = t.replace(/(.)\1+/g, '$1');            // collapse doubles
  return t.replace(/\s+/g, ' ').trim();
}

export const keyNospace = (s) => phoneticKey(s).replaceAll(' ', '');

/** `q` is dropped: recognisers write the glottal stop as għ, as nothing, or as
    whatever consonant follows — never as a consonant of its own. */
export const softKey = (s) => keyNospace(s).replaceAll('q', '');

export function phoneticSimilarity(a, b, soft = false) {
  const ka = soft ? softKey(a) : keyNospace(a);
  const kb = soft ? softKey(b) : keyNospace(b);
  if (!ka && !kb) return 1.0;
  if (!ka || !kb) return 0.0;
  return ratio(ka, kb);
}

/* ── Grading ─────────────────────────────────────────────────────────────── */

export const AGAIN = 1, HARD = 2, GOOD = 3, EASY = 4;

/** Mirrors backend `_assess`: a phonetic floor under the orthographic score, so
    the recogniser's word boundaries do not cost the learner marks. */
export function assess(said, target) {
  const phon = phoneticSimilarity(said, target, true);
  const s = round4(Math.max(phon, 0.6 * phon + 0.4 * score(said, target)));
  return {
    said: normalise(said),
    target: normalise(target),
    score: s,
    grade: s >= 0.95 ? EASY : s >= 0.78 ? GOOD : s >= 0.55 ? HARD : AGAIN,
    verdict: s >= 0.95 ? 'perfect' : s >= 0.75 ? 'close' : s >= 0.5 ? 'partial' : 'off',
    diff: diffWords(said, target),
  };
}

/* Scripted conversation, ported from backend/dialogue.py.

   Matching is a few microseconds of string work, so there was never a reason for
   it to be a network round trip — it was on the server only because the Maltese
   rules were. With those ported (text.js), the whole turn happens in the page.

   Same constants, same order of operations, same backstops as the Python.
   tests/test_dialogue_parity.py replays all 334 recorded transcripts through both
   and compares verdict, score and which line was matched. */

import * as text from './text.js';

export const CORRECT = 0.86;
export const CLOSE = 0.62;
export const MAX_ATTEMPTS = 2;

let doc = { dialogues: [] };

export function load(data) { doc = data || { dialogues: [] }; }
export const all = () => doc.dialogues || [];
export const get = (id) => all().find((d) => d.id === id) || null;
export const node = (did, nid) => get(did)?.nodes?.[nid] || null;

export function present(did, nid) {
  const n = node(did, nid);
  if (!n) return null;
  return {
    dialogue: did,
    node: nid,
    say_mt: n.say_mt,
    say_en: n.say_en,
    expect_en: n.expect_en || '',
    options: (n.accept || []).filter((a) => !a.open).map((a) => a.en),
  };
}

export const start = (did) => (get(did) ? present(did, get(did).start) : null);

/* A quoted slot, and only that. Maltese writes the apostrophe as a letter —
   `ta' Marija`, `x'inhu` — so a naive quote pair would span from one to the next
   and blank out the words between. */
const QUOTED = /(?<!\w)['‘’"]([^'‘’"]+)['‘’"](?!\w)/u;

/** Score a sentence with a quoted foreign word on its Maltese frame only.

    `Kif tgħid 'cheese' bil-Malti?` asks the learner for the one word a Maltese
    recogniser has never been trained on. Grading the whole sentence punishes them
    for supplying it. */
function frameScore(said, target) {
  const frame = target.replace(QUOTED, ' ');
  const want = text.normalise(frame).split(' ').map(text.softKey).filter(Boolean);
  const got = text.normalise(said).split(' ').map(text.softKey);
  if (!want.length) return 0;
  let hits = 0;
  let i = 0;
  for (const w of want) {
    while (i < got.length) {
      if (text.phoneticSimilarity(got[i], w) >= 0.8) { hits += 1; i += 1; break; }
      i += 1;
    }
  }
  return hits / want.length;
}

/* The variable slot in a frame: `Jien …`, `Għandi … sena`. The three-dot spelling
   is accepted too, because that is what a keyboard produces. */
const SLOT = /…|\.\.\./;

/** Phonetic keys, one per word, with the fused article split off its noun.
    `mill-Awstralja` is one word to split() and two to a recogniser. */
function keys(s) {
  return text.normalise(s).split(/[\s-]+/).map(text.softKey).filter(Boolean);
}

/* Keys that are one word in Maltese, where nothing general can tell: `hu`/`huwa` and
   `hi`/`hija` are the pronoun long and short (two letters is too few for a ratio, and
   a rule about added letters would swallow `huma`, meaning *they*); and `minn` fused
   with the assimilating article — mill-, mir-, mit-, mid-, mis-, miċ-, miż-, mix- —
   one preposition spelled nine ways. */
const ONE_WORD = [
  new Set(['hu', 'huwa']),
  new Set(['hi', 'hiya']),
  new Set(['min', 'mil', 'mir', 'mit', 'mid', 'mis', 'mic', 'miz']),
];

/** Is this word of the answer the frame's word? Keys, not spellings.

    Tolerant, because the recogniser is inventive — but not about a first letter that
    has been *swapped*. Maltese conjugates on it: `nibda` I start, `tibda` you start,
    `jibda` he starts, 0.80 apart and inside the bar. Answering `X'ħin tibda?` with
    `Tibda fid-disgħa` reads the question back rather than answering it, and
    `m'għandix` is not `għandi` either. A first letter added or dropped is the
    recogniser — `nħobb` for `inħobb`, `isimni` for `jisimni` — and below three
    letters even that is guesswork, so short keys match outright or are listed above.

    `text.ratio`, not `phoneticSimilarity`: these are keys already, and keying a key
    collapses the doubled vowel silent għ leaves behind (`noqgħod` → `nood` → `nod`). */
function sameWord(said, want) {
  if (said === want) return true;
  if (ONE_WORD.some((group) => group.has(said) && group.has(want))) return true;
  if (Math.min(said.length, want.length) < 3) return false;
  if (said.slice(0, 1) !== want.slice(0, 1)) return said.slice(1) === want || want.slice(1) === said;
  return text.ratio(said, want) >= 0.8;
}

/** Does the answer *open with* — and *close with* — the frame around its slot?

    An open question is a fixed Maltese frame with one variable in it, and the frame
    is what the scene teaches. `Jien …` wants `Jien` before the name; `Għandi … sena`
    wants `Għandi` before the age and `sena` after it. The words before the slot are
    looked for in order from the start, the words after it in order from the end
    backwards, and what is left between the two runs is the slot — never judged.

    Both runs may step over words the frame does not mention: `Iva, għandi ħuti`,
    `Jien inħobb il-ħut`, `Tliet kmamar żgħar`. What they cannot do is change places
    with the slot — `Pietru jien` has the keyword and nothing after it, and scores
    half. The slot counts as one more thing to supply: `Jien Pietru` scores 1.0,
    `Jien` alone 0.5, an answer with none of the frame in it 0. */
function anchorScore(said, frame) {
  const parts = frame.split(SLOT);
  const slot = parts.length > 1;
  const wantPre = keys(parts[0]);
  const wantPost = slot ? keys(parts.slice(1).join(' ')) : [];
  const got = keys(said);

  const total = wantPre.length + wantPost.length + (slot ? 1 : 0);
  if (!total) return 0;

  // Forwards from the start for what comes before the slot…
  let i = 0;
  let preHits = 0;
  for (const w of wantPre) {
    let at = -1;
    for (let k = i; k < got.length; k += 1) if (sameWord(got[k], w)) { at = k; break; }
    if (at < 0) break;
    i = at + 1;
    preHits += 1;
  }
  // …and backwards from the end for what comes after it, never crossing into the
  // words the opening run already claimed.
  let j = got.length - 1;
  let postHits = 0;
  for (let n = wantPost.length - 1; n >= 0; n -= 1) {
    let at = -1;
    for (let k = j; k >= i; k -= 1) if (sameWord(got[k], wantPost[n])) { at = k; break; }
    if (at < 0) break;
    j = at - 1;
    postHits += 1;
  }

  let hits = preHits + postHits;
  // Whatever is left between the two runs is the answer to the question. It counts
  // only once some of the frame is there, or every stray word would look like a
  // filled slot and "hello" would score half marks for a name.
  if (slot && hits && j >= i) hits += 1;
  return hits / total;
}

function bestAnchor(said, frames) {
  const best = (frames || []).reduce((b, f) => Math.max(b, anchorScore(said, f)), 0);
  return Math.round(best * 10000) / 10000;
}

/** Is this listed answer a deliberate step outside the frame?

    Most accepted answers on an open question are the frame with an example in the
    slot — `Għandi tletin sena.` A few use none of it: `Dak sigriet!` for an age,
    `Ma niftakarx!` for a name. Saying one of those is a real answer and keeps its
    ordinary match score instead of being marked down against a frame it never used.

    None of it, not some of it: an answer that half-uses the frame is graded on the
    frame like any other, which keeps a sloppy frame visible. */
function outsideFrames(candidate, frames) {
  return !!candidate && bestAnchor(candidate.mt, frames) === 0;
}

/** How much of an accepted answer the learner produced, in order.

    The fallback for the handful of free nodes with no slot in them — "is your family
    big or small", "how do you feel" — where the accepted answers are whole sentences
    rather than a frame with a name in it, so there is nothing to anchor. */
function frameRecall(said, target) {
  const want = text.normalise(target).split(' ').map(text.softKey).filter(Boolean);
  const got = text.normalise(said).split(' ').map(text.softKey);
  if (!want.length) return 0;
  let hits = 0;
  let i = 0;
  for (const w of want) {
    while (i < got.length) {
      if (text.phoneticSimilarity(got[i], w) >= 0.8) { hits += 1; i += 1; break; }
      i += 1;
    }
  }
  return hits / want.length;
}

function bestFrame(said, accepted) {
  let best = null;
  let bestScore = 0;
  for (const candidate of accepted || []) {
    if (candidate.open) continue;
    const s = frameRecall(said, candidate.mt);
    if (s > bestScore) { best = candidate; bestScore = s; }
  }
  return [best, Math.round(bestScore * 10000) / 10000];
}

export function bestMatch(said, accepted) {
  let best = null;
  let bestScore = 0;
  for (const candidate of accepted || []) {
    if (candidate.open) continue;
    const phon = text.phoneticSimilarity(said, candidate.mt, true);
    let s = Math.max(phon, 0.6 * phon + 0.4 * text.score(said, candidate.mt));
    if (QUOTED.test(candidate.mt)) s = Math.max(s, frameScore(said, candidate.mt));
    if (s > bestScore) { best = candidate; bestScore = s; }
  }
  return [best, Math.round(bestScore * 10000) / 10000];
}

export function evaluate(did, nid, said, attempts = 0) {
  const n = node(did, nid);
  if (!n) return { error: 'unknown node' };

  said = text.normalise(said);
  const [match0, score0] = bestMatch(said, n.accept);

  let verdict;
  let frameScored = false;
  let [match, score] = [match0, score0];
  if (n.free) {
    // A name, a place, a number: never marked wrong, because the app cannot know
    // it. But the frame around the slot is ordinary Maltese, so that part is
    // scored and reported — "Jien Pietru" is not the same as "hello".
    const [framed, recall] = bestFrame(said, n.accept);
    const frames = n.frames || [];
    if (frames.length) {
      // The node says where the slot is, so the frame is looked for where it
      // belongs: `Jien` at the start, `sena` at the end, the town or the age in
      // between and unjudged. Nothing else counts as the frame — a name on its own
      // scores 0 here even when it sounds like the example answer's name, because
      // the sentence the scene teaches was not said.
      const anchor = bestAnchor(said, frames);
      // Unless what they said is one of the deliberate steps outside the frame.
      // `Dak sigriet!` is a real answer to how old you are, and how near they came
      // to *it* is a real score — better than the 0 its frame would give. Near it,
      // though: below the bar the rest of the app calls "almost", the answer is
      // neither the frame nor the sentence, and 31% of a line nobody was aiming at
      // is not feedback.
      if (!(outsideFrames(match, frames) && score > anchor && score >= CLOSE)) {
        // The score is the frame's, so the correction card is the nearest example
        // answer; `framed` is null when nothing overlapped at all, and then the
        // ordinary match is still the best line to show. On the escape path both
        // stay as they are, or the card would name a line the score never came from.
        score = anchor;
        frameScored = true;
        match = framed || match;
      }
    } else if (recall > score) {
      // No slot to anchor on, so this is how much of an example answer they made —
      // not a frame score, and not claimed as one.
      match = framed || match;
      score = recall;
    }
    verdict = text.fold(said).length >= 2 ? 'correct' : 'wrong';
  } else if (score >= CORRECT) verdict = 'correct';
  else if (score >= CLOSE) verdict = 'close';
  else verdict = 'wrong';

  // Never let someone loop on one line: after a couple of tries the target has
  // been shown and spoken twice, and being stuck is worse than being waved on.
  let movedOn = false;
  if (verdict !== 'correct' && attempts >= MAX_ATTEMPTS) { verdict = 'correct'; movedOn = true; }

  let reply = n[verdict] || n.wrong || {};
  if (movedOn) reply = { mt: 'Ejja nkomplu.', en: "Let's carry on." };
  const nextNode = verdict === 'correct' ? n.next : nid;

  const out = {
    verdict,
    // A name, a town, an age: accepted as given, whatever it is.
    free: !!n.free,
    // …and whether the score is the frame around that slot rather than a match
    // against a listed answer. The UI says so out loud, because "100%" means two
    // different things: the frame was right, or the whole sentence was.
    frame_scored: frameScored,
    moved_on: movedOn,
    score,
    said,
    matched_mt: match ? match.mt : null,
    matched_en: match ? match.en : null,
    reply_mt: reply.mt || '',
    reply_en: reply.en || '',
    advance: verdict === 'correct',
    dialogue: did,
    node: nid,
  };

  if (verdict !== 'correct' && match) {
    out.say_this_mt = match.mt;
    out.say_this_en = match.en;
    out.diff = text.diffWords(said, match.mt);
  }

  if (nextNode && verdict === 'correct') out.next = present(did, nextNode);
  else if (verdict === 'correct') { out.next = null; out.finished = true; }
  return out;
}

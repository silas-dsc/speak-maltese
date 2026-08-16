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

/* `Iva, għandi ħuti`, `Le, jien turist`, `Bonġu, jien ħabib ta' Marija`: a yes, a no,
   a greeting or a hesitation in front of the frame is still the frame — the accepted
   answers open that way and so does the scene that prompts them, `Bonġu, min qed
   jitkellem?`. Anything else in first place is not the frame. */
const OPENERS = ['iva', 'le', 'mela', 'allura', 'ehm', 'emm', 'mm', 'ok',
  'bonġu', 'bonswa', 'skużani', 'grazzi', 'merħba'].map(text.softKey);

/** Does the answer *start with* — and *end with* — the frame around its slot?

    An open question is a fixed Maltese frame with one variable in it, and the frame
    is what the scene teaches. `Jien …` wants `Jien` first; `Għandi … sena` wants
    `Għandi` first and `sena` last, whatever goes between them. The first word that
    is not there stops the run — a keyword turning up mid-sentence is not the frame.

    The slot counts as one more thing to supply: `Jien Pietru` scores 1.0, `Jien`
    alone 0.5, and an answer with none of the frame in it 0. */
function anchorScore(said, frame) {
  const parts = frame.split(SLOT);
  const slot = parts.length > 1;
  const wantPre = keys(parts[0]);
  const wantPost = slot ? keys(parts.slice(1).join(' ')) : [];
  const got = keys(said);

  const total = wantPre.length + wantPost.length + (slot ? 1 : 0);
  if (!total) return 0;

  const start = wantPre.length && got.length
    && OPENERS.some((o) => text.phoneticSimilarity(got[0], o) >= 0.8) ? 1 : 0;
  let preHits = 0;
  for (let i = 0; i < wantPre.length; i += 1) {
    if (start + i < got.length
      && text.phoneticSimilarity(got[start + i], wantPre[i]) >= 0.8) preHits += 1;
    else break;
  }
  let postHits = 0;
  for (let i = 0; i < wantPost.length; i += 1) {
    const j = got.length - 1 - i;
    const w = wantPost[wantPost.length - 1 - i];
    if (j >= 0 && text.phoneticSimilarity(got[j], w) >= 0.8) postHits += 1;
    else break;
  }

  let hits = preHits + postHits;
  // Something left over between the anchors is the answer to the question. It only
  // counts once some of the frame is there, or every stray word would look like a
  // filled slot and "hello" would score half marks for a name.
  if (slot && hits && got.length - start - preHits - postHits > 0) hits += 1;
  return hits / total;
}

function bestAnchor(said, frames) {
  const best = (frames || []).reduce((b, f) => Math.max(b, anchorScore(said, f)), 0);
  return Math.round(best * 10000) / 10000;
}

/** Is this listed answer a deliberate step outside the frame?

    Most accepted answers on an open question are the frame with an example in the
    slot — `Għandi tletin sena.` A few are an escape from it: `Dak sigriet!`,
    `Ma niftakarx!`. Saying one of those is a real answer and keeps its ordinary
    match score instead of being marked down against a frame it never used. */
function outsideFrames(candidate, frames) {
  return !!candidate && bestAnchor(candidate.mt, frames) < 1;
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
      // to *it* is a real score — better than the 0 its frame would give.
      if (!(outsideFrames(match, frames) && score > anchor)) { score = anchor; frameScored = true; }
      // `framed` is whichever example answer is nearest, for the correction card;
      // it is null when nothing overlapped at all, and then the ordinary match is
      // still the best line to show.
      match = framed || match;
    } else if (recall > score) { match = framed || match; score = recall; frameScored = true; }
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

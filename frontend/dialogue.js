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

/* …and the other way of being right.

   One absolute threshold has to do two jobs that pull against each other: high enough
   that a different sentence is turned down, low enough that a garbled correct one is
   not. It cannot do both, and 0.86 resolves that by siding with strictness — which is
   why an answer said right comes back rejected when the recogniser drops a word.
   `Naħseb li iva` heard as `naħseb iva` scores 0.900. One more slip and it is out, on a
   sentence the learner said perfectly well.

   Rank does what a threshold cannot, and the app already knows this: `MIN_CONFIDENCE`
   in app.js accepts the target when the *audio* ranks it clear of a field of other
   lines. This is the same question asked of the transcript, for the turns where the
   audio could not answer it — which are exactly the turns where the transcript is
   worst, and where demanding 0.86 of it is least reasonable.

   Measured by degrading all 334 real transcripts with the recogniser's own observed
   error types, at 3x, 5x and 8x the rate observed on clean synthesised speech — a
   learner's accented voice against a Maltese model is somewhere in there — and asking
   each node to reject the nearest line it does *not* accept:

                               said right, accepted     said wrong, accepted
                                3x     5x     8x         3x     5x     8x
     threshold 0.86 (before)   98.2%  91.3%  75.7%      3.6%   2.4%   1.5%
     threshold 0.78            99.4%  99.4%  94.6%     14.1%  12.3%   9.9%   ← too lenient
     0.86 with sound distance  99.1%  93.7%  79.0%      3.6%   2.4%   1.5%
     …and accepted on a lead  100.0% 100.0%  97.9%      3.6%   2.4%   1.5%   ← this

   The last row turns nothing away that the old threshold accepted, and turns nothing
   *in* either: the wrong-answer column is identical at all three levels. Dropping the
   threshold to 0.78 instead buys less and costs five times as many wrong answers.

   Cost: the rival scan is 377 comparisons, ~20ms here, and it only runs for a score in
   [0.66, 0.86) — the band where the app was about to turn the learner down anyway. */
export const NEAREST = 0.66;   // below this, nearest or not, nothing was said
export const LEAD = 0.06;      // how far clear of every rival the target has to be
export const MAX_ATTEMPTS = 2;

let doc = { dialogues: [] };

export function load(data) { doc = data || { dialogues: [] }; }
export const all = () => doc.dialogues || [];
export const get = (id) => all().find((d) => d.id === id) || null;
export const node = (did, nid) => get(did)?.nodes?.[nid] || null;

/** Every Maltese answer the script will ever accept, deduplicated.

    Used as the field a spoken answer is ranked against. Grading a learner cannot rely on
    an absolute confidence — measured on real recordings the target scored 0.766 where its
    own near-misses scored 0.784, so there is no cut between them, and the value moves with
    the speaker. Asking instead "does the line we asked for explain this audio better than
    the other things you could have said" is scale-free, and it worked on the same clips
    where a threshold accepted nothing. */
export function everyAnswer() {
  const out = new Set();
  for (const d of all()) {
    for (const node of Object.values(d.nodes || {})) {
      for (const a of node.accept || []) {
        // Open answers are a frame with anything in the gap — a name, a town — so they
        // are not a line anybody says verbatim and make a poor alternative.
        if (a.open) continue;
        const mt = (a.mt || '').trim();
        if (mt) out.add(mt);
      }
    }
  }
  return [...out];
}

/** Every answer one dialogue accepts, deduplicated.

    The field a spoken answer is ranked against is drawn from the whole script, which
    treats `In-nanna tagħmel il-pastizzi` as an equally likely thing to have said in the
    middle of a pharmacy scene. It is not, and the app knows it is not — so this exposes
    the scene's own lines, which are the alternatives a learner might actually produce
    here. Harder alternatives make ranking stricter, not looser, which is the safe
    direction for a grader; see `FIELD_LOCAL` in `app.js` for why it is nonetheless off. */
export function answersIn(did) {
  const out = new Set();
  for (const node of Object.values(get(did)?.nodes || {})) {
    for (const a of node.accept || []) {
      if (a.open) continue;
      const mt = (a.mt || '').trim();
      if (mt) out.add(mt);
    }
  }
  return [...out];
}

export function present(did, nid) {
  const n = node(did, nid);
  if (!n) return null;
  return {
    dialogue: did,
    node: nid,
    say_mt: n.say_mt,
    say_en: n.say_en,
    expect_en: n.expect_en || '',
    // The Maltese frame the answer is scored on, `Jien …`, `Għandi … sena`. It goes
    // on the screen beside the English: grading the frame while telling the learner
    // "anything goes" asks them to guess the half that is marked.
    frames: n.frames || [],
    options: (n.accept || []).filter((a) => !a.open).map((a) => a.en),
    // Whether the answer is the learner's own — their name, their town — which
    // changes "say this" into "say something like this".
    free: !!n.free,
    // One model answer, in Maltese, so the app can show and *say* what it is waiting
    // for before the learner has to produce it. `options` was English only, which
    // tells somebody what to mean and not what to utter.
    //
    // Deliberately a non-open one. Those are the entries `every_line()`
    // pre-synthesises, so the line offered here is always one the static build has
    // audio for; an open entry is a frame with a gap in it (`Jisimni …`), which is
    // neither speakable nor a model of anything.
    answer: (n.accept || []).filter((a) => !a.open)
      .map((a) => ({ mt: a.mt, en: a.en || '' }))[0] || null,
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
      if (text.ratio(got[i], w) >= 0.8) { hits += 1; i += 1; break; }
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
    `m'għandix` is not `għandi` either. A vowel or glide added or dropped in front is
    the recogniser — `nħobb` for `inħobb`, `isimni` for `jisimni` — but a consonant is
    not: `x'jaħdem` is the question with its interrogative attached, and `żmien` is
    not `minn`. Below three letters even that is guesswork, so short keys match
    outright or are listed above. Nor is a negation its affirmative: Maltese negates
    with `ma …-x`, the `-x` keys to a trailing s, and `Ma niekolx laħam` is the answer
    beside the frame rather than an attempt at it.

    `text.ratio`, not `phoneticSimilarity`: these are keys already, and keying a key
    collapses the doubled vowel silent għ leaves behind (`noqgħod` → `nood` → `nod`). */
function sameWord(said, want) {
  if (said === want) return true;
  if (ONE_WORD.some((group) => group.has(said) && group.has(want))) return true;
  if (Math.min(said.length, want.length) < 3) return false;
  if (said === `${want}s`) return false;
  if (said.slice(0, 1) !== want.slice(0, 1)) {
    let added = '';
    if (said.slice(1) === want) added = said.slice(0, 1);
    else if (want.slice(1) === said) added = want.slice(0, 1);
    return added !== '' && 'aeiouy'.includes(added);
  }
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
  return text.round4(best);
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
      if (text.ratio(got[i], w) >= 0.8) { hits += 1; i += 1; break; }
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
  return [best, text.round4(bestScore)];
}

/** How close what was said is to one particular line.

    Three readings, and the most generous wins, because each is blind to something the
    others see. `phoneticSimilarity` counts characters shared in order and cannot say
    whether a change was a small one. `soundSimilarity` can — it charges a substitution
    by kind, so `qadima` heard as `qatima` costs a third of a character rather than a
    whole one. And the word-aligned score is the only one of the three that notices a
    word is in the wrong place. */
export function pairScore(said, want) {
  const phon = text.phoneticSimilarity(said, want, true);
  return Math.max(phon, text.soundSimilarity(said, want),
                  0.6 * phon + 0.4 * text.score(said, want));
}

function bestMatch(said, accepted) {
  let best = null;
  let bestScore = 0;
  for (const candidate of accepted || []) {
    if (candidate.open) continue;
    let s = pairScore(said, candidate.mt);
    if (QUOTED.test(candidate.mt)) s = Math.max(s, frameScore(said, candidate.mt));
    if (s > bestScore) { best = candidate; bestScore = s; }
  }
  return [best, text.round4(bestScore)];
}

/* Every accepted line in the script with its phonetic key, built once. `everyAnswer`
   walks 113 nodes and this keys 377 strings through the Maltese rules; the script is
   loaded at boot and never edited, so doing it per utterance would be pure waste. */
let rivalCache = null;
export function rivals() {
  if (!rivalCache) {
    const seen = new Map();
    for (const mt of everyAnswer()) {
      const k = text.softKey(mt);
      if (!seen.has(k)) seen.set(k, mt);
    }
    rivalCache = [...seen.entries()].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  }
  return rivalCache;
}

/** The best score reached by any line this node does *not* accept.

    Scored with `pairScore`, the same function the match itself goes through: a rival
    measured more meanly than the target would make the lead free. Excluded by key
    rather than by identity, because the same sentence is accepted at several nodes and
    a rival that is the same words is not a rival. */
function nearestRival(said, accepted) {
  const ours = new Set((accepted || []).filter((a) => !a.open).map((a) => text.softKey(a.mt)));
  let best = 0;
  for (const [k, mt] of rivals()) {
    if (ours.has(k)) continue;
    const s = pairScore(said, mt);
    if (s > best) best = s;
  }
  return text.round4(best);
}

export function evaluate(did, nid, said, attempts = 0) {
  const n = node(did, nid);
  if (!n) return { error: 'unknown node' };

  said = text.normalise(said);
  const [match0, score0] = bestMatch(said, n.accept);

  let verdict;
  let frameScored = false;
  let onLead = false;
  let [match, score] = [match0, score0];
  if (n.free) {
    // A name, a place, a number: never marked wrong, because the app cannot know
    // it. But the frame around the slot is ordinary Maltese, so that part is
    // scored and reported — "Jien Pietru" is not the same as "hello".
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
        score = anchor;
        frameScored = true;
      }
      // `match` is left as the nearest listed answer either way: it is the line the
      // correction card shows, and on the escape path it is the line the score was
      // measured against. The frame has no better candidate — recall favours the
      // shortest answer, not the nearest one.
    } else {
      // No slot to anchor on: score by how much of an example answer they made —
      // not a frame, and not claimed as one.
      const [framed, recall] = bestFrame(said, n.accept);
      if (recall > score) { match = framed || match; score = recall; }
    }
    verdict = text.fold(said).length >= 2 ? 'correct' : 'wrong';
  } else if (score >= CORRECT) verdict = 'correct';
  else if (score >= NEAREST && score - nearestRival(said, n.accept) >= LEAD) {
    // Nearer to this answer than to anything else the app knows, by a clear margin.
    // Right, then, and heard badly.
    verdict = 'correct';
    onLead = true;
  } else if (score >= CLOSE) verdict = 'close';
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
    // Accepted because it was the clear nearest, not because it scored well. The UI
    // says so and still shows the line: waving through a mangled answer without
    // showing what it should have sounded like teaches the mangling.
    on_lead: onLead,
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

  // On anything short of correct — and on anything accepted only on its lead — show
  // the target so they can say it back.
  if ((verdict !== 'correct' || onLead) && match) {
    out.say_this_mt = match.mt;
    out.say_this_en = match.en;
    out.diff = text.diffWords(said, match.mt);
  }

  if (nextNode && verdict === 'correct') out.next = present(did, nextNode);
  else if (verdict === 'correct') { out.next = null; out.finished = true; }
  return out;
}

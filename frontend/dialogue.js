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
  const [match, score] = bestMatch(said, n.accept);

  let verdict;
  if (n.free) {
    // A name, a place, a number — nothing the app has any business checking.
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

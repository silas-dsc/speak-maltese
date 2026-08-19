/* Marking the mini-games, and choosing what to ask next.

   The items themselves are built by `backend/games.py` and shipped as data — derived
   from the scripted dialogues and the deck, so nothing here has to know how a tile
   puzzle is made or which words sound alike. That is on purpose: every other piece of
   logic in this app exists twice, in Python and again in JavaScript, with a parity test
   holding the two together. This one does not, because there is nothing to port.

   What is left is small enough to be worth getting exactly right:

   * a build answer is correct only if the tiles are the right words *in the right
     order*, and only if the redundant ones were left alone;
   * a session interleaves the kinds rather than serving eight of one, because eight
     tile puzzles in a row trains the puzzle rather than the language;
   * and what a learner got right is worth remembering only where it is evidence of
     something. See `earned()`. */

/** Is this answer right? One shape per kind, all of them exact.

    Deliberately strict about order. A learner who assembles `Nixtieq kafè jekk
    jogħġbok` in the wrong order has not built the sentence, and a game that accepts it
    teaches that Maltese word order is optional — which is the one thing this exercise
    exists to teach. */
export function mark(item, answer) {
  switch (item.kind) {
    case 'build': {
      const placed = answer || [];
      return placed.length === item.answer.length
        && placed.every((word, i) => word === item.answer[i]);
    }
    case 'grammar':
    case 'hearing':
      return answer === item.answer;
    case 'listening':
      // `which` is one option; `words` is a set of three, in any order.
      return item.ask === 'which'
        ? answer === item.answer
        : sameSet(answer || [], item.answer);
    default:
      return false;
  }
}

function sameSet(a, b) {
  if (a.length !== b.length) return false;
  const left = [...a].sort();
  const right = [...b].sort();
  return left.every((x, i) => x === right[i]);
}

/** What was wrong about it, in the learner's terms rather than the data's.

    A bare "wrong" on a tile puzzle is the least useful thing an exercise can say: the
    two ways to fail — wrong words, or right words in the wrong order — need different
    corrections, and the app knows which it was. */
export function critique(item, answer) {
  if (item.kind !== 'build') return '';
  const placed = answer || [];
  if (!placed.length) return 'Nothing placed yet.';
  if (sameSet(placed, item.answer)) return 'The right words — wrong order.';
  const extra = placed.filter((w) => !item.answer.includes(w));
  if (extra.length) {
    return `${extra.join(', ')} ${extra.length === 1 ? 'does' : 'do'} not belong in this one.`;
  }
  return 'Some words are missing.';
}

/** A session: `count` items, the kinds interleaved, stable for a given seed.

    Round-robin rather than shuffled. A shuffle of a pooled list gives runs — four tile
    puzzles together happens often in eight draws — and a run is the thing to avoid:
    the second and third are answered by momentum rather than by knowing anything.

    `kinds` may name a single kind, for a learner who has come specifically to do
    listening. Then interleaving is not wanted and not done. */
export function session(payload, { count = 8, kinds = null, seed = Date.now() } = {}) {
  const wanted = (kinds && kinds.length ? kinds : ['build', 'hearing', 'listening', 'grammar'])
    .filter((k) => (payload[k] || []).length);
  if (!wanted.length) return [];

  const rng = mulberry(seed);
  const queues = wanted.map((k) => shuffle(payload[k], rng));
  const out = [];
  for (let round = 0; out.length < count; round += 1) {
    let took = 0;
    for (const queue of queues) {
      if (out.length >= count) break;
      if (round < queue.length) { out.push(queue[round]); took += 1; }
    }
    if (!took) break;                    // every kind exhausted
  }
  return out;
}

/* A seeded shuffle so a session can be reproduced from its seed — which is what makes
   "the same session again" possible after a reload, and what keeps the tests from
   depending on the clock. */
function mulberry(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(items, rng) {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/** The Maltese a correct answer is evidence of knowing — or nothing.

    The review deck is scheduled by FSRS, and FSRS is only as good as what it is told.
    A card filed as known because it was recognised among three options will come back
    at the interval of something produced from memory, which is how a deck fills up with
    words the learner cannot actually say.

    So: only the two kinds where the learner produced the whole thing count. Building a
    sentence tile by tile is production of a sort — the words are given but the sentence
    is theirs — and a listening fragment answered correctly is comprehension of specific
    lines. Choosing one word out of three, in either the grammar or the hearing game, is
    recognition, and recognition is not what the deck measures. Nothing is filed for it.

    The same reasoning as the drill's `peeked`: better an empty deck than a deck of
    things marked learned on the strength of a lucky guess between three options. */
export function earned(item, correct) {
  if (!correct) return [];
  if (item.kind === 'build') return [{ mt: item.mt, en: item.prompt_en }];
  return [];
}

/** The line to play for an item, if it has one. */
export function audioFor(item) {
  if (item.kind === 'build') return item.mt;
  if (item.kind === 'hearing') return item.say;
  if (item.kind === 'listening') return item.script;
  if (item.kind === 'grammar') {
    // The sentence with the gap filled, which is the whole point of hearing it.
    return item.ask_mt.replace('___', item.options[item.answer]);
  }
  return '';
}

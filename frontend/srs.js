/* FSRS-5, ported from backend/srs.py so the schedule can live in the browser.

   The learner's review history is the one thing in this app that cannot be
   regenerated: decks, audio and images all rebuild from the repo, but nobody can
   recover which words you were about to forget. Keeping it server-side meant one
   shared database for every visitor, and on free hosting it also meant losing the
   lot on each restart. So the scheduler runs here and the history stays on the
   device.

   This file is a direct translation, not a reimplementation — the constants, the
   order of operations and the clamping all mirror the Python. `tests/test_srs_parity`
   runs both over the same review sequences and compares, because a scheduler that
   silently disagrees with its own tests is worse than no scheduler.

   Grades: 1 = Again, 2 = Hard, 3 = Good, 4 = Easy. */

export const W = [
  0.40255, 1.18385, 3.17300, 15.69105, 7.19490, 0.53450, 1.46040, 0.00460,
  1.54575, 0.11920, 1.01925, 1.93950, 0.11000, 0.29605, 2.26980, 0.23150,
  2.98980, 0.51655, 0.66210,
];

export const DECAY = -0.5;
export const FACTOR = 19.0 / 81.0;

const MIN_STABILITY = 0.01;
const MIN_INTERVAL = 1;
const MAX_INTERVAL = 365 * 5;

const LEARNING_STEPS = [1, 10];      // minutes
const RELEARNING_STEPS = [10];

export const AGAIN = 1, HARD = 2, GOOD = 3, EASY = 4;

const DAY_MS = 86400000;

/* Python's round() is banker's rounding — round(0.5) is 0, round(2.5) is 2 —
   while JS Math.round always goes up. Interval lengths are compared against the
   Python in the parity test, so the rounding has to match too. */
function roundHalfEven(x) {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

const clampS = (s) => Math.max(MIN_STABILITY, s);
const clampD = (d) => Math.min(10.0, Math.max(1.0, d));

export function newState() {
  return {
    stability: 0, difficulty: 0, reps: 0, lapses: 0,
    state: 'new', step: 0, due: null, lastReview: null,
    prodReps: 0, prodCorrect: 0,
  };
}

export function retrievability(elapsedDays, stability) {
  if (stability <= 0) return 0;
  return (1.0 + FACTOR * elapsedDays / stability) ** DECAY;
}

export function intervalFor(stability, targetRetention) {
  if (stability <= 0) return MIN_INTERVAL;
  const days = (stability / FACTOR) * (targetRetention ** (1.0 / DECAY) - 1.0);
  return Math.max(MIN_INTERVAL, Math.min(MAX_INTERVAL, roundHalfEven(days)));
}

const initStability = (grade) => clampS(W[grade - 1]);
const initDifficulty = (grade) => clampD(W[4] - Math.exp(W[5] * (grade - 1)) + 1.0);

function nextDifficulty(d, grade) {
  const delta = -W[6] * (grade - 3);
  const damped = d + delta * (10.0 - d) / 9.0;
  return clampD(W[7] * initDifficulty(EASY) + (1.0 - W[7]) * damped);
}

function stabilityOnSuccess(d, s, r, grade) {
  const hardPenalty = grade === HARD ? W[15] : 1.0;
  const easyBonus = grade === EASY ? W[16] : 1.0;
  const growth = Math.exp(W[8])
    * (11.0 - d)
    * (s ** -W[9])
    * (Math.exp(W[10] * (1.0 - r)) - 1.0)
    * hardPenalty
    * easyBonus;
  return clampS(s * (1.0 + growth));
}

function stabilityOnLapse(d, s, r) {
  const lapsed = W[11]
    * (d ** -W[12])
    * (((s + 1.0) ** W[13]) - 1.0)
    * Math.exp(W[14] * (1.0 - r));
  return clampS(Math.min(lapsed, s));   // a lapse must never raise stability
}

const stabilitySameDay = (s, grade) => clampS(s * Math.exp(W[17] * (grade - 3 + W[18])));

/** Apply one review. Returns a new state object; the input is not mutated. */
export function review(card, grade, targetRetention = 0.9, at = null) {
  const c = { ...card };
  const t = at ? new Date(at).getTime() : Date.now();
  grade = Math.max(AGAIN, Math.min(EASY, Math.trunc(grade)));

  if (c.state === 'new') {
    c.stability = initStability(grade);
    c.difficulty = initDifficulty(grade);
    c.reps = 1;
    c.lastReview = t;
    if (grade === EASY) {
      c.state = 'review';
      c.step = 0;
      c.due = t + intervalFor(c.stability, targetRetention) * DAY_MS;
    } else {
      c.state = 'learning';
      c.step = grade === AGAIN ? 0 : Math.min(1, LEARNING_STEPS.length - 1);
      c.due = t + LEARNING_STEPS[c.step] * 60000;
    }
    return c;
  }

  let elapsed = 0;
  if (c.lastReview) elapsed = Math.max(0, (t - c.lastReview) / DAY_MS);
  const r = c.stability > 0 ? retrievability(elapsed, c.stability) : 0.9;

  c.reps += 1;
  c.difficulty = nextDifficulty(c.difficulty, grade);
  const sameDay = elapsed < 1.0;

  if (grade === AGAIN) {
    c.lapses += 1;
    c.stability = stabilityOnLapse(c.difficulty, c.stability, r);
    c.state = 'relearning';
    c.step = 0;
    c.due = t + RELEARNING_STEPS[0] * 60000;
  } else {
    if (sameDay && (c.state === 'learning' || c.state === 'relearning')) {
      c.stability = stabilitySameDay(c.stability, grade);
    } else {
      c.stability = stabilityOnSuccess(c.difficulty, c.stability, r, grade);
    }

    if (c.state === 'learning' || c.state === 'relearning') {
      const steps = c.state === 'learning' ? LEARNING_STEPS : RELEARNING_STEPS;
      if (grade === EASY || c.step >= steps.length - 1) {
        c.state = 'review';
        c.step = 0;
        c.due = t + intervalFor(c.stability, targetRetention) * DAY_MS;
      } else {
        c.step += 1;
        c.due = t + steps[c.step] * 60000;
      }
    } else {
      c.state = 'review';
      let days = intervalFor(c.stability, targetRetention);
      if (grade === HARD) days = Math.max(MIN_INTERVAL, roundHalfEven(days * 0.8));
      c.due = t + days * DAY_MS;
    }
  }

  c.lastReview = t;
  return c;
}

/** Next interval per grade, for the answer buttons. */
export function preview(card, targetRetention = 0.9, at = null) {
  const t = at ? new Date(at).getTime() : Date.now();
  const out = {};
  for (const g of [AGAIN, HARD, GOOD, EASY]) {
    const next = review(card, g, targetRetention, t);
    out[g] = humanise(next.due ? (next.due - t) / 1000 : 0);
  }
  return out;
}

export function humanise(seconds) {
  const mins = seconds / 60;
  if (mins < 60) return `${Math.max(1, Math.round(mins))}m`;
  const hours = mins / 60;
  if (hours < 24) return `${Math.round(hours)}h`;
  const days = hours / 24;
  if (days < 30) return `${Math.round(days)}d`;
  if (days < 365) return `${(days / 30.4).toFixed(1)}mo`;
  return `${(days / 365).toFixed(1)}y`;
}

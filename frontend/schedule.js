/* Session planning and progress, ported from backend/curriculum.py and db.py.

   The pedagogy is unchanged — frequency-ordered introduction, phrases interleaved
   with words, new cards spread through the due queue rather than front-loaded,
   production favoured over recognition. Only the storage moved. */

import * as store from './store.js';
import * as srs from './srs.js';

const DAY_MS = 86400000;

/* ── Queue ───────────────────────────────────────────────────────────────── */

export async function buildQueue(limit = 20, { topics = null, includeNew = true,
  maxTier = null, settings } = {}) {
  const [cards, states] = await Promise.all([store.getCards(), store.getStates()]);
  const byId = new Map(cards.map((c) => [c.id, c]));
  const now = Date.now();

  const live = states
    .filter((s) => byId.has(s.cardId) && !s.suspended)
    .map((s) => ({ ...byId.get(s.cardId), ...s, id: s.cardId }));

  const reviews = await store.getReviews();
  const doneToday = countSince(reviews, now - DAY_MS);
  const reviewBudget = Math.max(0, settings.daily_review - doneToday);

  let due = live
    .filter((c) => c.state !== 'new' && c.due != null && c.due <= now)
    .filter((c) => !topics || topics.includes(c.topic))
    .sort((a, b) => a.due - b.due)
    .slice(0, Math.min(limit, reviewBudget));

  let fresh = [];
  if (includeNew && due.length < limit) {
    const introduced = newIntroducedSince(reviews, live, now - DAY_MS);
    const newBudget = Math.max(0, settings.daily_new - introduced);
    fresh = live
      .filter((c) => c.state === 'new')
      .filter((c) => !topics || topics.includes(c.topic))
      .filter((c) => !maxTier || c.tier <= maxTier)
      // phrases first within a tier, so you always leave with something sayable
      .sort((a, b) => (a.tier - b.tier)
        || ((a.kind === 'phrase' ? 0 : 1) - (b.kind === 'phrase' ? 0 : 1))
        || String(a.id).localeCompare(String(b.id)))
      .slice(0, Math.min(limit - due.length, newBudget));
  }

  const queue = interleave(due, fresh).slice(0, limit);
  for (const c of queue) {
    c.mode = pickMode(c);
    c.intervals = srs.preview(c, settings.target_retention);
  }
  return queue;
}

export const countSince = (reviews, since) => reviews.filter((r) => r.at >= since).length;

/** Distinct barely-seen cards reviewed in the window — the daily new allowance. */
export function newIntroducedSince(reviews, live, since) {
  const young = new Set(live.filter((c) => (c.reps ?? 0) <= 2).map((c) => c.id));
  const seen = new Set();
  for (const r of reviews) if (r.at >= since && young.has(r.cardId)) seen.add(r.cardId);
  return seen.size;
}

/** Spread new cards through the review queue — interleaving beats blocking. */
export function interleave(due, fresh) {
  if (!fresh.length) return due;
  if (!due.length) return fresh;
  const out = [];
  const gap = Math.max(1, Math.floor(due.length / (fresh.length + 1)));
  let ni = 0;
  due.forEach((card, i) => {
    out.push(card);
    if (ni < fresh.length && (i + 1) % gap === 0) out.push(fresh[ni++]);
  });
  return out.concat(fresh.slice(ni));
}

function pickMode(card) {
  if (card.state === 'new') return 'listen';
  const prod = card.prodReps || 0;
  const reps = card.reps || 0;
  if (reps < 2) return 'recognise';
  if (prod * 2 < reps) return 'produce';
  return ['produce', 'recognise', 'listen'][Math.floor(Math.random() * 3)];
}

/* ── Recording a review ──────────────────────────────────────────────────── */

export async function recordReview(cardId, grade, mode, { score = null, said = null,
  elapsedMs = null, settings } = {}) {
  const prev = (await store.getState(cardId)) || { cardId };
  const next = srs.review(prev, grade, settings.target_retention);
  if (mode === 'produce') {
    next.prodReps = (prev.prodReps || 0) + 1;
    next.prodCorrect = (prev.prodCorrect || 0) + (grade >= srs.GOOD ? 1 : 0);
  }
  next.suspended = prev.suspended || 0;
  await store.putState(cardId, next);
  await store.logReview({ cardId, grade, mode, score, said, elapsedMs, at: Date.now() });
  return next;
}

/** A phrase produced correctly in conversation is exactly what should be scheduled. */
export async function registerFromDrill(items, settings, topic = 'drill') {
  const added = [];
  for (const it of items) {
    const mt = (it.mt || '').trim();
    const en = (it.en || '').trim();
    if (!mt || !en) continue;
    const id = `t${slug(mt)}`;
    const existing = await store.getState(id);
    await store.addCards([{
      id, kind: mt.includes(' ') ? 'phrase' : 'vocab', mt, en,
      tier: 3, topic, source: 'drill',
    }]);
    // Only credit it the first time; re-meeting a phrase must not reset its schedule.
    if (!existing || existing.state === 'new') {
      await recordReview(id, srs.GOOD, 'produce', { settings });
      added.push(id);
    }
  }
  return added;
}

/** Mirrors backend `_slug`: readable, and a digest when the readable part is cut,
    because two long phrases sharing a prefix must not collapse onto one card. */
export function slug(s) {
  const keep = [...s].map((ch) => (/[\p{L}\p{N}]/u.test(ch) ? ch.toLowerCase() : '-'))
    .join('').replace(/^-+|-+$/g, '');
  if (keep.length <= 48) return keep;
  return `${keep.slice(0, 41).replace(/-+$/, '')}-${hash6(s)}`;
}

/* FNV-1a. Not the backend's SHA-1 — ids only have to be stable and distinct
   within one device, and this avoids pulling in SubtleCrypto for a card key. */
function hash6(s) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i += 1) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0').slice(0, 6);
}

/* ── Counts and progress ─────────────────────────────────────────────────── */

export async function counts() {
  const [states, reviews] = await Promise.all([store.getStates(), store.getReviews()]);
  const now = Date.now();
  return {
    new: states.filter((s) => s.state === 'new').length,
    due: states.filter((s) => s.state !== 'new' && s.due != null && s.due <= now).length,
    learned: states.filter((s) => s.state === 'review').length,
    solid: states.filter((s) => s.state === 'review' && s.stability >= 21).length,
    total: states.length,
    today: countSince(reviews, now - DAY_MS),
  };
}

export async function stats() {
  const [cards, states, reviews] = await Promise.all([
    store.getCards(), store.getStates(), store.getReviews()]);
  const byId = new Map(cards.map((c) => [c.id, c]));
  const base = await counts();

  const byDay = new Map();
  for (const r of reviews) {
    const d = localDay(r.at);
    const acc = byDay.get(d) || { d, n: 0, good: 0 };
    acc.n += 1;
    if (r.grade >= srs.GOOD) acc.good += 1;
    byDay.set(d, acc);
  }
  const history = [...byDay.values()].sort((a, b) => a.d.localeCompare(b.d)).slice(-60)
    .map((h) => ({ d: h.d, n: h.n, retention: h.n ? h.good / h.n : 0 }));

  const topics = new Map();
  for (const s of states) {
    const card = byId.get(s.cardId);
    if (!card?.topic) continue;
    const t = topics.get(card.topic) || { topic: card.topic, total: 0, learned: 0 };
    t.total += 1;
    if (s.state === 'review') t.learned += 1;
    topics.set(card.topic, t);
  }

  const weak = states.filter((s) => s.lapses > 0)
    .sort((a, b) => (b.lapses - a.lapses) || (b.difficulty - a.difficulty))
    .slice(0, 15)
    .map((s) => ({ ...byId.get(s.cardId), lapses: s.lapses }))
    .filter((w) => w.mt);

  return {
    ...base,
    history,
    topics: [...topics.values()].sort((a, b) => b.total - a.total),
    speaking: {
      attempts: states.reduce((n, s) => n + (s.prodReps || 0), 0),
      correct: states.reduce((n, s) => n + (s.prodCorrect || 0), 0),
    },
    weak,
    streak: streak(new Set(reviews.map((r) => localDay(r.at)))),
  };
}

/* Days are local, and so is the streak that walks them — the two have to agree.
   The server got this wrong in the other direction: it stored UTC and compared
   against the local date, so an early-morning review east of Greenwich landed on
   the previous day and the streak read zero. */
export function localDay(ms) {
  const d = new Date(ms);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

export function streak(days) {
  let n = 0;
  const cur = new Date();
  while (days.has(localDay(cur.getTime()))) {
    n += 1;
    cur.setDate(cur.getDate() - 1);
  }
  return n;
}

export function estimateLevel(learned) {
  if (learned < 30) return 'A0';
  if (learned < 120) return 'A1';
  if (learned < 320) return 'A2';
  if (learned < 700) return 'B1';
  return 'B2';
}

"""FSRS-5 scheduler (Free Spaced Repetition Scheduler).

Chosen over SM-2 because it models memory as two separate quantities — *stability*
(how many days the memory lasts) and *difficulty* (how hard this item is for you) —
and schedules to hit an explicit target retention rather than climbing a fixed
multiplier ladder. Same recall, materially fewer reviews.

Grades: 1 = Again, 2 = Hard, 3 = Good, 4 = Easy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# FSRS-5 default weights (19), fitted over a very large public review corpus.
# w[0:4]  initial stability per grade
# w[4:6]  initial difficulty
# w[6:8]  difficulty update + mean reversion
# w[8:11] stability growth on success
# w[11:15] stability after a lapse
# w[15:17] hard penalty / easy bonus
# w[17:19] same-day (short-term) stability update
W = [
    0.40255, 1.18385, 3.17300, 15.69105, 7.19490, 0.53450, 1.46040, 0.00460,
    1.54575, 0.11920, 1.01925, 1.93950, 0.11000, 0.29605, 2.26980, 0.23150,
    2.98980, 0.51655, 0.66210,
]

DECAY = -0.5
FACTOR = 19.0 / 81.0

MIN_STABILITY = 0.01
MIN_INTERVAL = 1        # days
MAX_INTERVAL = 365 * 5  # days

# Sub-day steps for brand-new / lapsed cards, in minutes.
LEARNING_STEPS = (1, 10)
RELEARNING_STEPS = (10,)

AGAIN, HARD, GOOD, EASY = 1, 2, 3, 4


def now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class CardState:
    stability: float = 0.0
    difficulty: float = 0.0
    reps: int = 0
    lapses: int = 0
    state: str = "new"      # new | learning | review | relearning
    step: int = 0           # index into LEARNING_STEPS / RELEARNING_STEPS
    due: datetime | None = None
    last_review: datetime | None = None


def retrievability(elapsed_days: float, stability: float) -> float:
    """Probability of recall after `elapsed_days`, given `stability`."""
    if stability <= 0:
        return 0.0
    return (1.0 + FACTOR * elapsed_days / stability) ** DECAY


def interval_for(stability: float, target_retention: float) -> int:
    """Days until retrievability decays to `target_retention`.

    At the default 0.9 this returns ≈ stability, which is the property that makes
    stability readable as "days this will survive".
    """
    if stability <= 0:
        return MIN_INTERVAL
    days = (stability / FACTOR) * (target_retention ** (1.0 / DECAY) - 1.0)
    return max(MIN_INTERVAL, min(MAX_INTERVAL, round(days)))


def _clamp_s(s: float) -> float:
    return max(MIN_STABILITY, s)


def _clamp_d(d: float) -> float:
    return min(10.0, max(1.0, d))


def _init_stability(grade: int) -> float:
    return _clamp_s(W[grade - 1])


def _init_difficulty(grade: int) -> float:
    return _clamp_d(W[4] - math.exp(W[5] * (grade - 1)) + 1.0)


def _next_difficulty(d: float, grade: int) -> float:
    # FSRS-5 damps the update as difficulty approaches the 10 ceiling.
    delta = -W[6] * (grade - 3)
    damped = d + delta * (10.0 - d) / 9.0
    # then mean-revert towards the difficulty an "easy" first answer would give
    return _clamp_d(W[7] * _init_difficulty(EASY) + (1.0 - W[7]) * damped)


def _stability_on_success(d: float, s: float, r: float, grade: int) -> float:
    hard_penalty = W[15] if grade == HARD else 1.0
    easy_bonus = W[16] if grade == EASY else 1.0
    growth = (
        math.exp(W[8])
        * (11.0 - d)
        * (s ** -W[9])
        * (math.exp(W[10] * (1.0 - r)) - 1.0)
        * hard_penalty
        * easy_bonus
    )
    return _clamp_s(s * (1.0 + growth))


def _stability_on_lapse(d: float, s: float, r: float) -> float:
    lapsed = (
        W[11]
        * (d ** -W[12])
        * (((s + 1.0) ** W[13]) - 1.0)
        * math.exp(W[14] * (1.0 - r))
    )
    # A lapse must never *raise* stability.
    return _clamp_s(min(lapsed, s))


def _stability_same_day(s: float, grade: int) -> float:
    """Short-term update for a second look on the same day (learning steps)."""
    return _clamp_s(s * math.exp(W[17] * (grade - 3 + W[18])))


def review(card: CardState, grade: int, target_retention: float = 0.9,
           at: datetime | None = None) -> CardState:
    """Apply one review and return the updated state."""
    at = at or now()
    grade = max(AGAIN, min(EASY, int(grade)))

    if card.state == "new":
        card.stability = _init_stability(grade)
        card.difficulty = _init_difficulty(grade)
        card.reps = 1
        card.last_review = at
        if grade == EASY:
            card.state = "review"
            card.step = 0
            card.due = at + timedelta(days=interval_for(card.stability, target_retention))
        else:
            card.state = "learning"
            card.step = 0 if grade == AGAIN else min(1, len(LEARNING_STEPS) - 1)
            card.due = at + timedelta(minutes=LEARNING_STEPS[card.step])
        return card

    elapsed = 0.0
    if card.last_review:
        elapsed = max(0.0, (at - card.last_review).total_seconds() / 86400.0)
    r = retrievability(elapsed, card.stability) if card.stability > 0 else 0.9

    card.reps += 1
    card.difficulty = _next_difficulty(card.difficulty, grade)
    same_day = elapsed < 1.0

    if grade == AGAIN:
        card.lapses += 1
        card.stability = _stability_on_lapse(card.difficulty, card.stability, r)
        card.state = "relearning"
        card.step = 0
        card.due = at + timedelta(minutes=RELEARNING_STEPS[0])
    else:
        if same_day and card.state in ("learning", "relearning"):
            card.stability = _stability_same_day(card.stability, grade)
        else:
            card.stability = _stability_on_success(card.difficulty, card.stability, r, grade)

        if card.state in ("learning", "relearning"):
            steps = LEARNING_STEPS if card.state == "learning" else RELEARNING_STEPS
            if grade == EASY or card.step >= len(steps) - 1:
                card.state = "review"
                card.step = 0
                card.due = at + timedelta(days=interval_for(card.stability, target_retention))
            else:
                card.step += 1
                card.due = at + timedelta(minutes=steps[card.step])
        else:
            card.state = "review"
            days = interval_for(card.stability, target_retention)
            if grade == HARD:
                days = max(MIN_INTERVAL, round(days * 0.8))
            card.due = at + timedelta(days=days)

    card.last_review = at
    return card


def preview(card: CardState, target_retention: float = 0.9,
            at: datetime | None = None) -> dict[int, str]:
    """Human-readable next interval per grade, for the answer buttons."""
    at = at or now()
    out: dict[int, str] = {}
    for g in (AGAIN, HARD, GOOD, EASY):
        clone = CardState(**vars(card))
        nxt = review(clone, g, target_retention, at)
        out[g] = _humanise((nxt.due - at).total_seconds() if nxt.due else 0)
    return out


def _humanise(seconds: float) -> str:
    mins = seconds / 60.0
    if mins < 60:
        return f"{max(1, round(mins))}m"
    hours = mins / 60.0
    if hours < 24:
        return f"{round(hours)}h"
    days = hours / 24.0
    if days < 30:
        return f"{round(days)}d"
    if days < 365:
        return f"{days / 30.4:.1f}mo"
    return f"{days / 365.0:.1f}y"

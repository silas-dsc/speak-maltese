"""Re-run the duration-weight CV with the set the original sweep actually cared about.

`DUR_WEIGHT = 0.1` was chosen against a column labelled "near-miss rejected", and that
label means the opposite of what this session has been calling a near-miss: it is the
learner's *correct* audio scored against a *different* deck line, which must be refused.
Lowering the weight makes a long hypothesis cheaper, so that set is exactly the one at risk
and leaving it out would be measuring the change on the half that flatters it.

It needs no new model pass. The cache holds confidence and token count for every deck line
on every clip, so pairing a clip with a wrong line is just choosing a different entry as
the target and leaving the rest as the field — the same arithmetic the app does.
"""
import json
import random
from pathlib import Path

CACHE = Path("data/eval_clips/duration_cache.json")
FIELD, SEEDS = 24, (1, 2, 3, 4, 5)
FLOOR, MARGIN = 0.15, 0.02
DUR_INTERCEPT, DUR_SLOPE, DUR_SD = 28.28, 1.8794, 13.27
WEIGHTS = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
FOLDS, PAIRINGS = 5, 4
REWARD = ("honest", "near_miss")
CHARGE = ("nonanswer", "noise", "reversed", "wrong_line")

def main() -> int:
    """Everything below runs a sweep, so it lives behind a call rather than at
    import time. `test_script_imports` imports every script in this directory: a
    module that loads a model and reads a cache while being imported fails wherever
    onnxruntime or the audio is absent, which is CI.
    """
    recs = json.loads(CACHE.read_text())


    def prior(t, f):
        return -0.5 * ((f - (DUR_INTERCEPT + DUR_SLOPE * t)) / DUR_SD) ** 2


    def accepted(rec, w):
        f = rec["frames"]
        target = rec["conf"] + w * prior(rec["tokens"], f)
        scored = [c + w * prior(t, f) for t, c in rec["rivals"]]
        hits = 0
        for seed in SEEDS:
            rng = random.Random(seed)
            drawn = rng.sample(scored, min(FIELD, len(scored))) if scored else []
            best = max(drawn) if drawn else -float("inf")
            hits += int(rec["conf"] >= FLOOR and target > best + MARGIN)
        return hits / len(SEEDS)


    # Build the wrong-line pairings: correct audio, a target it is not.
    paired = []
    for rec in recs:
        if rec["group"] != "honest" or len(rec["rivals"]) < FIELD + 2:
            continue
        # HARD pairings, not random ones. `constrained_ctc` says plainly that another deck
        # line is an easy negative — "Bonġu against In-nanna tagħmel il-pastizzi is not a
        # decision anything gets wrong" — and that what the app must survive is a learner who
        # nearly said it. So the target is the wrong line that fits this audio BEST, which is
        # the pairing most likely to be wrongly accepted, and the next few after it.
        order = sorted(range(len(rec["rivals"])), key=lambda i: -rec["rivals"][i][1])
        for idx in order[:PAIRINGS]:
            tok, conf = rec["rivals"][idx]
            rest = rec["rivals"][:idx] + rec["rivals"][idx + 1:]
            # The real line the learner said is still in the field, which is the point: the
            # correct line ought to beat the wrong one it is being scored as.
            paired.append({"group": "wrong_line", "frames": rec["frames"], "conf": conf,
                           "tokens": tok,
                           "rivals": rest + [[rec["tokens"], rec["conf"]]]})
    allrecs = recs + paired
    print(f"{len(recs)} clips + {len(paired)} wrong-line pairings\n")

    GROUPS = ["honest", "near_miss", "wrong_mt", "nonanswer", "noise", "reversed", "wrong_line"]


    def rates(w, subset):
        per = {g: [0.0, 0] for g in GROUPS}
        for rec in subset:
            if rec["group"] not in per:
                continue
            per[rec["group"]][0] += accepted(rec, w)
            per[rec["group"]][1] += 1
        return {g: (v[0] / v[1] if v[1] else 0.0, v[1]) for g, v in per.items()}


    print("  weight" + "".join(f"{g:>12}" for g in GROUPS))
    for w in WEIGHTS:
        r = rates(w, allrecs)
        mark = "  <- deployed" if w == 0.1 else ""
        print(f"  {w:5.3f} " + "".join(f"{r[g][0] * 100:11.0f}%" for g in GROUPS) + mark)


    def score(subset, w, lam):
        good = [accepted(r, w) for r in subset if r["group"] in REWARD]
        bad = [accepted(r, w) for r in subset if r["group"] in CHARGE]
        g = sum(good) / len(good) if good else 0.0
        b = sum(bad) / len(bad) if bad else 0.0
        return g - lam * b, g, b


    by_group = {}
    for r in allrecs:
        by_group.setdefault(r["group"], []).append(r)
    folds = [[] for _ in range(FOLDS)]
    for group, items in sorted(by_group.items()):
        rng = random.Random(len(group) * 31)
        order = items[:]
        rng.shuffle(order)
        for i, rec in enumerate(order):
            folds[i % FOLDS].append(rec)

    for lam in (1.0, 3.0):
        picks, held = [], []
        for k in range(FOLDS):
            train = [r for i, f in enumerate(folds) if i != k for r in f]
            best = max(WEIGHTS, key=lambda w: score(train, w, lam)[0])
            picks.append(best)
            _s, g, b = score(folds[k], best, lam)
            held.append((g, b))
        print(f"\n  lambda {lam}: picks {picks}")
        print(f"    held-out with the fold's pick: credited "
              f"{sum(g for g, _ in held) / FOLDS * 100:.1f}%, "
              f"admitted {sum(b for _, b in held) / FOLDS * 100:.1f}%")
        for w in (0.025, 0.05, 0.1):
            t = [score(f, w, lam) for f in folds]
            print(f"    fixed {w:5.3f}: credited {sum(x[1] for x in t) / FOLDS * 100:.1f}%, "
                  f"admitted {sum(x[2] for x in t) / FOLDS * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

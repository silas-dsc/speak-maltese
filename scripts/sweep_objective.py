"""Sweep the accept rule against the objective actually wanted, not balanced accuracy.

Every constant in the grader was fitted to balanced accuracy over accept/reject clips.
That objective treats a refused correct answer and an admitted wrong one as equally bad,
and under it the folds unanimously picked floor 0.55. The stated aim is the opposite:
maximise the share of honest correct answers that get credited, and only insist on
blocking what is wildly different — silence, filler, English, a fragment, a change of
subject.

So: reward accepting honest attempts, charge only for admitting a non-answer, and report
the middle categories without charging for them.

One thing this cannot do is charge for the real non-answers alone. They are refused at
every floor and margin in the grid, so that penalty term is identically zero and the
optimum degenerates to "accept everything" — which would admit digital silence. The
synthetic negatives (silence, white noise, reversed speech) are therefore charged
alongside them, as the floor's guard against a rule that has stopped discriminating.
"""
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")

from constrained_ctc import confidence, encode, load, rank_score

from backend import dialogue
from backend import text as mtext
from make_negatives import read_clip

CLIPS = Path("data/eval_clips")
CACHE = CLIPS / "objective_cache.json"
FIELD = 24          # RANK_AGAINST in frontend/app.js
SEEDS = (1, 2, 3, 4, 5)
NONANSWER = ("silence", "filler", "english", "partial", "offtopic")


def norm(s):
    return mtext.normalise(s).lower().strip()


def rows(path, delim="\t"):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter=delim))


def gather():
    """(group, path, target_text) for every clip that has a target to score against."""
    out = []
    # me_* only. The 25 synth_*.mp3 in the same manifest are the app's own text-to-speech
    # renderings, and counting them as honest attempts measures TTS, which is the exact
    # thing the learner clips were recorded to stop happening.
    for r in rows(CLIPS / "manifest.tsv"):
        if r["file"].startswith("me_"):
            out.append(("honest", CLIPS / r["file"], r["text"]))
    for r in rows(CLIPS / "errors" / "manifest.tsv"):
        p = CLIPS / "errors" / r["file"]
        kind, cls = r.get("kind") or "", r.get("class") or "accept"
        group = ("nonanswer" if kind in NONANSWER
                 else "near_miss" if cls == "accept" else "wrong_mt")
        out.append((group, p, r["text"]))
    neg = CLIPS / "negatives" / "manifest.tsv"
    if neg.exists():
        src_text = {x["file"]: x["text"] for x in rows(CLIPS / "manifest.tsv")}
        real = [t for t in (norm(x) for x in dialogue.accepted_lines()) if t]
        for i, r in enumerate(rows(neg)):
            kind, src = r.get("kind") or "", r.get("source") or ""
            # `quiet` is an attenuated copy of a correct answer, built when a quiet clip was
            # assumed to be a bad one. The recogniser normalises each mel bin over the clip,
            # so a uniform gain change cancels: these are the same correct answer and
            # accepting them is right. Charging for it would be fitting to a wrong label.
            group = ("honest_quiet" if kind == "quiet"
                     else "reversed" if kind == "reversed" else "noise")
            text = src_text.get(src) if src.startswith("me_") else None
            # Silence and hiss were built from no sentence, so they need a stand-in target.
            # Rotating through the deck stops the result being an artefact of one line's
            # length — a short line is much easier for silence to beat than a long one.
            out.append((group, CLIPS / "negatives" / r["file"],
                        text or real[i % len(real)]))
    return out


def build_cache():
    logprobs_for, vocab, blank, space = load("frontend/stt")
    deck = [x for x in (norm(x) for x in dialogue.accepted_lines()) if x]
    ids = {line: encode(line, vocab, space) for line in deck}
    recs = []
    items = gather()
    fallback = deck[0]
    for i, (group, path, text) in enumerate(items):
        wave = read_clip(path)
        if wave is None:
            continue
        want = norm(text) if text else fallback
        post = logprobs_for(wave)
        tid = ids.get(want) or encode(want, vocab, space)
        if not tid or len(tid) > len(post):
            # No room to align the target: the app refuses this, so it is a refusal at
            # every setting rather than a clip dropped from the denominator.
            recs.append({"group": group, "conf": 0.0, "target": -1e9, "rivals": []})
            continue
        rivals = [rank_score(post, ids[l], blank) for l in deck
                  if l != want and ids[l] and len(ids[l]) <= len(post)]
        recs.append({"group": group, "conf": confidence(post, tid, blank),
                     "target": rank_score(post, tid, blank), "rivals": rivals})
        if (i + 1) % 100 == 0:
            print(f"  scored {i + 1}/{len(items)}", flush=True)
    CACHE.write_text(json.dumps(recs))
    return recs


recs = json.loads(CACHE.read_text()) if CACHE.exists() else build_cache()
print(f"cached {len(recs)} clips")

GROUPS = ["honest", "honest_quiet", "near_miss", "wrong_mt",
          "nonanswer", "noise", "reversed"]


def accept_rates(floor, margin):
    """Share of each group accepted, averaged over independent draws of the field."""
    per = {g: [0.0, 0] for g in GROUPS}
    for rec in recs:
        if rec["group"] not in per:
            continue
        hits = 0
        for seed in SEEDS:
            rng = random.Random(seed)
            pool = rec["rivals"]
            drawn = rng.sample(pool, min(FIELD, len(pool))) if pool else []
            best = max(drawn) if drawn else -float("inf")
            hits += int(rec["conf"] >= floor and rec["target"] > best + margin)
        per[rec["group"]][0] += hits / len(SEEDS)
        per[rec["group"]][1] += 1
    return {g: (v[0] / v[1] if v[1] else 0.0, v[1]) for g, v in per.items()}


FLOORS = [0.00, 0.05, 0.10, 0.15, 0.25, 0.35, 0.45, 0.55]
MARGINS = [-0.10, -0.05, 0.00, 0.02, 0.05, 0.10]

print(f"\nfield of {FIELD} sampled rivals, averaged over {len(SEEDS)} draws\n")
grid = {}
for floor in FLOORS:
    for margin in MARGINS:
        grid[(floor, margin)] = accept_rates(floor, margin)

n = accept_rates(0.15, 0.02)
print("  set sizes: " + ", ".join(f"{g} {n[g][1]}" for g in GROUPS) + "\n")

print("  honest accepted (want high) by floor x margin")
print("  floor  " + "  ".join(f"{m:+.2f}" for m in MARGINS))
for floor in FLOORS:
    print(f"  {floor:5.2f}  " + "  ".join(
        f"{grid[(floor, m)]['honest'][0] * 100:5.0f}" for m in MARGINS))

print("\n  non-answers admitted (want zero) by floor x margin")
print("  floor  " + "  ".join(f"{m:+.2f}" for m in MARGINS))
for floor in FLOORS:
    print(f"  {floor:5.2f}  " + "  ".join(
        f"{grid[(floor, m)]['nonanswer'][0] * 100:5.0f}" for m in MARGINS))

print("\n  silence and hiss admitted (want zero) by floor x margin")
print("  floor  " + "  ".join(f"{m:+.2f}" for m in MARGINS))
for floor in FLOORS:
    print(f"  {floor:5.2f}  " + "  ".join(
        f"{grid[(floor, m)]['noise'][0] * 100:5.0f}" for m in MARGINS))

print("\n  objective = honest - lambda * (non-answers + hiss/silence admitted)")
for lam in (1.0, 3.0, 10.0):
    best = max(grid, key=lambda k: grid[k]["honest"][0]
               - lam * (grid[k]["nonanswer"][0] + grid[k]["noise"][0]))
    r = grid[best]
    print(f"  lambda {lam:5.1f} -> floor {best[0]:.2f}, margin {best[1]:+.2f}  "
          f"honest {r['honest'][0] * 100:.0f}%  near-miss {r['near_miss'][0] * 100:.0f}%  "
          f"wrong-mt {r['wrong_mt'][0] * 100:.0f}%  "
          f"non-answers {r['nonanswer'][0] * 100:.0f}%  "
          f"hiss/silence {r['noise'][0] * 100:.0f}%  "
          f"reversed {r['reversed'][0] * 100:.0f}%  "
          f"quiet-correct {r['honest_quiet'][0] * 100:.0f}%")

dep = grid[(0.15, 0.02)]
print(f"\n  deployed (0.15, +0.02)      honest {dep['honest'][0] * 100:.0f}%  "
      f"near-miss {dep['near_miss'][0] * 100:.0f}%  "
      f"wrong-mt {dep['wrong_mt'][0] * 100:.0f}%  "
      f"non-answers {dep['nonanswer'][0] * 100:.0f}%  "
      f"hiss/silence {dep['noise'][0] * 100:.0f}%  "
      f"reversed {dep['reversed'][0] * 100:.0f}%  "
      f"quiet-correct {dep['honest_quiet'][0] * 100:.0f}%")

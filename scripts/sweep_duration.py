"""Sweep DUR_WEIGHT — the joint sweep the code says was never possible.

`constrained_ctc` carries a note that DUR_FRAMES, DUR_SD_SLOPE and the rest ship switched
off because pricing them needs the eval clips, the negatives and a loaded model at once.
All three are available now, and the diagnostic that motivated this says the weight is the
thing to look at: when a correct-but-imperfect answer loses the field, the rival that beat
it is *longer* than the target (median length ratio 1.38, only 29% shorter), and four deck
lines cause 61 of 108 losses. A handful of long lines are absorbing arbitrary audio, which
is what an under-weighted length penalty looks like.

`rank_score` is `confidence + DUR_WEIGHT * duration_prior(tokens, frames)`, so caching
confidence and token count per rival plus frames per clip makes every weight free.
"""
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")

from constrained_ctc import (DUR_INTERCEPT, DUR_SD, DUR_SLOPE, confidence, encode, load)

from backend import dialogue
from backend import text as mtext
from make_negatives import read_clip

CLIPS = Path("data/eval_clips")
CACHE = Path("data/eval_clips/duration_cache.json")
FIELD, SEEDS = 24, (1, 2, 3, 4, 5)
FLOOR, MARGIN = 0.15, 0.02
NONANSWER = ("silence", "filler", "english", "partial", "offtopic")
GROUPS = ["honest", "near_miss", "wrong_mt", "nonanswer", "noise", "reversed"]


def norm(s):
    return mtext.normalise(s).lower().strip()


def rows(p):
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def prior(tokens, frames):
    return -0.5 * ((frames - (DUR_INTERCEPT + DUR_SLOPE * tokens)) / DUR_SD) ** 2


def gather():
    out = []
    for r in rows(CLIPS / "manifest.tsv"):
        if r["file"].startswith("me_"):
            out.append(("honest", CLIPS / r["file"], r["text"]))
    for r in rows(CLIPS / "errors" / "manifest.tsv"):
        kind, cls = r.get("kind") or "", r.get("class") or "accept"
        out.append((("nonanswer" if kind in NONANSWER
                     else "near_miss" if cls == "accept" else "wrong_mt"),
                    CLIPS / "errors" / r["file"], r["text"]))
    neg = CLIPS / "negatives" / "manifest.tsv"
    if neg.exists():
        src = {x["file"]: x["text"] for x in rows(CLIPS / "manifest.tsv")}
        real = [x for x in (norm(v) for v in dialogue.accepted_lines()) if x]
        for i, r in enumerate(rows(neg)):
            kind, s = r.get("kind") or "", r.get("source") or ""
            out.append(("reversed" if kind == "reversed" else "noise",
                        CLIPS / "negatives" / r["file"],
                        (src.get(s) if s.startswith("me_") else None) or real[i % len(real)]))
    return out


def build():
    logprobs_for, vocab, blank, space = load("frontend/stt")
    deck = [x for x in (norm(x) for x in dialogue.accepted_lines()) if x]
    ids = {l: encode(l, vocab, space) for l in deck}
    recs, items = [], gather()
    for i, (group, path, text) in enumerate(items):
        wave = read_clip(path)
        if wave is None:
            continue
        want = norm(text) if text else deck[0]
        post = logprobs_for(wave)
        frames = len(post)
        tid = ids.get(want) or encode(want, vocab, space)
        if not tid or len(tid) > frames:
            recs.append({"group": group, "frames": frames, "conf": 0.0,
                         "tokens": 0, "rivals": []})
            continue
        recs.append({
            "group": group, "frames": frames,
            "conf": confidence(post, tid, blank), "tokens": len(tid),
            "rivals": [[len(ids[l]), confidence(post, ids[l], blank)]
                       for l in deck if l != want and ids[l] and len(ids[l]) <= frames],
        })
        if (i + 1) % 100 == 0:
            print(f"  scored {i + 1}/{len(items)}", flush=True)
    CACHE.write_text(json.dumps(recs))
    return recs


def main() -> int:
    """Everything below runs a sweep, so it lives behind a call rather than at
    import time. `test_script_imports` imports every script in this directory: a
    module that loads a model and reads a cache while being imported fails wherever
    onnxruntime or the audio is absent, which is CI.
    """
    recs = json.loads(CACHE.read_text()) if CACHE.exists() else build()
    print(f"cached {len(recs)} clips\n")


    def rates(weight):
        per = {g: [0.0, 0] for g in GROUPS}
        for rec in recs:
            if rec["group"] not in per:
                continue
            f = rec["frames"]
            target = rec["conf"] + weight * prior(rec["tokens"], f)
            scored = [c + weight * prior(t, f) for t, c in rec["rivals"]]
            hits = 0
            for seed in SEEDS:
                rng = random.Random(seed)
                drawn = rng.sample(scored, min(FIELD, len(scored))) if scored else []
                best = max(drawn) if drawn else -float("inf")
                hits += int(rec["conf"] >= FLOOR and target > best + MARGIN)
            per[rec["group"]][0] += hits / len(SEEDS)
            per[rec["group"]][1] += 1
        return {g: (v[0] / v[1] if v[1] else 0.0, v[1]) for g, v in per.items()}


    WEIGHTS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0]
    print(f"floor {FLOOR}, margin {MARGIN:+.2f}, field of {FIELD}, {len(SEEDS)} draws")
    print("\n  weight" + "".join(f"{g:>11}" for g in GROUPS))
    out = {}
    for w in WEIGHTS:
        r = out[w] = rates(w)
        mark = "  <- deployed" if w == 0.1 else ""
        print(f"  {w:5.2f} " + "".join(f"{r[g][0] * 100:10.0f}%" for g in GROUPS) + mark)

    print("\n  want high: honest, near_miss.  want low: nonanswer, noise, reversed.")
    print("  relaxed about: wrong_mt.\n")
    for lam in (1.0, 3.0):
        best = max(out, key=lambda w: out[w]["honest"][0] + out[w]["near_miss"][0]
                   - lam * (out[w]["nonanswer"][0] + out[w]["noise"][0]))
        r = out[best]
        print(f"  lambda {lam:4.1f} -> weight {best:.2f}: honest {r['honest'][0] * 100:.0f}%, "
              f"near-miss {r['near_miss'][0] * 100:.0f}%, "
              f"non-answers {r['nonanswer'][0] * 100:.0f}%, noise {r['noise'][0] * 100:.0f}%")
    print(f"\n  group sizes: " + ", ".join(f"{g} {out[0.1][g][1]}" for g in GROUPS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

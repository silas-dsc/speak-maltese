#!/usr/bin/env python3
"""Price a change to the acceptance rule the way every constant in it was priced.

`MIN_CONFIDENCE`, `DUR_WEIGHT`, `MIN_MARGIN` were each chosen by sweeping them against
three sets at once: a learner's own recordings, which must be accepted; the same audio
paired with the wrong line, which must not be; and the negatives, which must not be
either. That sweep was run by hand and never committed, which is why `DUR_FRAMES`,
`DUR_SD_SLOPE`, `MARGIN_SIGMAS` and `FIELD_LOCAL` all ship switched off — there was no
way to price them. This is that harness.

    python scripts/make_negatives.py                        # once
    python scripts/sweep_grader.py --models frontend/stt    # collect, then sweep
    python scripts/sweep_grader.py --grid lambda            # re-sweep, no model needed

**The expensive part happens once.** Scoring a field of alternatives against every clip
means a CTC forward pass per hypothesis; varying λ afterwards is arithmetic on numbers
already computed. So `collect` stores, per clip and per hypothesis, the sequence
log-likelihood, the greedy path, the frame counts both ways and the token count — and
every parameter combination is then a pure pass over that cache. A grid of two hundred
settings costs the same as one.

The exception is `FIELD_LOCAL`, which changes *which* alternatives are scored rather than
how they are compared, so it cannot be applied after the fact. The cache therefore holds
a superset — every line the clip's own scene accepts, plus a seeded sample of the whole
script — and the mix is chosen when deciding.

**What this cannot tell you.** The field is sampled, and the README measures a four-point
spread across seeds with the prior on. So a difference of one clip in twenty-five is not
a result, and a rule is only worth moving on if it wins across `--seeds`. The published
percentages were also measured against a negative set whose composition was never
recorded (see `make_negatives.py`), so compare candidates against the *current* rule as
this harness measures it, never against the numbers in the README.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import dialogue, text as mtext  # noqa: E402
from backend.config import DATA_DIR  # noqa: E402
from constrained_ctc import (  # noqa: E402
    DUR_INTERCEPT, DUR_SLOPE, DUR_SD, DUR_WEIGHT, ctc_logp, greedy_logp,
    speech_frames,
)

CLIPS = DATA_DIR / "eval_clips"
CACHE = CLIPS / "sweep_cache.json"

# The deployed rule, as it stands. Everything is measured against this row.
DEPLOYED = {
    "dur_weight": DUR_WEIGHT,
    "dur_sd_slope": 0.0,
    "dur_frames": "total",
    "floor": 0.15,   # see MIN_CONFIDENCE in frontend/app.js
    "min_margin": 0.02,
    "margin_sigmas": 0.0,
    "field_local": 0.0,
    "field_size": 24,
    # Which duration constants the prior uses. See CALIBRATIONS: the deployed set is
    # measurably miscalibrated, and that is what makes it work.
    "calibration": "deployed",
}

# How many wrong lines each clip is paired with. 25 clips x 7 is the published 175.
WRONG_PER_CLIP = 7


# ── The decision, as arithmetic over precomputed scores ─────────────────────
# Pure, so it can be checked without a model, a microphone or a single audio file.

def confidence(logp: float, greedy: float, frames_total: int) -> float:
    """Per-frame likelihood ratio against the model's own best path.

    Always on the *total* frame count, whatever `dur_frames` says. The floor asks "is
    there anything here at all", and a floor whose denominator moved with the recording's
    padding would not be one."""
    return float(np.exp((logp - greedy) / max(1, frames_total)))


# The deployed constants and the ones a refit on 63,114 distillation passes gives.
# They are not a small correction to each other: the intercept is 93.38 against 28.28 and
# the sd is 27.11 against 13.27, so the deployed prior scores z at roughly twice the true
# scale and z-squared at four times it. That is why it works — it charges a short rival
# +2.274 confidence units where a calibrated prior charges +0.131, and only the blunt
# version reverses the failures the prior was added for. Kept as presets rather than a
# correction, because the choice between them is a choice about what the term is *for*.
CALIBRATIONS = {
    "deployed": (28.28, 1.8794, 13.27, 0.0),
    "refit": (93.38, 1.6238, 27.11, 0.0),
    "refit-sd": (93.38, 1.6238, 24.595, -0.0251),
}


def duration_prior(tokens: int, frames: int, sd_slope: float,
                   calibration: str = "deployed") -> float:
    intercept, slope, sd0, sd_slope0 = CALIBRATIONS[calibration]
    # An explicit dur_sd_slope overrides the preset's own, so the existing sdslope grid
    # keeps meaning what it did.
    sd = max(1e-6, sd0 + (sd_slope if sd_slope else sd_slope0) * tokens)
    return -0.5 * ((frames - (intercept + slope * tokens)) / sd) ** 2


def rank_of(entry: dict, rec: dict, params: dict) -> float:
    frames = (rec["frames_speech"] if params["dur_frames"] == "speech"
              else rec["frames_total"])
    return (confidence(entry["logp"], rec["greedy"], rec["frames_total"])
            + params["dur_weight"] * duration_prior(
                entry["tokens"], frames, params["dur_sd_slope"],
                params["calibration"]))


def choose_field(rec: dict, params: dict, rng) -> list[dict]:
    """The alternatives this rule would actually rank against.

    `field_local` decides the share drawn from the clip's own scene; the rest tops up
    from the whole script, so a thin scene never shrinks the field."""
    size = params["field_size"]
    want_local = int(round(params["field_local"] * size))
    local = [e for e in rec["field"] if e.get("local")]
    glob = [e for e in rec["field"] if not e.get("local")]
    rng.shuffle(local)
    rng.shuffle(glob)
    picked = local[:want_local]
    picked += glob[:max(0, size - len(picked))]
    if len(picked) < size:
        # Top up from the local entries the share did not already claim. Indexing `local`
        # by how many are picked instead would skip the ones between `want_local` and
        # that count, and hand back a field short of `size` — and ranking against fewer
        # alternatives is a different change wearing the same clothes.
        picked += local[want_local:][:size - len(picked)]
    return picked[:size]


def decide(rec: dict, params: dict, seed: int = 0) -> dict:
    """Would the app mark this audio as the line it was asked for?

    Mirrors `app.js`: the field is compared on `rank`, the floor tests `confidence`, and
    the target has to clear the runner-up by a margin that never drops below
    `min_margin`."""
    rng = np.random.default_rng(seed)
    conf = confidence(rec["target"]["logp"], rec["greedy"], rec["frames_total"])
    field = choose_field(rec, params, rng)
    target_rank = rank_of(rec["target"], rec, params)

    passes_floor = conf >= params["floor"]
    if not field:
        # No field to rank against: the floor is all there is, which is what the FastAPI
        # dev build actually does.
        return {"accepted": passes_floor, "confidence": conf, "rank": target_rank,
                "runner_up": None, "spread": 0.0, "need": 0.0,
                "clear": True, "passes_floor": passes_floor,
                "reason": "floor only" if passes_floor else "under the floor"}

    ranks = np.array([rank_of(e, rec, params) for e in field], dtype=float)
    runner_up = float(ranks.max())
    spread = float(ranks.std(ddof=1)) if len(ranks) > 1 else 0.0
    need = max(params["min_margin"], params["margin_sigmas"] * spread)
    clear = bool(target_rank > runner_up + need)

    # Both causes, not the first one found. Which of the two a candidate rule is losing
    # accepts to is the whole diagnostic — a floor that is too high and a field that is
    # too hard want opposite fixes, and collapsing them to one label hides that.
    if clear and passes_floor:
        reason = "accepted"
    elif not clear and not passes_floor:
        reason = "lost the field and under the floor"
    else:
        reason = "lost the field" if not clear else "under the floor"
    return {"accepted": bool(clear and passes_floor), "confidence": conf,
            "rank": target_rank, "runner_up": runner_up, "spread": spread,
            "need": need, "clear": clear, "passes_floor": passes_floor,
            "reason": reason}


def measure(records: list[dict], params: dict, seeds: int = 3) -> dict:
    """Accept rates by what the record is, averaged over field draws."""
    tally: dict[str, list[int]] = {}
    totals: dict[str, int] = {}
    for seed in range(seeds):
        hits: dict[str, int] = {}
        for rec in records:
            kind = rec["kind"]
            hits[kind] = hits.get(kind, 0) + int(decide(rec, params, seed)["accepted"])
            if seed == 0:
                totals[kind] = totals.get(kind, 0) + 1
        for kind, n in hits.items():
            tally.setdefault(kind, []).append(n)
    return {kind: {"accepted": float(np.mean(v)), "of": totals[kind],
                   "rate": float(np.mean(v)) / max(1, totals[kind]),
                   "spread": int(max(v) - min(v))}
            for kind, v in tally.items()}


# ── Collecting the scores, which is the part that needs the model ───────────

def scene_for(flat: str) -> str | None:
    """Which dialogue accepts this line, if any. `FIELD_LOCAL` needs it; nothing else
    does, and an eval manifest does not record it."""
    for d in dialogue.all_dialogues():
        for line in dialogue.answers_in(d["id"]):
            if mtext.normalise(line).lower().strip() == flat:
                return d["id"]
    return None


def encode(flat: str, vocab: dict, space: str) -> list[int]:
    return [vocab[space if ch == " " else ch] for ch in flat
            if (space if ch == " " else ch) in vocab]


def collect(model: str, clips_dir: Path, seed: int, field_pool: int) -> list[dict]:
    """One pass over the audio, scoring every hypothesis any rule might want."""
    from make_negatives import read_clip, read_negatives
    from constrained_ctc import load

    manifest = clips_dir / "manifest.tsv"
    if not manifest.exists():
        print(f"no manifest at {manifest} — record clips first:\n"
              f"  python scripts/compare_stt.py --record 25", file=sys.stderr)
        return []
    with manifest.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t")
                if (r.get("text") or "").strip()]
    rows = [r for r in rows if r["file"].startswith("me_")] or rows
    if not rows:
        return []

    negatives = read_negatives()
    if not negatives:
        print("no negatives — run scripts/make_negatives.py first", file=sys.stderr)

    logprobs_for, vocab, blank, space = load(model)
    # Accepted answers, not every speakable line: see `dialogue.accepted_lines`.
    every = [mtext.normalise(x).lower().strip() for x in dialogue.accepted_lines()]
    every = [x for x in every if x]
    rng = np.random.default_rng(seed)

    records: list[dict] = []

    def score(wave, target_flat: str, kind: str, name: str) -> dict | None:
        post = logprobs_for(wave)
        greedy = greedy_logp(post)
        target_ids = encode(target_flat, vocab, space)
        if not target_ids:
            return None
        scene = scene_for(target_flat)
        local_lines = ([mtext.normalise(x).lower().strip()
                        for x in dialogue.answers_in(scene)] if scene else [])
        pool = [x for x in every if x != target_flat]
        rng.shuffle(pool)
        chosen = pool[:field_pool]
        field = []
        for line in dict.fromkeys(list(local_lines) + chosen):
            if line == target_flat:
                continue
            ids = encode(line, vocab, space)
            if not ids or len(ids) > len(post):
                continue                       # unalignable: no hypothesis at all
            field.append({"tokens": len(ids), "local": line in local_lines,
                          "logp": float(ctc_logp(post, ids, blank))})
        return {"clip": name, "kind": kind, "frames_total": int(len(post)),
                "frames_speech": int(speech_frames(wave)),
                "greedy": float(greedy),
                "target": {"tokens": len(target_ids),
                           "logp": float(ctc_logp(post, target_ids, blank))},
                "field": field}

    for i, row in enumerate(rows, 1):
        wave = read_clip(clips_dir / row["file"])
        if wave is None:
            continue
        flat = mtext.normalise(row["text"]).lower().strip()
        rec = score(wave, flat, "learner", row["file"])
        if rec:
            records.append(rec)
        # The same audio against lines it is not. Rejecting these is the other half of
        # the job, and a rule that only ever sees correct pairings looks perfect.
        others = [r for r in rows if r["file"] != row["file"]]
        rng.shuffle(others)
        for other in others[:WRONG_PER_CLIP]:
            wrong = mtext.normalise(other["text"]).lower().strip()
            if wrong == flat:
                continue
            rec = score(wave, wrong, "wrong-line", f"{row['file']}~{other['file']}")
            if rec:
                records.append(rec)
        print(f"  {i}/{len(rows)} {row['file']}", flush=True)

    # Negatives get paired with a real line, because the app always asks for one.
    for i, neg in enumerate(negatives):
        wave = read_clip(CLIPS / "negatives" / neg["file"])
        if wave is None or not wave.size:
            continue
        # Cycled rather than fixed: the prior charges by length, so grading all 90
        # negatives against one line would measure that line's token count as much as
        # it measures the rule.
        flat = mtext.normalise(rows[i % len(rows)]["text"]).lower().strip()
        rec = score(wave, flat, neg["kind"], neg["file"])
        if rec:
            records.append(rec)

    return records


# ── Grids ──────────────────────────────────────────────────────────────────

GRIDS = {
    "lambda": ("dur_weight", [0.0, 0.05, 0.1, 0.15, 0.3]),
    "floor": ("floor", [0.0, 0.15, 0.2, 0.35, 0.45, 0.55, 0.65]),
    "sigmas": ("margin_sigmas", [0.0, 0.25, 0.5, 1.0, 2.0]),
    # Loosening, which nobody had swept: every value of `sigmas` above zero makes the
    # margin *stricter*, so the direction that could buy accepts had never been priced.
    # 0.0 is as loose as this knob goes — `need` is a max against zero, so a negative
    # min_margin is clamped away. Accepting a target the model ranks second is a
    # different rule, not a smaller number; `--probe-loss` prices that separately.
    "margin": ("min_margin", [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]),
    "calib": ("calibration", ["deployed", "refit", "refit-sd"]),
    "local": ("field_local", [0.0, 0.25, 0.5, 0.75]),
    "frames": ("dur_frames", ["total", "speech"]),
    "sdslope": ("dur_sd_slope", [0.0, 0.25, 0.5, 1.0]),
    "field": ("field_size", [12, 24, 48, 96]),
}

ORDER = ("learner", "wrong-line", "silence", "hiss", "quiet", "reversed")


def report(records: list[dict], axis: str, values: list, seeds: int) -> None:
    print(f"\nsweeping {axis} — {len(records)} scored pairings, {seeds} field draws")
    header = f"  {axis:>12} " + " ".join(f"{k:>12}" for k in ORDER)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for value in values:
        params = dict(DEPLOYED)
        params[axis] = value
        got = measure(records, params, seeds)
        cells = []
        for kind in ORDER:
            if kind in got:
                g = got[kind]
                cells.append(f"{g['accepted']:.0f}/{g['of']} ({g['rate'] * 100:.0f}%)")
            else:
                cells.append("-")
        mark = "  <- deployed" if value == DEPLOYED[axis] else ""
        print(f"  {str(value):>12} " + " ".join(f"{c:>12}" for c in cells) + mark)
    print("\n  `learner` wants to be high; everything else wants to be zero. A"
          "\n  one-clip difference is inside the seed spread — see the module note.")


def probe_loss(records: list[dict], offsets: list[float], seeds: int = 5) -> None:
    """Price accepting a target the model ranks second — leniency the app cannot express.

    `min_margin` cannot go below zero: `need` is a max against `margin_sigmas * spread`,
    so a negative value is clamped away, and sweeping it down to 0.0 buys no accepts at
    all. The reason is in the deficits — the learner clips that lose, lose by 0.08 to 0.16,
    which is four to eight times the margin. Reaching them means accepting a line the
    model ranks below a rival, which is a different rule rather than a smaller number, so
    it is priced here rather than offered as a setting.

    A negative offset is the honest form of "be more lenient": how much of a loss are we
    willing to forgive, and what comes through the door with it."""
    print(f"\nprobing a forgiven loss — {len(records)} scored pairings, {seeds} field draws")
    head = (f"  {'forgive':>12} {'learner':>12} {'wrong-line':>12} {'silence':>12} "
            f"{'hiss':>12} {'quiet':>12} {'reversed':>12}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for off in offsets:
        params = dict(DEPLOYED)
        # A negative `min_margin` survives here because this bypasses the max in decide().
        hits: dict[str, list[int]] = {}
        totals: dict[str, int] = {}
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            got: dict[str, int] = {}
            for rec in records:
                kind = rec["kind"]
                conf = confidence(rec["target"]["logp"], rec["greedy"],
                                  rec["frames_total"])
                field = choose_field(rec, params, rng)
                target_rank = rank_of(rec["target"], rec, params)
                if field:
                    runner_up = max(rank_of(e, rec, params) for e in field)
                    clear = target_rank > runner_up + off
                else:
                    clear = True
                got[kind] = got.get(kind, 0) + int(clear and conf >= params["floor"])
                totals[kind] = totals.get(kind, 0) + 1 if seed == 0 else totals[kind]
            for kind, n in got.items():
                hits.setdefault(kind, []).append(n)
        cells = []
        for kind in ("learner", "wrong-line", "silence", "hiss", "quiet", "reversed"):
            if kind not in hits:
                cells.append(f"{'-':>12}")
                continue
            mean = float(np.mean(hits[kind]))
            of = totals[kind]
            cells.append(f"{mean:.0f}/{of} ({mean / max(1, of) * 100:.0f}%)".rjust(12))
        mark = "  <- deployed" if off == DEPLOYED["min_margin"] else ""
        print(f"  {off:>12.3f} " + " ".join(cells) + mark)
    print("\n  A negative number forgives a loss of that size. The learner clips that")
    print("  lose need 0.08 to 0.16 forgiven, so read what the negatives do at those")
    print("  offsets before reading the accept rate as good news.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None,
                    help="a directory holding model.onnx/vocab.txt/config.json; "
                         "omit to re-sweep the cache")
    ap.add_argument("--clips-dir", type=Path, default=CLIPS)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--grid", default="lambda", choices=sorted(GRIDS) + ["all"])
    ap.add_argument("--seeds", type=int, default=3,
                    help="field draws to average over; the spread matters as much as "
                         "the mean")
    ap.add_argument("--field-pool", type=int, default=64,
                    help="global alternatives cached per clip, so field_size and "
                         "field_local can vary without rescoring")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--probe-loss", action="store_true",
                    help="price forgiving a target that the model ranks second")
    args = ap.parse_args()

    if args.models:
        records = collect(args.models, args.clips_dir, args.seed, args.field_pool)
        if not records:
            return 2
        args.cache.write_text(json.dumps(records), encoding="utf-8")
        print(f"\n{len(records)} pairings cached → {args.cache}")
    else:
        if not args.cache.exists():
            print(f"no cache at {args.cache} — run once with --models", file=sys.stderr)
            return 2
        records = json.loads(args.cache.read_text(encoding="utf-8"))

    if args.probe_loss:
        probe_loss(records, [-0.20, -0.16, -0.12, -0.08, -0.04, 0.0, 0.02], args.seeds)
        return 0

    axes = sorted(GRIDS) if args.grid == "all" else [args.grid]
    for name in axes:
        axis, values = GRIDS[name]
        report(records, axis, values, args.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

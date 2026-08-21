"""The JavaScript GOP and the Python GOP must be the same function.

Two implementations of a log-space forward-backward will agree on easy inputs and diverge
on the ones that matter — an unalignable target, a repeated token needing a blank between,
a frame no path can reach. This repository has already been bitten by a JS/Python split at
the fourth decimal (see `round4`), and a per-sound score that disagreed between the app and
the harness would be worse than none: the tuning would be done against numbers the learner
never sees.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

np = pytest.importorskip("numpy")
node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node is not installed")

DRIVER = r"""
import { readFileSync } from 'node:fs';
import { occupancy, tokenGop, gopScore, worstSound,
         GOP_IGNORE, GOP_MIN } from '../frontend/nanostt.js';

const cfg = JSON.parse(readFileSync(process.argv[2], 'utf-8'));
const LP = Float64Array.from(cfg.logprobs);
const V = cfg.vocab, F = LP.length / V, BLK = cfg.blank;

const out = { gopMin: GOP_MIN, ignore: [...GOP_IGNORE].sort() , cases: {} };
for (const [name, c] of Object.entries(cfg.cases)) {
  const occ = occupancy(LP, F, V, c.ids, BLK);
  out.cases[name] = {
    gop: Array.from(tokenGop(LP, F, V, c.ids, BLK)),
    mass: occ ? c.ids.map((_, i) => {
      let m = 0; for (let t = 0; t < F; t += 1) m += occ[i * F + t]; return m;
    }) : null,
    score: gopScore(LP, F, V, c.ids, c.toks, BLK),
    worst: worstSound(LP, F, V, c.ids, c.toks, BLK),
  };
}
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def both():
    from gop import GOP_IGNORE, GOP_MIN, gop_score, occupancy, token_gop, worst_sound

    rng = np.random.default_rng(11)
    V, F, blank = 5, 14, 4
    raw = rng.normal(0.0, 2.0, (F, V))
    post = raw - np.log(np.exp(raw).sum(axis=1, keepdims=True))

    cases = {
        # An ordinary target.
        "plain": {"ids": [0, 1, 2], "toks": ["a", "b", "c"]},
        # A repeat, which needs a blank between the two emissions.
        "repeat": {"ids": [0, 0, 1], "toks": ["a", "a", "b"]},
        # Every token ignored: the score must be "no opinion", not a number.
        "allIgnored": {"ids": [0, 1], "toks": ["q", "g"]},
        # Exactly as long as the audio: the tightest alignment that still exists.
        "exact": {"ids": list(range(4)) * 3 + [0, 1], "toks": list("abcd") * 3 + ["a", "b"]},
        # One token, so the extended sequence is the shortest it can be. Deliberately a
        # token that is *not* a blind spot — `d` is, and picking it here would test the
        # no-opinion path a second time instead of the single-token alignment.
        "single": {"ids": [2], "toks": ["c"]},
    }
    cfg = {"logprobs": post.ravel().tolist(), "vocab": V, "blank": blank, "cases": cases}

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "cfg.json"
        cfg_path.write_text(json.dumps(cfg))
        driver = ROOT / "tests" / "_gop_driver.mjs"
        driver.write_text(DRIVER)
        try:
            proc = subprocess.run([node, str(driver), str(cfg_path)], cwd=ROOT,
                                  capture_output=True, text=True, timeout=180)
        finally:
            driver.unlink(missing_ok=True)
    if proc.returncode != 0:
        pytest.fail(f"node driver failed:\n{proc.stderr}")
    js = json.loads(proc.stdout)

    py = {}
    for name, c in cases.items():
        gam = occupancy(post, c["ids"], blank)
        py[name] = {
            "gop": [float(x) for x in token_gop(post, c["ids"], blank)],
            "mass": [float(x) for x in gam.sum(axis=1)],
            "score": gop_score(post, c["ids"], c["toks"], blank),
            "worst": worst_sound(post, c["ids"], c["toks"], blank),
        }
    return js, py, {"ignore": sorted(GOP_IGNORE), "min": GOP_MIN}


def test_the_two_implementations_agree_per_token(both):
    js, py, _ = both
    for name in py:
        a, b = js["cases"][name]["gop"], py[name]["gop"]
        assert len(a) == len(b), f"{name}: token count differs"
        for i, (x, y) in enumerate(zip(a, b)):
            assert abs(x - y) < 1e-9, f"{name} token {i}: js {x} vs python {y}"


def test_the_occupancies_agree_and_each_token_owns_a_frame(both):
    js, py, _ = both
    for name in py:
        a, b = js["cases"][name]["mass"], py[name]["mass"]
        for i, (x, y) in enumerate(zip(a, b)):
            assert abs(x - y) < 1e-9, f"{name} token {i}: mass js {x} vs python {y}"
            # CTC emits every target once per path, so its occupancy sums to at least one
            # frame. A recurrence with the skip rule wrong still produces plausible
            # scores; this is what catches it.
            assert x > 0.99, f"{name} token {i} owns no frames ({x})"


def test_a_line_of_blind_spots_has_no_opinion(both):
    js, py, _ = both
    import math

    assert math.isnan(py["allIgnored"]["score"])
    assert js["cases"]["allIgnored"]["score"] is None, (
        "JSON has no NaN, so JavaScript must send null — a number here would be read as a "
        "verdict on a line the model cannot judge")
    assert js["cases"]["allIgnored"]["worst"] is None
    assert py["allIgnored"]["worst"] is None


def test_the_scores_and_the_blamed_sound_agree(both):
    js, py, _ = both
    for name in ("plain", "repeat", "exact", "single"):
        assert abs(js["cases"][name]["score"] - py[name]["score"]) < 1e-9, name
        jw, pw = js["cases"][name]["worst"], py[name]["worst"]
        assert jw["token"] == pw["token"] and jw["index"] == pw["index"], (
            f"{name}: the two engines blame different sounds — {jw} vs {pw}")


def test_the_constants_have_not_drifted(both):
    js, _py, pyc = both
    assert js["gopMin"] == pyc["min"], "GOP_MIN differs between the app and the harness"
    assert js["ignore"] == pyc["ignore"], (
        "the ignored graphemes differ, so the two engines are scoring different sounds")


def test_the_threshold_credits_no_backwards_speech():
    """`GOP_MIN` is the whole reason the near-miss verdict is safe.

    The floor cannot separate a learner who nearly said the line from audio that is not the
    line: it admits 67% of time-reversed clips. The verdict rests entirely on GOP refusing
    them, so a change to the threshold that let one through would silently start
    congratulating people on backwards speech. Guarded against the measured set rather than
    against an opinion."""
    import json

    from gop import GOP_MIN, score_tokens

    cache = ROOT / "data" / "eval_clips" / "gop_cache.json"
    if not cache.exists():
        pytest.skip("no gop cache; run scripts/gop.py --models frontend/stt")
    rows = json.loads(cache.read_text())

    # The shipped definition, not a copy of it: this test previously restated the formula
    # and so kept passing against a rule that no longer existed.
    def score(row):
        return score_tokens(row["gop"], row["tokens"])

    credited = {}
    for row in rows:
        s = score(row)
        if row["kind"] != "learner" and not np.isnan(s) and s >= GOP_MIN:
            credited.setdefault(row["kind"], []).append(row["clip"])

    assert "reversed" not in credited, (
        f"GOP_MIN={GOP_MIN} credits backwards speech: {credited.get('reversed')[:4]}")
    assert "hiss" not in credited and "silence" not in credited, (
        f"GOP_MIN={GOP_MIN} credits noise: { {k: v[:3] for k, v in credited.items()} }")

    learner = [score(r) for r in rows if r["kind"] == "learner"]
    kept = [s for s in learner if not np.isnan(s) and s >= GOP_MIN]
    # The other half of the trade: a threshold safe because it refuses everything would
    # make the verdict useless.
    assert len(kept) / len(learner) > 0.85, (
        f"GOP_MIN={GOP_MIN} only credits {len(kept)}/{len(learner)} of the learner's own "
        f"correct recordings, so it would refuse the near-misses too")

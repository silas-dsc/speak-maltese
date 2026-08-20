"""The duration prior's inputs, checked on both sides of the port.

`constrained_ctc.rank_score` and `nanostt.rankScore` have to agree, and the pair already
has a parity test — but it needs an MP3 decoder and rendered eval clips, so it skips
almost everywhere, including here. What this file adds is a check that runs on numpy and
node alone: a waveform built in code rather than decoded, so the frame energy, the speech
span and the prior can be compared with nothing installed that the app does not need.

Two things are being defended.

**The span.** `features` normalises each mel bin over time, which is exactly the step
that throws absolute level away — so a frame's energy has to be taken before that or not
at all. If the JS ever computes it after, every span becomes the whole clip and nothing
fails loudly.

**The inertness.** `DUR_SD_SLOPE` and `DUR_FRAMES` exist so a joint re-sweep of the
constants and `DUR_WEIGHT` has somewhere to land, and until that happens they must
change nothing. A default that quietly starts charging hypotheses differently would move
every ranking decision in the app, which is not a thing to discover later.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
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
import { features, speechSpan, speechFrames, durationSd, durationPrior,
         rankScore } from '../frontend/nanostt.js';

const wave = Float32Array.from(JSON.parse(readFileSync(process.argv[2], 'utf-8')));
const { frames, energy } = features(wave);

/* A tiny hand-built posterior grid, so `rankScore`'s acoustic half is the same numbers
   on both sides and only the prior half can differ. */
const V = 4, BLK = 3;
const lp = Float64Array.from(JSON.parse(readFileSync(process.argv[3], 'utf-8')));
const nf = lp.length / V;

console.log(JSON.stringify({
  frames,
  energy: Array.from(energy),
  span: speechSpan(energy),
  spanLoose: speechSpan(energy, 60),
  speechFrames: speechFrames(energy),
  sd: [0, 1, 5, 13, 40].map(durationSd),
  prior: [[5, 100], [13, 107], [25, 100], [40, 200]].map(([k, f]) => durationPrior(k, f)),
  rankDefault: rankScore(lp, nf, V, [0, 1], BLK),
  rankSpoken: rankScore(lp, nf, V, [0, 1], BLK, 2),
}));
"""


def load_cc():
    spec = importlib.util.spec_from_file_location(
        "cc_dur", ROOT / "scripts" / "constrained_ctc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synth_wave() -> "np.ndarray":
    """Silence, then speech-ish, then silence — the shape of a push-to-talk recording.

    Two tone bursts with a gap, because the gap is the interesting part: an interior
    pause belongs to the utterance and must survive trimming, where the head and tail
    must not."""
    rng = np.random.default_rng(4)
    sr = 16000
    quiet = (rng.normal(0, 1e-4, int(0.6 * sr))).astype(np.float32)

    def burst(seconds: float) -> "np.ndarray":
        t = np.arange(int(seconds * sr), dtype=np.float32) / sr
        tone = 0.3 * (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 900 * t))
        return (tone + rng.normal(0, 0.01, t.size)).astype(np.float32)

    return np.concatenate([quiet, burst(0.7), quiet[:int(0.2 * sr)], burst(0.5), quiet])


@pytest.fixture(scope="module")
def pair():
    cc = load_cc()
    wave = synth_wave()

    rng = np.random.default_rng(11)
    grid = np.log(rng.dirichlet(np.ones(4), size=7)).astype(np.float64)

    wave_path = ROOT / "tests" / "_duration_wave.json"
    lp_path = ROOT / "tests" / "_duration_logprobs.json"
    driver = ROOT / "tests" / "_duration_driver.mjs"
    wave_path.write_text(json.dumps([float(x) for x in wave]), encoding="utf-8")
    lp_path.write_text(json.dumps([float(x) for x in grid.ravel()]), encoding="utf-8")
    driver.write_text(DRIVER, encoding="utf-8")
    try:
        proc = subprocess.run([node, str(driver), str(wave_path), str(lp_path)],
                              cwd=ROOT, capture_output=True, text=True, timeout=120)
    finally:
        for path in (wave_path, lp_path, driver):
            path.unlink(missing_ok=True)
    if proc.returncode != 0:
        pytest.fail(f"node driver failed:\n{proc.stderr}")
    return cc, wave, grid, json.loads(proc.stdout)


def test_the_frame_energy_matches(pair):
    """Taken before the per-bin normalisation on both sides, or the span is meaningless."""
    cc, wave, _grid, js = pair
    py = cc.frame_energy(wave)
    got = np.asarray(js["energy"], dtype=np.float64)
    assert got.size == py.size
    # float32 accumulation in JS against float32 in numpy: compare in relative terms,
    # because the absolute magnitudes span many orders between silence and a burst.
    assert np.allclose(got, py, rtol=1e-3, atol=1e-9)


def test_the_speech_span_matches(pair):
    cc, wave, _grid, js = pair
    energy = cc.frame_energy(wave)
    assert tuple(js["span"]) == cc.speech_span(energy)
    assert tuple(js["spanLoose"]) == cc.speech_span(energy, 60.0)


def test_the_span_drops_the_padding_and_keeps_the_pause(pair):
    """The head and tail are silence and must go; the gap between the two bursts is
    inside the utterance and must stay."""
    cc, wave, _grid, js = pair
    start, end = js["span"]
    total = js["frames"]
    assert start > 20, "leading silence was not trimmed"
    assert end < total - 20, "trailing silence was not trimmed"
    # 0.7s + 0.2s gap + 0.5s of content at 100fps, give or take the window.
    assert 130 <= end - start <= 155


def test_the_speech_frame_count_matches(pair):
    cc, wave, _grid, js = pair
    assert js["speechFrames"] == cc.speech_frames(wave)
    assert js["speechFrames"] < js["frames"] // 2


def test_the_duration_sd_matches(pair):
    cc, _wave, _grid, js = pair
    for tokens, got in zip((0, 1, 5, 13, 40), js["sd"]):
        assert got == pytest.approx(cc.duration_sd(tokens), rel=1e-12)


def test_the_duration_prior_matches(pair):
    cc, _wave, _grid, js = pair
    for (tokens, frames), got in zip(((5, 100), (13, 107), (25, 100), (40, 200)),
                                     js["prior"]):
        assert got == pytest.approx(cc.duration_prior(tokens, frames), rel=1e-12)


def test_the_defaults_are_inert(pair):
    """The two new switches must reproduce the deployed arithmetic exactly.

    Not a tautology: it is the check that stops a later edit to `DUR_SD_SLOPE` or
    `DUR_FRAMES` from silently becoming the shipped behaviour."""
    cc, _wave, _grid, _js = pair
    assert cc.DUR_FRAMES == "total"
    assert cc.DUR_SD_SLOPE == 0.0
    for tokens in (1, 5, 13, 40, 62):
        assert cc.duration_sd(tokens) == cc.DUR_SD
        expected = cc.DUR_INTERCEPT + cc.DUR_SLOPE * tokens
        assert cc.duration_prior(tokens, 100) == pytest.approx(
            -0.5 * ((100 - expected) / cc.DUR_SD) ** 2, rel=1e-12)


def test_the_rank_score_only_moves_when_told_to(pair):
    """Omitting `speech` has to leave the score bit-for-bit where it was, and passing it
    has to change only the prior's half."""
    cc, _wave, grid, js = pair
    ids, blank = [0, 1], 3
    assert js["rankDefault"] == pytest.approx(
        cc.rank_score(grid, ids, blank), rel=1e-12)
    assert js["rankSpoken"] == pytest.approx(
        cc.rank_score(grid, ids, blank, speech=2), rel=1e-12)
    delta = js["rankSpoken"] - js["rankDefault"]
    expected = cc.DUR_WEIGHT * (cc.duration_prior(len(ids), 2)
                                - cc.duration_prior(len(ids), len(grid)))
    assert delta == pytest.approx(expected, rel=1e-12)

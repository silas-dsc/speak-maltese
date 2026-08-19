"""The browser's features have to be the features the model was trained on.

`scripts/distill_stt.py` fed the student NeMo-style log-mel computed in Python — 64
Slaney mel bins over a reflect-padded 512-point STFT with a 320-sample Hann window,
pre-emphasised, then normalised per bin. `frontend/nanostt.js` recomputes all of that in
JavaScript so the recogniser can run on the device.

Nothing about a mismatch announces itself. A window off by 80 samples, `periodic` instead
of symmetric, zero padding instead of reflect, or the population variance instead of the
sample variance all produce plausible numbers and slightly worse Maltese — which would be
read as the small model being disappointing rather than as a bug on this side. So the two
implementations are compared directly, on real audio, to float32 resolution.

The decode is checked here too, for the reason it exists: merging repeated frames after
removing blanks eats every Maltese geminate, and that is a difference of one line whose
symptom is `grazzi` coming back as `grazi`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# CI installs only what the app needs, so the comparison this file makes — against the
# Python that produced the training data — simply does not run there. Skipped rather
# than faked: it is a numerical check or it is nothing.
np = pytest.importorskip("numpy")

node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node is not installed")

CLIPS = ROOT / "data" / "eval_clips"

DRIVER = r"""
import { readFileSync } from 'node:fs';
import { features, ctcDecode, melFilters, analysisWindow,
         ctcLogp, greedyLogp, targetConfidence, encodeTarget,
         durationPrior, rankScore } from '../frontend/nanostt.js';

const samples = Float32Array.from(JSON.parse(readFileSync(process.argv[2], 'utf-8')));
const { data, nMels, frames } = features(samples);

/* The geminate case, on the vocabulary the model actually uses: `▁` is the space and
   `<blk>` the blank. Repeated frames of the same letter must survive as two letters. */
const idToTok = ['▁', 'g', 'r', 'a', 'z', 'i', '<blk>'];
const BLK = 6;

/* The forward algorithm, on hand-built log-probabilities so Python can score exactly
   the same numbers. Two frames, three symbols, and a repeat that needs a blank. */
const V = 4, SBLK = 3;
const LP = Float64Array.from(JSON.parse(readFileSync(process.argv[3], 'utf-8')));
const LPF = LP.length / V;
const tokToId = new Map([['a', 0], ['b', 1], ['▁', 2], ['<blk>', 3]]);

console.log(JSON.stringify({
  nMels, frames,
  ctc: {
    ab:   ctcLogp(LP, LPF, V, [0, 1], SBLK),
    aa:   ctcLogp(LP, LPF, V, [0, 0], SBLK),
    a:    ctcLogp(LP, LPF, V, [0], SBLK),
    empty: ctcLogp(LP, LPF, V, [], SBLK),
    greedy: greedyLogp(LP, LPF, V),
    confAb: targetConfidence(LP, LPF, V, [0, 1], SBLK),
    rankAb: rankScore(LP, LPF, V, [0, 1], SBLK),
    rankA:  rankScore(LP, LPF, V, [0], SBLK),
  },
  duration: [[5, 100], [25, 100], [12, 60], [1, 1], [40, 200]]
    .map(([k, f]) => durationPrior(k, f)),
  encoded: {
    plain: encodeTarget('ab', tokToId),
    spaced: encodeTarget('a b', tokToId),
    unknown: encodeTarget('a?z b', tokToId),
  },
  mel: Array.from(data),
  melBankRow: Array.from(melFilters()[20]),
  window: Array.from(analysisWindow()),
  geminate: ctcDecode([1, 2, 3, 4, 6, 4, 5], idToTok, BLK),
  repeatsNoBlank: ctcDecode([1, 1, 2, 2, 3, 3], idToTok, BLK),
  spaces: ctcDecode([1, 0, 2], idToTok, BLK),
}));
"""


@pytest.fixture(scope="module")
def pair():
    """The same clip through both implementations.

    Ordered so that the cheap reasons to skip come first: this needs real audio and an
    MP3 decoder, and an environment missing either should say so rather than raise a
    collection error five times over."""
    clips = sorted(CLIPS.glob("synth_*.mp3"))
    if not clips:
        pytest.skip("no eval clips; run scripts/compare_stt.py --synth 25")
    decode_audio = pytest.importorskip("faster_whisper.audio").decode_audio

    from compare_stt import _NEMO, _mel_filters, _nemo_features

    wave = np.asarray(decode_audio(str(clips[0]), sampling_rate=16000), dtype=np.float32)

    fb = _mel_filters(_NEMO["n_fft"] // 2 + 1, 64, _NEMO["sample_rate"])
    pad = (_NEMO["n_fft"] - _NEMO["win_length"]) // 2
    window = np.pad(np.hanning(_NEMO["win_length"]).astype(np.float32), (pad, pad))
    py = _nemo_features(wave, 64, fb, window)[0]            # (64, T)

    samples = ROOT / "tests" / "_nanostt_samples.json"
    driver = ROOT / "tests" / "_nanostt_driver.mjs"
    # A small, fixed log-probability grid: the scorer is compared symbol for symbol,
    # so it must be the same numbers on both sides rather than the same audio.
    rng = np.random.default_rng(7)
    grid = np.log(rng.dirichlet(np.ones(4), size=9)).astype(np.float64)
    lp_path = ROOT / "tests" / "_nanostt_logprobs.json"
    lp_path.write_text(json.dumps([float(x) for x in grid.ravel()]), encoding="utf-8")

    samples.write_text(json.dumps([float(x) for x in wave]), encoding="utf-8")
    driver.write_text(DRIVER, encoding="utf-8")
    try:
        proc = subprocess.run([node, str(driver), str(samples), str(lp_path)], cwd=ROOT,
                              capture_output=True, text=True, timeout=120)
    finally:
        samples.unlink(missing_ok=True)
        lp_path.unlink(missing_ok=True)
        driver.unlink(missing_ok=True)
    if proc.returncode != 0:
        pytest.fail(f"node driver failed:\n{proc.stderr}")
    js = json.loads(proc.stdout)
    return py, js, window, fb, grid


def test_the_two_implementations_agree_on_frame_count(pair):
    """An off-by-one here would misalign every frame after the first and still look
    like a working recogniser."""
    py, js, _w, _fb, _g = pair
    assert js["nMels"] == py.shape[0]
    assert js["frames"] == py.shape[1], f"JS {js['frames']} vs Python {py.shape[1]}"


def test_the_analysis_window_matches(pair):
    """320-sample symmetric Hann, zero-padded into a 512-point frame. `periodic=True`
    is the same shape and the wrong one."""
    _py, js, window, _fb, _g = pair
    assert np.allclose(np.array(js["window"]), window, atol=1e-6)


def test_the_mel_filterbank_matches(pair):
    """Slaney scale *and* Slaney normalisation — librosa with `htk=False`."""
    _py, js, _w, fb, _g = pair
    assert np.allclose(np.array(js["melBankRow"]), fb[:, 20], atol=1e-6)


def test_the_features_match_on_real_audio(pair):
    """The whole pipeline end to end: pre-emphasis, reflect padding, STFT, mel, log,
    per-bin normalisation."""
    py, js, _w, _fb, _g = pair
    got = np.array(js["mel"], dtype=np.float32).reshape(js["nMels"], js["frames"])
    diff = np.abs(got - py)
    assert diff.max() < 2e-3, f"max abs diff {diff.max():.2e}"
    # Correlation catches a systematic distortion that a loose tolerance would let past.
    assert np.corrcoef(got.ravel(), py.ravel())[0, 1] > 0.99999


def test_the_decode_keeps_geminates(pair):
    """`grazzi`, not `grazi`. Merge repeats first, then drop blanks."""
    _py, js, _w, _fb, _g = pair
    assert js["geminate"] == "grazzi"
    # Repeats with no blank between them are one letter — that is what CTC means, and
    # it is why the blank in the case above is load-bearing.
    assert js["repeatsNoBlank"] == "gra"
    assert js["spaces"] == "g r"


def test_the_target_scorer_matches_python(pair):
    """The CTC forward algorithm, symbol for symbol against `constrained_ctc.py`.

    This is the number the app now grades on, so a JS-only bug here would not look like
    a bug — it would look like the recogniser being unfair. `aa` against `a` is the case
    worth staring at: a repeated symbol needs a blank between its copies, which is the
    whole reason `irrid` and `irid` can be told apart at all."""
    _py, js, _w, _fb, grid = pair
    from constrained_ctc import ctc_logp, confidence, greedy_logp

    blank = 3
    for name, ids in (("ab", [0, 1]), ("aa", [0, 0]), ("a", [0])):
        want = ctc_logp(grid, ids, blank)
        got = js["ctc"][name]
        assert abs(got - want) < 1e-9, f"{name}: JS {got} vs Python {want}"

    assert abs(js["ctc"]["greedy"] - greedy_logp(grid)) < 1e-9
    assert abs(js["ctc"]["confAb"] - confidence(grid, [0, 1], blank)) < 1e-9
    # An empty target cannot be scored, and must not come back as "perfectly likely".
    assert js["ctc"]["empty"] < -1e29

    # `aa` and `a` must score *differently* — that is the property the app depends on,
    # and the one greedy decoding cannot express. Which of them wins is a property of
    # the audio, not of the algorithm: over nine frames the single symbol has to spend
    # eight of them on blanks, so it is not automatically the likelier reading.
    assert js["ctc"]["aa"] != js["ctc"]["a"]
    assert abs(js["ctc"]["aa"] - ctc_logp(grid, [0, 0], blank)) < 1e-9


def test_the_duration_prior_matches_python(pair):
    """Same arithmetic on both sides, or the browser ranks differently from the reference.

    This one is worth its own test rather than riding on `rankScore`: the prior is three
    constants and a squared z-score, which is exactly the shape of thing that gets ported
    with a sign flipped or a standard deviation squared twice and still looks plausible."""
    _py, js, _w, _fb, _g = pair
    from constrained_ctc import duration_prior

    for got, (tokens, frames) in zip(js["duration"],
                                     ((5, 100), (25, 100), (12, 60), (1, 1), (40, 200))):
        want = duration_prior(tokens, frames)
        assert abs(got - want) < 1e-9, f"{tokens}/{frames}: JS {got} vs Python {want}"
        assert got <= 0, "the prior is a penalty, never a bonus"


def test_the_prior_charges_a_short_hypothesis_for_a_long_utterance(pair):
    """The artefact this exists to remove.

    Five tokens is not a plausible reading of two seconds of audio, and without the prior
    nothing in the scorer says so — which is why all five of the learner's rank failures
    lost to `Bonġu!`. A hundred frames is two seconds; the prior has to prefer 25 tokens
    to 5 by a wide margin, and be nearly indifferent between lengths that are both
    plausible."""
    from constrained_ctc import DUR_WEIGHT, duration_prior

    short, right = duration_prior(5, 100), duration_prior(25, 100)
    assert right > short
    assert DUR_WEIGHT * (right - short) > 0.5, "too weak to overturn a 0.14 gap"
    # …and it must not start refereeing between two reasonable readings.
    assert abs(DUR_WEIGHT * (duration_prior(38, 100) - duration_prior(40, 100))) < 0.02


def test_the_floor_and_the_ranking_do_not_share_a_number(pair):
    """`confidence` is the floor's, `rank` is the field's, and merging them would put the
    duration prior into `MIN_CONFIDENCE` — where it would mean that the bar for "is there
    speech here" moves with the length of whichever line happened to be asked for."""
    _py, js, _w, _fb, grid = pair
    from constrained_ctc import confidence, rank_score

    assert abs(js["ctc"]["rankAb"] - rank_score(grid, [0, 1], 3)) < 1e-9
    assert js["ctc"]["rankAb"] < js["ctc"]["confAb"], "the prior can only cost"
    # The prior has to *reach* the decision: two hypotheses of different length cannot be
    # separated by the same amount before and after it.
    d_conf = abs(confidence(grid, [0, 1], 3) - confidence(grid, [0], 3))
    d_rank = abs(js["ctc"]["rankAb"] - js["ctc"]["rankA"])
    assert abs(d_rank - d_conf) > 1e-6

    # Which way it pushes depends on which side of the expected duration the audio sits,
    # and this fixture is on the other side from the real case: nine frames is far too
    # short for either reading, so the prior prefers the shorter one — correctly, and as
    # the mirror image of `Bonġu!` winning a two-second utterance. The direction that
    # matters in practice is asserted above, at a hundred frames.
    assert d_rank < d_conf


def test_encoding_a_target_drops_only_what_the_model_lacks(pair):
    """Spaces become the word-delimiter token; punctuation the vocabulary has no token
    for is dropped rather than turned into the unknown symbol, which would make every
    target a near-miss of itself."""
    _py, js, _w, _fb, _g = pair
    assert js["encoded"]["plain"] == [0, 1]
    assert js["encoded"]["spaced"] == [0, 2, 1]
    assert js["encoded"]["unknown"] == [0, 2, 1], "'?' and 'z' should vanish, not encode"

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
import { features, ctcDecode, melFilters, analysisWindow } from '../frontend/nanostt.js';

const samples = Float32Array.from(JSON.parse(readFileSync(process.argv[2], 'utf-8')));
const { data, nMels, frames } = features(samples);

/* The geminate case, on the vocabulary the model actually uses: `▁` is the space and
   `<blk>` the blank. Repeated frames of the same letter must survive as two letters. */
const idToTok = ['▁', 'g', 'r', 'a', 'z', 'i', '<blk>'];
const BLK = 6;
const doubled = [1, 6, 2, 3, 4, 4, 5, 5, 6];   // g r a z z i, with a blank between the pairs
const collapsed = [1, 2, 3, 4, 6, 4, 5, 6, 5];

console.log(JSON.stringify({
  nMels, frames,
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
    samples.write_text(json.dumps([float(x) for x in wave]), encoding="utf-8")
    driver.write_text(DRIVER, encoding="utf-8")
    try:
        proc = subprocess.run([node, str(driver), str(samples)], cwd=ROOT,
                              capture_output=True, text=True, timeout=120)
    finally:
        samples.unlink(missing_ok=True)
        driver.unlink(missing_ok=True)
    if proc.returncode != 0:
        pytest.fail(f"node driver failed:\n{proc.stderr}")
    js = json.loads(proc.stdout)
    return py, js, window, fb


def test_the_two_implementations_agree_on_frame_count(pair):
    """An off-by-one here would misalign every frame after the first and still look
    like a working recogniser."""
    py, js, _w, _fb = pair
    assert js["nMels"] == py.shape[0]
    assert js["frames"] == py.shape[1], f"JS {js['frames']} vs Python {py.shape[1]}"


def test_the_analysis_window_matches(pair):
    """320-sample symmetric Hann, zero-padded into a 512-point frame. `periodic=True`
    is the same shape and the wrong one."""
    _py, js, window, _fb = pair
    assert np.allclose(np.array(js["window"]), window, atol=1e-6)


def test_the_mel_filterbank_matches(pair):
    """Slaney scale *and* Slaney normalisation — librosa with `htk=False`."""
    _py, js, _w, fb = pair
    assert np.allclose(np.array(js["melBankRow"]), fb[:, 20], atol=1e-6)


def test_the_features_match_on_real_audio(pair):
    """The whole pipeline end to end: pre-emphasis, reflect padding, STFT, mel, log,
    per-bin normalisation."""
    py, js, _w, _fb = pair
    got = np.array(js["mel"], dtype=np.float32).reshape(js["nMels"], js["frames"])
    diff = np.abs(got - py)
    assert diff.max() < 2e-3, f"max abs diff {diff.max():.2e}"
    # Correlation catches a systematic distortion that a loose tolerance would let past.
    assert np.corrcoef(got.ravel(), py.ravel())[0, 1] > 0.99999


def test_the_decode_keeps_geminates(pair):
    """`grazzi`, not `grazi`. Merge repeats first, then drop blanks."""
    _py, js, _w, _fb = pair
    assert js["geminate"] == "grazzi"
    # Repeats with no blank between them are one letter — that is what CTC means, and
    # it is why the blank in the case above is load-bearing.
    assert js["repeatsNoBlank"] == "gra"
    assert js["spaces"] == "g r"

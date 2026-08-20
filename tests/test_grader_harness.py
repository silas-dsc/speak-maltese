"""The two harnesses that let a change to the grader be priced instead of guessed.

`make_negatives.py` rebuilds the set of things that must be turned away;
`sweep_grader.py` scores a parameter grid against those and against the clips that must
be accepted. Neither can be run to completion here — one needs a microphone and the other
needs a model — so what is checked is everything that does not:

**The transforms are what they claim.** A "reversed" clip that was not reversed, or a
"−30dB" copy at the original level, would make a negative set that quietly proves the
grader is safe.

**The sweep's arithmetic is the deployed arithmetic.** This is the one that matters. The
harness recomputes `confidence` and `rank` from cached log-likelihoods rather than calling
`constrained_ctc`, because that is what makes a two-hundred-point grid free — and it is
therefore a second implementation of the rule being measured. If it drifts, the sweep
prices a grader nobody ships. So it is held to `constrained_ctc` exactly, on the same
posteriors, to the last bit.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

np = pytest.importorskip("numpy")


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"harness_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mn():
    return load_script("make_negatives")


@pytest.fixture(scope="module")
def sw():
    return load_script("sweep_grader")


@pytest.fixture(scope="module")
def cc():
    return load_script("constrained_ctc")


# ── The negatives ──────────────────────────────────────────────────────────

def test_reversing_actually_reverses(mn):
    wave = np.linspace(-1, 1, 500, dtype=np.float32)
    got = mn.reversed_clip(wave)
    assert np.array_equal(got, wave[::-1])
    assert got is not wave, "must not alias the source; the caller still needs it"


def test_attenuation_lands_on_the_decibels_asked_for(mn):
    wave = (np.sin(np.linspace(0, 40, 4000)) * 0.8).astype(np.float32)
    quiet = mn.attenuated(wave, -30.0)
    # -30dB is a factor of 10**(-30/20) = 0.0316 in amplitude.
    assert mn.peak(quiet) / mn.peak(wave) == pytest.approx(10 ** (-1.5), rel=1e-6)
    assert mn.attenuated(wave, 0.0) == pytest.approx(wave, rel=1e-6)


def test_the_noise_levels_are_ordered_and_bounded(mn):
    rng = np.random.default_rng(0)
    levels = [float(np.std(mn.white_noise(20000, lvl, rng))) for lvl in mn.NOISE_LEVELS]
    assert levels == sorted(levels), "the five levels must actually differ, in order"
    for lvl, got in zip(mn.NOISE_LEVELS, levels):
        assert got == pytest.approx(lvl, rel=0.1)
    loud = mn.white_noise(20000, 2.0, rng)
    assert mn.peak(loud) <= 1.0, "a WAV cannot hold more than full scale"


def test_silence_is_actually_zero(mn):
    quiet = mn.digital_silence(1000)
    assert quiet.shape == (1000,)
    assert mn.peak(quiet) == 0.0


def test_the_stdlib_wav_path_agrees_with_a_real_decoder(tmp_path):
    """`audio_io` reads a plain WAV with the standard library so the negatives can be
    built with nothing installed — which is only safe if it returns the same samples a
    real decoder would. `--record` writes exactly this format, so this is the normal
    path, not a fallback."""
    sf = pytest.importorskip("soundfile")
    io = load_script("audio_io")
    rng = np.random.default_rng(0)
    wave = rng.uniform(-0.9, 0.9, 8000).astype(np.float32)

    mono = tmp_path / "mono.wav"
    io.write_wav(mono, wave)
    viasf, _rate = sf.read(str(mono), dtype="float32", always_2d=True)
    assert np.array_equal(io.read_audio(mono), viasf.mean(axis=1))

    # Stereo at another rate has to come back mono at 16k, the same length either way.
    other = tmp_path / "stereo24k.wav"
    sf.write(str(other), np.c_[wave, wave], 24000, subtype="PCM_16")
    got = io.read_audio(other)
    assert got.ndim == 1
    assert len(got) == int(round(len(wave) * 16000 / 24000))


def test_unreadable_audio_is_none_rather_than_a_crash(tmp_path):
    io = load_script("audio_io")
    junk = tmp_path / "not-audio.wav"
    junk.write_bytes(b"this is not a RIFF header")
    assert io.read_audio(junk) is None
    assert io.read_audio(tmp_path / "absent.wav") is None


def test_the_set_is_built_from_the_recordings_and_counted(mn, tmp_path):
    """The composition is the thing a later sweep has to be able to reproduce, so it is
    written down rather than implied.

    Runs everywhere: the fixtures are written and read through `audio_io`, which needs
    only the standard library for a plain WAV."""
    clips = tmp_path / "eval_clips"
    clips.mkdir()
    rng = np.random.default_rng(3)
    for i in range(1, 26):
        n = int(16000 * rng.uniform(1.2, 2.6))
        t = np.arange(n, dtype=np.float32) / 16000
        mn.write_clip(clips / f"me_{i:03d}.wav",
                      (0.4 * np.sin(2 * np.pi * 180 * t)).astype(np.float32))

    rows = mn.build(clips, tmp_path / "negatives", seed=5)
    counts = mn.summarise(rows)
    assert counts == {"silence": mn.N_SILENCE, "hiss": mn.N_NOISE,
                      "quiet": 25, "reversed": 25}
    assert len(rows) == 90, "the published count, which is what this reconstructs"

    # The manifest is the record, and it has to round-trip.
    again = mn.read_negatives(tmp_path / "negatives")
    assert len(again) == len(rows)
    assert {r["kind"] for r in again} == set(counts)
    assert all((tmp_path / "negatives" / r["file"]).exists() for r in again)

    # And the derived clips really are derived from the originals.
    original = mn.read_clip(clips / "me_001.wav")
    assert np.allclose(mn.read_clip(tmp_path / "negatives" / "neg_reversed_me_001.wav"),
                       original[::-1], atol=2e-4)
    ratio = (mn.peak(mn.read_clip(tmp_path / "negatives" / "neg_quiet_me_001.wav"))
             / mn.peak(original))
    assert ratio == pytest.approx(10 ** -1.5, rel=0.05)


def test_no_recordings_is_a_message_and_not_a_traceback(mn, tmp_path):
    assert mn.build(tmp_path / "nothing", tmp_path / "out") == []
    assert mn.read_negatives(tmp_path / "absent") == []


def test_synthetic_renders_are_not_used_as_negatives(mn, tmp_path):
    """A negative derived from TTS measures the synthesiser, and constants fitted on
    synthetic speech not transferring is the whole reason this set exists."""
    clips = tmp_path / "c"
    clips.mkdir()
    for name in ("me_001.wav", "synth_001.wav", "synth_002.mp3"):
        (clips / name).write_bytes(b"x")
    assert [p.name for p in mn.source_clips(clips)] == ["me_001.wav"]


# ── The sweep ──────────────────────────────────────────────────────────────

def make_record(cc_mod, kind="learner", frames=120, seed=3, n_field=6):
    """A cache row built from real posteriors, so the parity check has something to
    compare against."""
    rng = np.random.default_rng(seed)
    v_size, blank = 12, 0
    post = np.log(rng.dirichlet(np.ones(v_size), size=frames)).astype(np.float64)
    target = [int(rng.integers(1, v_size)) for _ in range(14)]
    field = [[int(rng.integers(1, v_size)) for _ in range(int(rng.integers(5, 30)))]
             for _ in range(n_field)]
    rec = {
        "clip": "x", "kind": kind, "frames_total": frames,
        "frames_speech": max(4, frames - 40),
        "greedy": float(cc_mod.greedy_logp(post)),
        "target": {"tokens": len(target),
                   "logp": float(cc_mod.ctc_logp(post, target, blank))},
        "field": [{"tokens": len(f), "local": i < 2,
                   "logp": float(cc_mod.ctc_logp(post, f, blank))}
                  for i, f in enumerate(field)],
    }
    return rec, post, target, field, blank


def test_the_sweep_reproduces_the_deployed_scorer_exactly(sw, cc):
    """The load-bearing test in this file. The harness is a second implementation of the
    rule, and a sweep that prices a slightly different rule is worse than no sweep."""
    rec, post, target, field, blank = make_record(cc)
    params = dict(sw.DEPLOYED, field_size=len(field))
    got = sw.decide(rec, params)

    assert got["confidence"] == pytest.approx(
        cc.confidence(post, target, blank), rel=1e-12, abs=0)
    assert got["rank"] == pytest.approx(
        cc.rank_score(post, target, blank), rel=1e-12, abs=0)
    for entry, ids in zip(rec["field"], field):
        assert sw.rank_of(entry, rec, params) == pytest.approx(
            cc.rank_score(post, ids, blank), rel=1e-12, abs=0)


def test_the_floor_reads_the_confidence_and_not_the_rank(sw, cc):
    """Two numbers, two jobs — the same split `app.js` makes. A floor on `rank` would
    move with the length of whichever line was asked for."""
    rec, _post, _t, field, _b = make_record(cc)
    base = dict(sw.DEPLOYED, field_size=len(field))
    conf = sw.decide(rec, base)["confidence"]
    # The floor cannot touch the confidence, only the verdict drawn from it.
    assert sw.decide(rec, dict(base, floor=0.0))["confidence"] == conf
    assert sw.decide(rec, dict(base, floor=1.5))["confidence"] == conf

    under = sw.decide(rec, dict(base, floor=conf + 0.01))
    assert under["passes_floor"] is False
    assert "under the floor" in under["reason"]
    assert sw.decide(rec, dict(base, floor=conf - 0.01))["passes_floor"] is True
    # Whether it also won its field is a separate axis, and must not move with the floor.
    assert (sw.decide(rec, dict(base, floor=0.0))["clear"]
            == sw.decide(rec, dict(base, floor=1.5))["clear"])
    # Raising the floor must never *gain* an accept.
    assert not sw.decide(rec, dict(base, floor=1.5))["accepted"]


def test_both_reasons_for_refusing_are_reported(sw, cc):
    """A floor that is too high and a field that is too hard want opposite fixes, so a
    sweep has to be able to tell which one it is paying."""
    rec, _post, _t, field, _b = make_record(cc)
    base = dict(sw.DEPLOYED, field_size=len(field))
    both = sw.decide(rec, dict(base, floor=1.5, margin_sigmas=50.0))
    assert both["clear"] is False and both["passes_floor"] is False
    assert both["reason"] == "lost the field and under the floor"


def test_the_margin_never_drops_below_the_swept_constant(sw, cc):
    """`margin_sigmas` scales the requirement; it must not be able to weaken it."""
    rec, _post, _t, field, _b = make_record(cc)
    base = dict(sw.DEPLOYED, field_size=len(field))
    loose = sw.decide(rec, dict(base, margin_sigmas=0.0))
    assert loose["need"] == pytest.approx(base["min_margin"])
    strict = sw.decide(rec, dict(base, margin_sigmas=3.0))
    assert strict["need"] >= loose["need"]
    assert strict["need"] == pytest.approx(max(base["min_margin"],
                                               3.0 * strict["spread"]))


def test_a_bigger_margin_can_only_cost_accepts(sw, cc):
    """Monotonic by construction, and worth pinning: it is the reason the switch is safe
    to sweep in one direction only."""
    accepted = []
    for seed in range(12):
        rec, _p, _t, field, _b = make_record(cc, seed=seed)
        base = dict(sw.DEPLOYED, field_size=len(field))
        row = [sw.decide(rec, dict(base, margin_sigmas=s))["accepted"]
               for s in (0.0, 0.5, 1.0, 4.0)]
        accepted.append(row)
    for row in accepted:
        # Once refused, never accepted again as the requirement grows.
        assert row == sorted(row, reverse=True), row


def test_the_field_mix_honours_the_local_share(sw, cc):
    rec, _p, _t, _f, _b = make_record(cc, n_field=20)
    for i, entry in enumerate(rec["field"]):
        entry["local"] = i < 8
    rng = np.random.default_rng(0)
    only_global = sw.choose_field(rec, dict(sw.DEPLOYED, field_size=10,
                                            field_local=0.0), rng)
    assert len(only_global) == 10
    assert not any(e["local"] for e in only_global)

    half = sw.choose_field(rec, dict(sw.DEPLOYED, field_size=10, field_local=0.5),
                           np.random.default_rng(0))
    assert len(half) == 10
    assert sum(e["local"] for e in half) == 5


def test_a_thin_field_is_topped_up_rather_than_shrunk(sw, cc):
    """Ranking against fewer alternatives is a different change wearing the same
    clothes, so the field size has to hold even when the local pool cannot fill it."""
    rec, _p, _t, _f, _b = make_record(cc, n_field=20)
    for i, entry in enumerate(rec["field"]):
        entry["local"] = i < 2
    got = sw.choose_field(rec, dict(sw.DEPLOYED, field_size=12, field_local=1.0),
                          np.random.default_rng(1))
    assert len(got) == 12


def test_an_empty_field_falls_back_to_the_floor(sw, cc):
    """What the FastAPI dev build actually does — weaker, but it must not crash or
    accept everything."""
    rec, _p, _t, _f, _b = make_record(cc)
    rec["field"] = []
    got = sw.decide(rec, sw.DEPLOYED)
    assert got["runner_up"] is None
    assert got["accepted"] == (got["confidence"] >= sw.DEPLOYED["floor"])


def test_the_frame_mode_changes_the_prior_and_not_the_confidence(sw, cc):
    """`dur_frames` is the switch that reprices the prior; the floor must not move with
    it, or the bar for "is there speech here" starts tracking the padding."""
    rec, _p, _t, field, _b = make_record(cc)
    base = dict(sw.DEPLOYED, field_size=len(field))
    total = sw.decide(rec, dict(base, dur_frames="total"))
    speech = sw.decide(rec, dict(base, dur_frames="speech"))
    assert total["confidence"] == speech["confidence"]
    assert total["rank"] != speech["rank"], "the prior should have seen a different length"


def test_the_measurement_counts_every_kind_and_reports_its_spread(sw, cc):
    records = []
    for seed in range(9):
        rec, _p, _t, _f, _b = make_record(cc, kind="learner" if seed < 5 else "hiss",
                                          seed=seed)
        records.append(rec)
    got = sw.measure(records, dict(sw.DEPLOYED, field_size=6), seeds=3)
    assert got["learner"]["of"] == 5
    assert got["hiss"]["of"] == 4
    for stats in got.values():
        assert 0.0 <= stats["rate"] <= 1.0
        assert stats["spread"] >= 0, "the seed spread is the error bar and must be there"


def test_the_grid_axes_all_name_a_real_parameter(sw):
    """A typo in a grid key would sweep nothing and report the deployed row N times."""
    for name, (axis, values) in sw.GRIDS.items():
        assert axis in sw.DEPLOYED, f"grid {name!r} sweeps unknown parameter {axis!r}"
        assert values, name
        assert sw.DEPLOYED[axis] in values, (
            f"grid {name!r} does not include the deployed value, so there is nothing "
            f"to compare against")


def test_the_cache_survives_a_round_trip_through_json(sw, cc, tmp_path):
    """The whole point of the cache is that a re-sweep needs no model, so it has to be
    plain JSON with no numpy left in it."""
    rec, _p, _t, field, _b = make_record(cc)
    path = tmp_path / "cache.json"
    path.write_text(json.dumps([rec]), encoding="utf-8")
    back = json.loads(path.read_text(encoding="utf-8"))
    params = dict(sw.DEPLOYED, field_size=len(field))
    assert sw.decide(back[0], params)["rank"] == pytest.approx(
        sw.decide(rec, params)["rank"], rel=1e-12)

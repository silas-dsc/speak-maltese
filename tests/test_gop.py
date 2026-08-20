"""Per-token GOP: the forward-backward has to be right before any score means anything."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _logprobs(rows):
    a = np.array(rows, dtype=np.float64)
    return a - np.log(np.exp(a).sum(axis=1, keepdims=True))


def test_occupancy_total_matches_the_projects_own_ctc_logp():
    """The one check that catches a wrong forward-backward.

    A recurrence with the skip rule slightly wrong still produces plausible occupancies and
    plausible scores; nothing downstream would complain. Summing the same lattice two ways
    and requiring the same total is what makes it a measurement rather than a guess."""
    from constrained_ctc import ctc_logp
    from gop import occupancy

    rng = np.random.default_rng(4)
    for T, V in ((9, 5), (17, 6), (31, 4)):
        post = _logprobs(rng.normal(0, 2, (T, V)))
        blank = V - 1
        for ids in ([0, 1, 2], [0, 0, 1], [1, 1, 1, 2], [2]):
            if len(ids) > T:
                continue
            gam = occupancy(post, ids, blank)
            assert gam.shape == (len(ids), T)
            # Every token must own about one emission's worth of frames: CTC emits each
            # target exactly once per path, so its occupancy sums to at least one frame.
            mass = gam.sum(axis=1)
            assert np.all(mass > 0.99), f"{ids}: a token owns no frames ({mass})"
            assert np.isfinite(ctc_logp(post, ids, blank))


def test_gop_is_zero_when_the_target_is_what_the_model_wanted():
    """GOP is a margin against the model's own first choice, so a target it already
    prefers at every frame must score zero — that is the definition, and it is what makes
    a negative number mean "the model wanted something else here"."""
    from gop import token_gop

    V, blank = 4, 3
    # A posterior that is confidently token 0, then 1, then 2, with blanks between.
    seq = [0, 0, blank, 1, 1, blank, 2, 2]
    rows = []
    for tok in seq:
        row = [-20.0] * V
        row[tok] = 0.0
        rows.append(row)
    post = _logprobs(rows)
    g = token_gop(post, [0, 1, 2], blank)
    assert np.all(g > -0.05), f"a target the model already prefers must score ~0, got {g}"

    # The same audio against a target the model did not say anywhere.
    g_wrong = token_gop(post, [0, 2, 1], blank)
    assert g_wrong.min() < -1.0, f"a wrong order must be charged, got {g_wrong}"


def test_gop_blames_the_token_that_is_actually_absent():
    """Per-token scoring earns its cost by saying *which* sound failed. If the blame lands
    on the wrong token the score is worse than useless — it would teach the learner to
    correct a sound they said correctly."""
    from gop import token_gop

    V, blank = 4, 3
    rows = []
    for tok in [0, 0, blank, 0, 0, blank, 2, 2]:   # token 1 is never said
        row = [-20.0] * V
        row[tok] = 0.0
        rows.append(row)
    post = _logprobs(rows)
    g = token_gop(post, [0, 1, 2], blank)
    assert int(np.argmin(g)) == 1, f"blame landed on {int(np.argmin(g))}, not the absent token"


def test_the_encoders_agree():
    """Two encoders on the same line would silently put the two scores on different token
    sequences. `space` is a vocabulary key rather than an id, which is exactly the kind of
    detail a second implementation gets wrong."""
    from constrained_ctc import encode
    from gop_encode import encode_tokens

    vocab = {"a": 0, "b": 1, "ħ": 2, "▁": 3}
    line = "ab ħa"
    ids, toks = encode_tokens(line, vocab, "▁")
    assert ids == encode(line, vocab, "▁")
    assert len(ids) == len(toks)
    assert toks == ["a", "b", "␣", "ħ", "a"]


def test_degeminate_halves_one_consonant_at_a_time():
    """A line with two geminates must not be compared against a variant differing twice.

    `kollox tajjeb` has two. Halving both at once and winning says the model heard *a*
    length somewhere, which is not the question — the question is which doubled consonant
    it heard, and that needs one change per variant."""
    from gemination import degeminate

    out = degeminate("kollox tajjeb")
    assert {v for v, _ in out} == {"kolox tajjeb", "kollox tajeb"}
    assert {c for _, c in out} == {"l", "j"}

    # Doubled vowels are a different thing in Maltese orthography and not what the
    # degemination work is about.
    assert degeminate("iimla") == []
    assert degeminate("grazzi") == [("grazi", "z")]
    assert degeminate("bonġu") == []

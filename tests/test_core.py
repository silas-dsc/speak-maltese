"""Unit tests for the parts where a silent bug would quietly teach bad Maltese
or corrupt the schedule."""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import srs, text  # noqa: E402
from backend import curriculum  # noqa: E402


# ── Text normalisation ─────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", [
    ("għandi", "andi"),            # għ is silent — recognisers drop it
    ("ħobż", "hobz"),              # diacritics stripped
    ("Il-ktieb", "ktieb"),         # article optional
    ("iż-żmien", "zmien"),
    ("Grazzi ħafna!", "grazzi hafna"),
])
def test_fold_is_tolerant(a, b):
    assert text.fold(a) == text.fold(b)


def test_fold_still_distinguishes_real_differences():
    assert text.fold("kelb") != text.fold("qalb")
    assert text.fold("nifhem") != text.fold("nifhimx")


def test_score_perfect_and_wrong():
    assert text.score("Bonġu, kif int?", "Bonġu, kif int?") == 1.0
    assert text.score("Bonġu kif int", "Bonġu, kif int?") > 0.95
    assert text.score("xejn", "Bonġu, kif int?") < 0.4


def test_score_partial_credit():
    s = text.score("Jien minn Awstralja", "Jien mill-Awstralja")
    assert 0.6 < s < 1.0


@pytest.mark.parametrize("said,target", [
    ("Jien mill Awstralja", "Jien mill-Awstralja."),   # recogniser drops the hyphen
    ("Noqgħod fit Tas-Sliema", "Noqgħod fit-Tas-Sliema"),
    ("id dar", "id-dar"),
])
def test_hyphenated_article_is_not_penalised(said, target):
    """Maltese fuses the article onto the next word; recognisers split it about half
    the time. That must not cost the learner a third of the score."""
    assert text.score(said, target) > 0.95


@pytest.mark.parametrize("bad,good", [
    ("Jien minn l-Awstralja.", "Jien mill-Awstralja."),
    ("Noqgħod fi il-Belt.", "Noqgħod fil-Belt."),
    ("Ġejt minn is-sena l-oħra.", "Ġejt mis-sena l-oħra."),
    ("Ħadt il-karozza minn id-dar.", "Ħadt il-karozza mid-dar."),
    ("Nitkellem bi il-Malti.", "Nitkellem bil-Malti."),
    ("Immur għal il-festa.", "Immur għall-festa."),
    ("Naħdem fi ix-xogħol.", "Naħdem fix-xogħol."),
    ("Il-ktieb ta il-tifel.", "Il-ktieb tal-tifel."),
    ("Mort sa il-baħar.", "Mort sal-baħar."),
])
def test_preposition_article_fusion_is_repaired(bad, good):
    assert text.apply_fusion(bad) == good
    assert text.lint_fusion(bad), "should have been flagged"


@pytest.mark.parametrize("ok", [
    "Jien mill-Awstralja.",
    "Noqgħod fil-Belt.",
    "Tini l-ilma, jekk jogħġbok.",
    "Ma nifhimx il-Malti.",
    "Il-ktieb fuq il-mejda.",
    "Ġejt ma' ħabib.",
])
def test_fusion_lint_leaves_correct_maltese_alone(ok):
    assert text.lint_fusion(ok) == []
    assert text.apply_fusion(ok) == ok


def test_tutor_lint_raises_a_correction_the_model_missed():
    """A weaker local model often replies without noticing the fusion error. The
    lint must supply the correction rather than let it through silently."""
    from backend import tutor

    data = tutor._coerce('{"reply_mt": "Mela!", "reply_en": "So!", '
                         '"correction": {"needed": false}}')
    tutor._lint("Jien minn l-Awstralja.", data)
    assert data["correction"]["needed"] is True
    assert data["correction"]["corrected_mt"] == "Jien mill-Awstralja."
    assert data["correction"]["repeat_prompt_mt"] == "Jien mill-Awstralja."
    assert any(i["should_be"] == "mill-" for i in data["correction"]["issues"])


def test_tutor_lint_repairs_the_models_own_output():
    """Worse than missing the error: 'correcting' it into another broken sentence."""
    from backend import tutor

    data = tutor._coerce(json.dumps({
        "reply_mt": "Mela int minn l-Awstralja!",
        "reply_en": "So you're from Australia!",
        "correction": {"needed": True, "corrected_mt": "Jien minn l-Awstralja.",
                       "repeat_prompt_mt": "Jien minn l-Awstralja.", "issues": []},
    }))
    tutor._lint("Jien minn l-Awstralja.", data)
    assert data["reply_mt"] == "Mela int mill-Awstralja!"
    assert data["correction"]["corrected_mt"] == "Jien mill-Awstralja."
    assert data["correction"]["repeat_prompt_mt"] == "Jien mill-Awstralja."


def test_tutor_lint_is_silent_on_correct_maltese():
    from backend import tutor

    data = tutor._coerce('{"reply_mt": "Mela int mill-Awstralja!", "reply_en": "x", '
                         '"correction": {"needed": false}}')
    tutor._lint("Jien mill-Awstralja.", data)
    assert data["correction"]["needed"] is False
    assert data["correction"]["issues"] == []


def test_hyphen_split_keeps_display_form():
    pairs = text.units("mill-Awstralja")
    assert [d for d, _ in pairs] == ["mill-", "Awstralja"]
    assert [k for _, k in pairs] == ["mill", "awstralja"]


def test_diff_marks_missing_and_extra_words():
    diff = text.diff_words("Jien Malta", "Jien minn Malta")
    ops = [d["op"] for d in diff]
    assert "ins" in ops                       # "minn" was omitted
    assert diff[0]["op"] == "equal"


@pytest.mark.parametrize("word,expected", [
    ("dar", "id-dar"),
    ("raġel", "ir-raġel"),
    ("xemx", "ix-xemx"),
    ("sena", "is-sena"),
    ("żmien", "iż-żmien"),
    ("ċavetta", "iċ-ċavetta"),
    ("baħar", "il-baħar"),
    ("ktieb", "il-ktieb"),
    ("omm", "l-omm"),
    ("ilma", "l-ilma"),
    ("għodwa", "l-għodwa"),
    ("skola", "l-iskola"),
])
def test_definite_article_assimilation(word, expected):
    assert text.definite(word) == expected


# ── FSRS ───────────────────────────────────────────────────────────────────

def test_new_card_grades_into_learning():
    c = srs.CardState()
    c = srs.review(c, srs.GOOD)
    assert c.state == "learning"
    assert c.reps == 1
    assert c.stability > 0


def test_easy_first_answer_skips_learning_steps():
    c = srs.review(srs.CardState(), srs.EASY)
    assert c.state == "review"
    assert (c.due - c.last_review) > timedelta(days=1)


def test_intervals_grow_with_successful_reviews():
    c = srs.CardState()
    at = srs.now()
    c = srs.review(c, srs.EASY, at=at)
    first = c.stability
    for _ in range(4):
        at = c.due
        c = srs.review(c, srs.GOOD, at=at)
    assert c.stability > first
    assert c.state == "review"


def test_again_lapses_and_shortens_stability():
    c = srs.review(srs.CardState(), srs.EASY)
    c = srs.review(c, srs.GOOD, at=c.due)
    before = c.stability
    c = srs.review(c, srs.AGAIN, at=c.due)
    assert c.lapses == 1
    assert c.state == "relearning"
    assert c.stability < before


def test_initial_difficulty_is_ordered_and_not_pinned_to_a_bound():
    """Regression: mismatched FSRS weights/formula silently clamped every grade to
    1.0, which made every card look maximally easy forever."""
    d = [srs._init_difficulty(g) for g in (1, 2, 3, 4)]
    assert d == sorted(d, reverse=True), "harder grades must give higher difficulty"
    assert all(1.0 < x < 10.0 for x in d), f"difficulty pinned to a bound: {d}"
    assert d[0] - d[3] > 1.0, "grades barely separated"


def test_initial_stability_is_ordered():
    s = [srs._init_stability(g) for g in (1, 2, 3, 4)]
    assert s == sorted(s)
    assert s[3] > s[0] * 5


def test_interval_matches_stability_at_default_retention():
    # The defining property of the FSRS power-law curve at r = 0.9.
    for stability in (5, 20, 90):
        assert srs.interval_for(stability, 0.9) == pytest.approx(stability, rel=0.02)


def test_difficulty_stays_in_range():
    c = srs.CardState()
    at = srs.now()
    c = srs.review(c, srs.AGAIN, at=at)
    for grade in (1, 1, 1, 4, 4, 4, 1, 3):
        at = c.due
        c = srs.review(c, grade, at=at)
        assert 1.0 <= c.difficulty <= 10.0


def test_retrievability_decays():
    assert srs.retrievability(0, 10) == pytest.approx(1.0)
    assert srs.retrievability(10, 10) < 0.95
    assert srs.retrievability(100, 10) < srs.retrievability(10, 10)


def test_higher_target_retention_means_shorter_intervals():
    assert srs.interval_for(30, 0.95) < srs.interval_for(30, 0.80)


def test_preview_offers_all_four_grades():
    p = srs.preview(srs.CardState())
    assert set(p) == {1, 2, 3, 4}
    assert all(isinstance(v, str) and v for v in p.values())


# ── Deck integrity ─────────────────────────────────────────────────────────

def test_decks_parse_and_have_unique_ids():
    vocab = curriculum._read_tsv(curriculum.VOCAB_TSV)
    phrases = curriculum._read_tsv(curriculum.PHRASES_TSV)
    assert len(vocab) > 250
    assert len(phrases) > 100
    ids = [r["id"] for r in vocab + phrases]
    assert len(ids) == len(set(ids)), "duplicate card ids"
    for row in vocab + phrases:
        assert row["mt"] and row["en"], f"incomplete row: {row}"
        assert int(row["tier"]) in (1, 2, 3, 4, 5)


def test_scenarios_are_well_formed():
    scenarios = curriculum.load_scenarios()
    assert len(scenarios) >= 10
    for s in scenarios:
        for key in ("id", "name", "name_en", "tutor_role", "opener_mt", "opener_en"):
            assert s.get(key), f"{s.get('id')} missing {key}"


def test_grammar_notes_present():
    notes = curriculum.grammar_notes()
    assert "definite article" in notes.lower()
    assert "ma" in notes and "-x" in notes

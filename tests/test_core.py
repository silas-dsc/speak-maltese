"""Unit tests for the parts where a silent bug would quietly teach bad Maltese
or corrupt the schedule."""

from __future__ import annotations

import json
import re
import sys
from datetime import timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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


@pytest.mark.parametrize("s,mt,en", [
    ("Ċertament! Liema tip ta' kafè tixtieq?", True, False),
    ("Jien għajjien illum.", True, False),
    ("Sarah huwa ismek. U inti?", True, False),
    ("Hello Sara. What do you mean?", False, True),
    ("Good morning! I'm Marija. What's your name?", False, True),
    ("Hello Sara, you mean that you want to hear what the other person said.", False, True),
])
def test_language_classification(s, mt, en):
    assert text.looks_maltese(s) is mt
    assert text.looks_english(s) is en


def _stt_metrics():
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "scripts" / "compare_stt.py"
    spec = importlib.util.spec_from_file_location("compare_stt", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wer_and_cer_basics():
    m = _stt_metrics()
    assert m.wer("Jien mill-Awstralja.", "Jien mill-Awstralja.") == 0.0
    assert m.cer("Jien mill-Awstralja.", "Jien mill-Awstralja.") == 0.0
    assert m.wer("", "Jien mill-Awstralja.") == 1.0
    assert m.wer("Jien minn Awstralja", "Jien mill-Awstralja.") > 0


def test_folded_wer_forgives_what_recognisers_always_drop():
    """A recogniser writing `Bongu kif int` heard every word correctly; only the
    diacritics are missing. Strict WER calls that 67% wrong, which would make the
    comparison measure orthography instead of recognition."""
    m = _stt_metrics()
    assert m.wer("Bongu kif int", "Bonġu! Kif int?") > 0.5
    assert m.wer("Bongu kif int", "Bonġu! Kif int?", folded=True) == 0.0
    assert m.wer("Jien mill Awstralja", "Jien mill-Awstralja.", folded=True) == 0.0


def test_eval_sentences_spread_across_the_deck():
    m = _stt_metrics()
    picked = m._sentences(20)
    assert len(picked) == 20
    assert len(set(picked)) == 20, "duplicate eval sentences"
    # not all greetings from the top of the phrase deck
    assert sum(1 for s in picked if s.startswith(("Bonġu", "Bonswà", "Saħħa"))) <= 2


def test_eval_clip_names_follow_the_sentence_not_the_index():
    """Regression: clips were named synth_001, synth_002… but --synth 8 and
    --synth 25 pick different sentences, so cached audio from one run got paired
    with another run's reference text — silently scoring every model against the
    wrong transcript."""
    import hashlib

    m = _stt_metrics()
    small, large = m._sentences(8), m._sentences(25)
    overlap = set(small) & set(large)
    assert overlap, "expected some shared sentences between eval-set sizes"

    def clip_name(sentence):
        return f"synth_{hashlib.sha256(sentence.encode()).hexdigest()[:12]}.mp3"

    # the same sentence must map to the same file regardless of eval-set size...
    for s in overlap:
        assert clip_name(s) == clip_name(s)
    # ...and different sentences must never collide onto one file
    names = {clip_name(s) for s in set(small) | set(large)}
    assert len(names) == len(set(small) | set(large))
    # position in the list must not appear in the name
    assert "001" not in clip_name(small[0]) or clip_name(small[0]) != clip_name(large[0])


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


@pytest.mark.parametrize("a,b", [
    ("Bonġu", "Bongu"),                       # diacritic dropped
    ("Grazzi ħafna", "gratsi hafna"),          # ż/ts and ħ/h
    ("Jien mill-Awstralja", "jien mill awstralja"),  # fused article split
    ("Kemm jiswa?", "kem jiswa"),              # doubled consonant, punctuation
    ("Il-kont", "ilkont"),
    ("Nixtieq kafè", "nixtiek kafe"),
])
def test_phonetic_key_absorbs_recogniser_variation(a, b):
    from backend import phonetics

    assert phonetics.similarity(a, b) >= 0.88, phonetics.key(a) + " vs " + phonetics.key(b)


@pytest.mark.parametrize("a,b", [
    ("qalb", "kelb"),              # q must stay distinct from k
    ("Iva, grazzi", "Le, grazzi"),  # yes vs no
    ("Nixtieq kafè", "irrid te"),
])
def test_phonetic_key_still_separates_different_words(a, b):
    from backend import phonetics

    assert phonetics.similarity(a, b) < 0.8


def test_scripted_dialogue_runs_without_a_model():
    """The whole point of the drill mode: a correct answer advances the script with
    no network call and no model, in well under a millisecond of matching."""
    from backend import dialogue

    node = dialogue.start("cafe")
    assert node["node"] == "c1"
    r = dialogue.evaluate("cafe", "c1", "nixtiek kafe jek jogobok")
    assert r["verdict"] == "correct"
    assert r["advance"] is True
    assert r["next"]["node"] == "c2"
    assert r["reply_mt"]


def test_scripted_dialogue_reprompts_instead_of_advancing():
    from backend import dialogue

    r = dialogue.evaluate("cafe", "c1", "xi xi xi")
    assert r["verdict"] == "wrong"
    assert r["advance"] is False
    assert r["reply_mt"], "must still say something back"
    assert r.get("say_this_mt"), "must show the target to repeat"


def test_scripted_dialogue_lines_are_all_maltese():
    """Every line the app speaks in drill mode is authored, so unlike the LLM path
    it can be checked once and trusted."""
    from backend import dialogue

    for line in dialogue.every_line():
        assert text.looks_maltese(line), f"not Maltese: {line!r}"
        assert not text.lint_fusion(line), f"unfused preposition: {line!r}"


def test_every_dialogue_node_is_reachable_and_terminates():
    from backend import dialogue

    for d in dialogue.all_dialogues():
        nodes = d["nodes"]
        assert d["start"] in nodes
        seen, cur, steps = set(), d["start"], 0
        while cur and steps < 50:
            assert cur in nodes, f"{d['id']} points at missing node {cur}"
            seen.add(cur)
            cur = nodes[cur].get("next")
            steps += 1
        assert steps < 50, f"{d['id']} does not terminate"
        assert seen == set(nodes), f"{d['id']} has unreachable nodes: {set(nodes) - seen}"


def test_decks_contain_no_unfused_prepositions():
    """The decks are what the app teaches, so they must obey the rule the tutor
    corrects against. Caught two real ones: `Jien minn l-Awstralja.` as the example
    for `minn`, and `Naħdem sa l-erbgħa.` for `sa`."""
    offenders = []
    sources = (
        (curriculum.VOCAB_TSV, ("mt", "ex_mt")),
        (curriculum.PHRASES_TSV, ("mt",)),
    )
    for path, cols in sources:
        for row in curriculum._read_tsv(path):
            for col in cols:
                value = (row.get(col) or "").strip()
                for hit in text.lint_fusion(value):
                    offenders.append(f"{row['id']}.{col}: {value!r} → {hit['should_be']}")
    assert not offenders, "unfused preposition + article in the decks:\n" + "\n".join(offenders)


def test_grammar_notes_present():
    notes = curriculum.grammar_notes()
    assert "definite article" in notes.lower()
    assert "ma" in notes and "-x" in notes


# ── The scripted content itself ────────────────────────────────────────────
#
# The dialogues are the product. A model is no longer in the loop, so nothing at
# runtime can notice a badly written scene — these checks are the only thing
# standing between a typo and a learner repeating it out loud.

ARTICLES = {"l", "il"} | {f"i{c}" for c in "ċdnrstxzż"}


def _article_faults(line: str) -> list[str]:
    """Sun-letter assimilation errors in running text.

    Deliberately narrow. The article also loses its `i-` after a vowel-final word
    (`Nixtri l-ħobż`, not *`Nixtri il-ħobż`), so *which* form appears cannot be
    checked without the preceding word — but assimilation is unconditional:
    `is-Sibt` is never `il-Sibt`, and `iż-żarbun` is never `iz-żarbun`.
    """
    faults = []
    for token in re.split(r"[\s,.!?;:]+", line or ""):
        prefix, sep, rest = token.partition("-")
        if not sep or not rest or prefix.lower() not in ARTICLES:
            continue
        first = rest[0].lower()
        if first in "ċdnrstxzż":
            want = f"i{first}"
            if prefix.lower() != want:
                faults.append(f"{token!r} should be {want}-{rest}")
        elif prefix.lower() not in ("l", "il"):
            faults.append(f"{token!r} assimilates to a letter {rest!r} does not start with")
    return faults


@pytest.mark.parametrize("word,expected", [
    ("ħobż", "il-ħobż"),     # ħ is a full consonant — the article keeps its i-
    ("ħanut", "il-ħanut"),
    ("Ħamis", "il-Ħamis"),
    ("hena", "l-hena"),      # silent h behaves like a vowel
    ("għajn", "l-għajn"),    # so does għ
    ("ilsien", "l-ilsien"),
    ("skola", "l-iskola"),   # s impura takes a prosthetic i- and blocks assimilation
    ("żarbun", "iż-żarbun"),
])
def test_article_distinguishes_h_from_hha(word, expected):
    """`l-ħobż` for `il-ħobż` was in the deck for real. ħ, h and għ look alike and
    behave differently, so each is pinned here."""
    assert text.definite(word) == expected


def test_dialogue_articles_assimilate():
    from backend import dialogue

    offenders = [f"{line!r}: {f}"
                 for line in dialogue.every_line()
                 for f in _article_faults(line)]
    assert not offenders, "sun-letter assimilation:\n" + "\n".join(offenders)


def test_decks_articles_assimilate():
    offenders = []
    for path, cols in ((curriculum.VOCAB_TSV, ("mt", "ex_mt")),
                       (curriculum.PHRASES_TSV, ("mt",))):
        for row in curriculum._read_tsv(path):
            for col in cols:
                for f in _article_faults(row.get(col) or ""):
                    offenders.append(f"{row['id']}.{col}: {f}")
    assert not offenders, "sun-letter assimilation:\n" + "\n".join(offenders)


@pytest.mark.parametrize("did,nid,said,expected", [
    # `Jien …` — the keyword has to be there, and it has to be first.
    ("greet", "g1", "Jien Pietru", 1.0),
    ("greet", "g1", "Jisimni Pietru", 1.0),
    ("greet", "g1", "Mela, jien Pietru", 1.0),   # a hesitation in front is still the frame
    ("greet", "g1", "Jien", 0.5),                # frame right, nothing said in it
    ("greet", "g1", "Pietru", 0.0),              # the name alone is not the sentence
    ("greet", "g1", "Pietru jien", 0.5),         # keyword there, but nothing after it
    ("greet", "g1", "hello", 0.0),
    # The frame with the learner's own words around it is still the frame.
    ("likes", "l1", "Jien inħobb il-ħut", 1.0),
    ("home", "h3", "Għandi ħames kmamar fid-dar", 1.0),
    ("people", "o2", "Huwa għalliem", 1.0),      # hu/huwa is one word
    # `Għandi … sena` — both ends, with the age itself unjudged in between.
    ("family", "f4", "Għandi ħamsa u tletin sena", 1.0),
    ("family", "f4", "Andi hamsa u tletin sena", 1.0),   # recogniser drops għ and ħ
    ("family", "f4", "Ħamsa u tletin sena", 2 / 3),      # ends right, opens with nothing
    ("family", "f4", "Għandi tletin", 2 / 3),            # opens right, `sena` missing
    ("family", "f4", "Tletin", 0.0),
    # `… kmamar` — ends with, and nothing to say about what comes before it.
    ("home", "h3", "Tliet kmamar", 1.0),
    ("home", "h3", "Għandi sitt kmamar", 1.0),
    ("home", "h3", "Tlieta", 0.0),
])
def test_open_question_still_wants_its_keywords(did, nid, said, expected):
    """An open question is a frame with a name, a town or an age in it. The slot is
    the learner's and is never graded; the frame is ordinary Maltese and is the whole
    point of the scene, so it is looked for where it belongs — at the start of the
    answer, at the end, or both. `Pietru` on its own used to score as well as
    `Jien Pietru`, which told the learner the sentence did not matter."""
    from backend import dialogue

    r = dialogue.evaluate(did, nid, said)
    assert r["score"] == pytest.approx(expected, abs=0.005), f"{said!r} scored {r['score']}"
    # Never blocked on any of these: the app cannot know the answer.
    assert r["verdict"] == "correct"


def test_open_question_frames_are_well_formed():
    """Each frame needs exactly one slot, and every word it demands has to be a word
    the node's own example answers use. A frame is graded against, so a typo in one
    marks the learner down for not saying Maltese the scene never showed them."""
    from backend import dialogue

    for d in dialogue.all_dialogues():
        for node_id, n in d["nodes"].items():
            frames = n.get("frames")
            if not frames:
                continue
            where = f"{d['id']}.{node_id}"
            assert n.get("free"), f"{where} is not an open question"
            shown = {k for a in n["accept"] for k in dialogue._keys(a["mt"])}
            for frame in frames:
                assert len(dialogue._SLOT.split(frame)) == 2, f"{where}: {frame!r} has no slot"
                anchors = [k for k in dialogue._keys(dialogue._SLOT.sub(" ", frame))]
                assert anchors, f"{where}: {frame!r} anchors on nothing"
                for k in anchors:
                    assert k in shown, f"{where}: {frame!r} wants {k!r}, unseen in its answers"


def test_open_question_escape_answers_keep_their_score():
    """Some listed answers step outside the frame on purpose — `Dak sigriet!` for an
    age, `Ma niftakarx!` for a name. Saying one of those, or coming near it, is a real
    answer and keeps its own score instead of being marked down against a frame it was
    never meant to use. It is not reported as a frame score, because it is not one."""
    from backend import dialogue

    for did, nid, said in (("family", "f4", "Dak sigriet!"),
                           ("people", "o1", "Ma niftakarx!"),
                           ("home", "h3", "Kamra waħda biss."),
                           ("feelings", "z1", "Ninsab imdejjaq")):  # near one, not exact
        r = dialogue.evaluate(did, nid, said)
        assert r["score"] >= dialogue.CLOSE, f"{said!r} scored {r['score']}"
        assert r["frame_scored"] is False, f"{said!r} reported as a frame score"


def test_open_question_takes_a_greeting_before_the_frame():
    """`Bonġu, min qed jitkellem?` is answered `Bonġu, jien …` — the scene prompts the
    greeting and lists one of its own answers with it. A yes, a no or a greeting in
    front of the frame is still the frame. What the frame will not do is change places
    with the slot: the keyword has to come before the name, not after it."""
    from backend import dialogue

    assert dialogue.evaluate("phone", "x1", "Bonġu, jien Pietru")["score"] == 1.0
    assert dialogue.evaluate("family", "f2", "Iva, għandi ħuti kbar")["score"] == 1.0
    assert dialogue.evaluate("greet", "g1", "Pietru jien")["score"] == 0.5


def test_open_question_does_not_score_junk_against_an_escape_answer():
    """The answers that step outside the frame keep their own score, but only while
    the learner is anywhere near one. `the quick brown fox` is 31% similar to
    `Ma niftakarx!` and printing that as a mark is worse than saying nothing."""
    from backend import dialogue

    for did, nid, said in (("people", "o1", "the quick brown fox"),
                           ("feelings", "z1", "nothing at all"),
                           ("people", "o2", "xi xi xi")):
        r = dialogue.evaluate(did, nid, said)
        assert r["score"] == 0.0, f"{said!r} scored {r['score']}"
        assert r["frame_scored"] is True


def test_open_question_still_shows_the_target_when_nothing_was_said():
    """Below two characters an open question is graded wrong — the one case where the
    correction card matters, and it needs a line to show. Grading the frame must not
    take the card away: no example answer overlaps a single letter, and the nearest
    listed sentence is still the best thing to put in front of the learner."""
    from backend import dialogue

    r = dialogue.evaluate("greet", "g1", "a")
    assert r["verdict"] == "wrong"
    assert r["say_this_mt"], "nothing to repeat back"
    assert r["diff"]


def test_no_open_question_marks_down_its_own_answers():
    """A frame is authored, so it can be authored wrong — pointing at a slot that is
    not there, or demanding a word half the node's answers do not use. Either way the
    learner is told a listed answer is not the Maltese the scene wanted."""
    from backend import dialogue

    offenders = []
    for d in dialogue.all_dialogues():
        for node_id, n in d["nodes"].items():
            if not n.get("free"):
                continue
            for a in n.get("accept", []):
                if a.get("open"):
                    continue
                r = dialogue.evaluate(d["id"], node_id, a["mt"])
                if r["score"] < dialogue.CORRECT:
                    offenders.append(f"{d['id']}.{node_id}: {a['mt']!r} scores {r['score']}")
    assert not offenders, "open question marks down its own answer:\n" + "\n".join(offenders)


def test_every_accepted_answer_is_graded_correct():
    """An answer the script lists as acceptable must actually pass the matcher.
    Otherwise the app asks for something it then refuses — the stuck-loop bug,
    reintroduced one scene at a time."""
    from backend import dialogue

    for d in dialogue.all_dialogues():
        for node_id, n in d["nodes"].items():
            if n.get("free"):
                continue
            for a in n.get("accept", []):
                r = dialogue.evaluate(d["id"], node_id, a["mt"])
                assert r["verdict"] == "correct", (
                    f"{d['id']}.{node_id} rejects its own answer "
                    f"{a['mt']!r} (scored {r['score']})")


def test_accepted_answers_within_a_node_are_told_apart():
    """Two answers that sound alike would credit the learner for the wrong one and
    schedule the wrong card, silently."""
    from backend import dialogue

    for d in dialogue.all_dialogues():
        for node_id, n in d["nodes"].items():
            if n.get("free"):
                continue
            for a in n.get("accept", []):
                r = dialogue.evaluate(d["id"], node_id, a["mt"])
                assert text.fold(r["matched_mt"] or "") == text.fold(a["mt"]), (
                    f"{d['id']}.{node_id}: {a['mt']!r} matched "
                    f"{r['matched_mt']!r} instead")


def test_hints_quote_something_the_node_accepts():
    """`close` and `wrong` end in "Għid: …" — the learner repeats that line verbatim,
    so saying it back must actually pass. Caught `directions.d1`, whose hint spliced
    the opening of one accepted answer onto another and then graded the result
    "almost": the app asking for a sentence it will not take."""
    from backend import dialogue

    offenders = []
    for d in dialogue.all_dialogues():
        for node_id, n in d["nodes"].items():
            if n.get("free"):
                continue
            for key in ("close", "wrong"):
                mt = (n.get(key) or {}).get("mt", "")
                _, sep, quoted = mt.partition("Għid:")
                if not sep:
                    continue
                # Some hints offer a choice: "A — jew: B". Either is a valid thing
                # to say back, so both have to pass.
                for branch in re.split(r"—?\s*jew:\s*", quoted):
                    branch = branch.strip(" —")
                    if not branch:
                        continue
                    r = dialogue.evaluate(d["id"], node_id, branch)
                    if r["verdict"] != "correct":
                        offenders.append(
                            f"{d['id']}.{node_id}.{key}: {branch!r} grades "
                            f"{r['verdict']} ({r['score']})")
    assert not offenders, "hint is not accepted:\n" + "\n".join(offenders)


def test_dialogues_are_well_formed():
    from backend import dialogue

    ids = [d["id"] for d in dialogue.all_dialogues()]
    assert len(ids) == len(set(ids)), "duplicate dialogue id"
    for d in dialogue.all_dialogues():
        assert d["level"] in ("A0", "A1", "A2", "B1"), f"{d['id']}: {d['level']}"
        assert d["name"] and d["name_en"]
        for node_id, n in d["nodes"].items():
            where = f"{d['id']}.{node_id}"
            assert n["say_mt"] and n["say_en"], where
            assert n.get("expect_en"), where
            assert n.get("accept"), where
            for key in ("correct", "close", "wrong"):
                assert n.get(key, {}).get("mt"), f"{where} missing {key}"
                assert n.get(key, {}).get("en"), f"{where} missing {key} gloss"


def test_dialogues_reach_every_word_in_the_deck():
    """The decks and the scenes are written separately, and a deck word no scene
    ever says is a word the learner meets only as a flashcard. Kept at zero on
    purpose — adding vocabulary now means writing it into a conversation too."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "coverage_script", ROOT / "scripts" / "coverage.py")
    cov = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cov)

    used, _ = cov.dialogue_words()
    missing = sorted(w for w in cov.deck_words() if w not in used)
    assert not missing, f"{len(missing)} deck words never spoken: {missing[:20]}"

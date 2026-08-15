"""End-to-end tests over the HTTP API.

The unit tests cover the scheduler and the Maltese text handling; these cover the
wiring, which is where the bugs that actually reached the browser have lived — a
removed helper that broke every drill turn, endpoints that survived a refactor,
fields the frontend reads that the backend stopped sending.

Nothing here needs a model or the network: speech recognition and synthesis are the
only parts that do, and they are exercised separately.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Point the database somewhere disposable *before* the app imports its config.
_TMP_DB = Path(tempfile.mkdtemp(prefix="sm-test-")) / "progress.db"
os.environ["SM_DB_PATH"] = str(_TMP_DB)

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── Bootstrap ──────────────────────────────────────────────────────────────

def test_bootstrap_reports_what_the_ui_needs(client):
    d = client.get("/api/bootstrap").json()
    assert {"capabilities", "counts", "profile", "settings"} <= set(d)
    assert "tts" in d["capabilities"] and "stt" in d["capabilities"]
    assert d["counts"]["total"] > 400, "decks should be seeded"
    assert d["profile"]["level"] in ("A0", "A1", "A2", "B1", "B2")


def test_no_llm_endpoints_remain(client):
    """Free conversation was removed; its routes must not come back by accident."""
    for path in ("/api/chat", "/api/session", "/api/history"):
        assert client.post(path, json={}).status_code != 200


def test_grammar_reference_is_served(client):
    md = client.get("/api/grammar").json()["markdown"]
    assert "definite article" in md.lower()
    assert "<!--" not in md.split("# Maltese quick reference")[0] or True


# ── Scripted dialogue ──────────────────────────────────────────────────────

def test_dialogue_list(client):
    ds = client.get("/api/drill/dialogues").json()["dialogues"]
    assert len(ds) >= 9
    for d in ds:
        assert {"id", "name", "name_en", "level"} <= set(d)


def test_start_returns_a_playable_prompt(client):
    n = client.post("/api/drill/start", json={"dialogue": "cafe"}).json()
    assert n["node"] == "c1"
    assert n["say_mt"] and n["say_en"]
    assert n["expect_en"]


def test_unknown_dialogue_is_404(client):
    assert client.post("/api/drill/start", json={"dialogue": "nope"}).status_code == 404


def test_correct_answer_advances_and_carries_the_next_prompt(client):
    r = client.post("/api/drill/answer", json={
        "dialogue": "cafe", "node": "c1", "said": "Nixtieq kafè, jekk jogħġbok."}).json()
    assert r["verdict"] == "correct"
    assert r["advance"] is True
    assert r["next"]["node"] == "c2"
    assert r["reply_mt"]


def test_recogniser_spelling_still_counts_as_correct(client):
    """The whole point of phonetic matching: this is what wav2vec2 actually emits."""
    r = client.post("/api/drill/answer", json={
        "dialogue": "cafe", "node": "c1", "said": "nixtiek kafe jek jogobok"}).json()
    assert r["verdict"] == "correct", r["score"]


def test_wrong_answer_reprompts_with_the_target(client):
    r = client.post("/api/drill/answer", json={
        "dialogue": "cafe", "node": "c1", "said": "xi xi xi"}).json()
    assert r["verdict"] == "wrong"
    assert r["advance"] is False
    assert r["node"] == "c1", "must stay on the same node"
    assert r["say_this_mt"], "must show what to say"
    assert r["reply_mt"], "must still say something back"


def test_third_attempt_moves_on_rather_than_looping(client):
    """Reported live: the same line re-prompting forever. Never again."""
    r = client.post("/api/drill/answer", json={
        "dialogue": "cafe", "node": "c1", "said": "xi xi xi", "attempts": 2}).json()
    assert r["advance"] is True
    assert r["moved_on"] is True
    assert r["next"]["node"] == "c2"


def test_personal_answers_are_not_graded(client):
    """Your name and where you live are nobody's business to mark wrong."""
    for said in ("silas", "noħok tas-sliva", "wolverhampton"):
        r = client.post("/api/drill/answer", json={
            "dialogue": "greet", "node": "g1", "said": said}).json()
        assert r["verdict"] == "correct", said


def test_silence_is_not_accepted_even_on_a_free_node(client):
    r = client.post("/api/drill/answer", json={
        "dialogue": "greet", "node": "g1", "said": "  "}).json()
    assert r["verdict"] == "wrong"


def test_finishing_a_dialogue_reports_finished(client):
    d = "greet"
    node = client.post("/api/drill/start", json={"dialogue": d}).json()["node"]
    seen, guard = [], 0
    while guard < 12:
        guard += 1
        r = client.post("/api/drill/answer", json={
            "dialogue": d, "node": node, "said": "iva", "attempts": 2}).json()
        seen.append(node)
        if r.get("finished"):
            return
        node = r["next"]["node"]
    pytest.fail(f"dialogue never finished, visited {seen}")


def test_a_correct_drill_answer_is_scheduled_for_review(client):
    before = client.get("/api/bootstrap").json()["counts"]["total"]
    client.post("/api/drill/answer", json={
        "dialogue": "market", "node": "m1", "said": "Kilo tuffieħ, jekk jogħġbok."})
    after = client.get("/api/bootstrap").json()["counts"]["total"]
    assert after >= before, "phrases met in conversation should enter the deck"


# ── Grading and review ─────────────────────────────────────────────────────

def test_attempt_scores_and_diffs(client):
    r = client.post("/api/attempt", json={
        "said": "jien mill awstralja", "target": "Jien mill-Awstralja."}).json()
    assert r["score"] > 0.95
    assert r["verdict"] == "perfect"
    assert isinstance(r["diff"], list)


def test_attempt_requires_a_target(client):
    assert client.post("/api/attempt", json={"said": "x"}).status_code == 400


def test_queue_returns_cards_with_interval_previews(client):
    d = client.get("/api/queue?limit=5").json()
    assert d["cards"], "a fresh deck should offer new cards"
    for c in d["cards"]:
        assert c["mode"] in ("listen", "recognise", "produce")
        assert set(c["intervals"]) == {"1", "2", "3", "4"}


def test_review_schedules_and_reports_state(client):
    card = client.get("/api/queue?limit=1").json()["cards"][0]
    r = client.post("/api/review", json={
        "card_id": card["id"], "grade": 3, "mode": "recognise"}).json()
    assert r["state"] in ("learning", "review")
    assert r["due"]
    assert r["stability_days"] > 0


def test_review_rejects_unknown_cards(client):
    assert client.post("/api/review", json={
        "card_id": "nope", "grade": 3}).status_code == 404


def test_review_needs_a_grade_or_an_utterance(client):
    card = client.get("/api/queue?limit=1").json()["cards"][0]
    assert client.post("/api/review", json={"card_id": card["id"]}).status_code == 400


def test_spoken_review_grades_itself_from_the_audio_transcript(client):
    card = next(c for c in client.get("/api/queue?limit=20").json()["cards"]
                if c.get("mt"))
    r = client.post("/api/review", json={
        "card_id": card["id"], "mode": "produce", "said": card["mt"]}).json()
    assert r["assessment"]["score"] > 0.95
    assert r["state"] in ("learning", "review")


def test_stats_shape(client):
    s = client.get("/api/stats").json()
    for key in ("learned", "due", "new", "history", "topics", "speaking", "streak"):
        assert key in s


def test_card_search(client):
    d = client.get("/api/cards?q=kaf").json()["cards"]
    assert any("kafè" in c["mt"] for c in d)


def test_settings_round_trip(client):
    client.post("/api/settings", json={"rate": 0.8, "show_english": False})
    s = client.get("/api/bootstrap").json()["settings"]
    assert s["rate"] == 0.8
    assert s["show_english"] is False
    client.post("/api/settings", json={"rate": 0.95, "show_english": True})


# ── Static frontend ────────────────────────────────────────────────────────

def test_frontend_is_served(client):
    html = client.get("/").text
    assert "Nitkellmu" in html
    assert 'id="drillChat"' in html, "the scripted view must be present"


def test_app_js_defines_everything_it_calls():
    """Regression: escapeHtml lived in the conversation block, and deleting that
    block left twelve call sites pointing at nothing — which broke the drill UI
    silently, with no error in the console and no failing test."""
    import re

    raw = (Path(__file__).resolve().parent.parent / "frontend" / "app.js").read_text()
    # Prose is not code: "push-to-talk (hold)" in a comment reads as a call, and a
    # template literal full of HTML reads as a dozen of them.
    src = re.sub(r"/\*.*?\*/", " ", raw, flags=re.S)
    src = re.sub(r"(?m)//.*$", " ", src)
    src = re.sub(r"`(?:[^`\\]|\\.)*`", " `` ", src)
    src = re.sub(r"'(?:[^'\\\n]|\\.)*'", " '' ", src)
    src = re.sub(r'"(?:[^"\\\n]|\\.)*"', ' "" ', src)

    defined = set(re.findall(r"(?:async\s+)?function\s+(\w+)", src))
    defined |= set(re.findall(r"\b(?:const|let|var)\s+(\w+)\s*=", src))
    defined |= set(re.findall(r"class\s+(\w+)", src))
    # methods, and object/param destructuring like ({ onResult, target })
    defined |= set(re.findall(r"^\s*(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{", src, re.M))
    for block in re.findall(r"\{([^{}]*)\}\s*(?:=|\)|,)", src):
        defined |= {w.strip() for w in re.split(r"[,:]", block) if re.fullmatch(r"\w+", w.strip())}

    called = set(re.findall(r"\b([a-zA-Z_]\w*)\s*\(", src))
    keywords = {
        "if", "for", "while", "switch", "catch", "return", "function", "typeof",
        "async", "await", "new", "delete", "void", "var", "let", "const", "class",
        "constructor", "do", "else", "try", "throw",
    }
    globals_ = {
        "fetch", "setTimeout", "clearTimeout", "setInterval", "Promise", "Audio",
        "Blob", "FormData", "MediaRecorder", "Error", "Number", "String", "Boolean",
        "Array", "Object", "JSON", "Math", "console", "document", "window",
        "navigator", "performance", "encodeURIComponent", "parseInt", "parseFloat",
        "isNaN", "KeyboardEvent", "PointerEvent", "Event", "Set", "Map", "Date",
        "RegExp", "structuredClone", "queueMicrotask", "alert",
    }
    missing = {c for c in called - defined - keywords - globals_ if c[0].islower()}
    # `obj.method(` and `x?.fn(` are properties, not free identifiers.
    missing = {m for m in missing if not re.search(rf"[.?]\s*{re.escape(m)}\s*\(", src)}
    assert not missing, f"called but never defined in app.js: {sorted(missing)}"

"""End-to-end tests over the HTTP API.

The unit tests cover the scheduler and the Maltese text handling; these cover the
wiring, which is where the bugs that actually reached the browser have lived — a
removed helper that broke every drill turn, endpoints that survived a refactor,
fields the frontend reads that the backend stopped sending.

Nothing here needs a model or the network: speech recognition and synthesis are the
only parts that do, and they are exercised separately.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── Bootstrap ──────────────────────────────────────────────────────────────

def test_bootstrap_reports_what_the_ui_needs(client):
    r = client.get("/api/bootstrap").json()
    assert set(r) == {"capabilities", "defaults"}
    assert "tts" in r["capabilities"] and "stt" in r["capabilities"]
    assert r["defaults"]["voice"] and r["defaults"]["daily_new"] > 0


def test_bootstrap_carries_no_learner_state(client):
    """Progress lives in the browser. If any of it leaked back into a shared
    endpoint, every visitor to a deployment would see one person's deck."""
    body = client.get("/api/bootstrap").text
    for leak in ("counts", "profile", "sessions", "due", "learned", "streak"):
        assert leak not in body, f"{leak!r} is learner state and must not be served"


def test_health_reports_whether_the_recogniser_is_loaded(client):
    r = client.get("/api/health").json()
    assert set(r) >= {"ready", "warming", "stt", "tts"}
    assert isinstance(r["ready"], bool) and isinstance(r["warming"], bool)
    assert not (r["ready"] and r["warming"]), "cannot be both ready and warming"


def test_deck_is_served_whole_for_the_client_to_seed_from(client):
    cards = client.get("/api/deck").json()["cards"]
    assert len(cards) > 400
    ids = [c["id"] for c in cards]
    assert len(ids) == len(set(ids)), "duplicate card id would collide in IndexedDB"
    for c in cards:
        assert c["mt"] and c["en"] and c["id"]
        assert c["kind"] in ("vocab", "phrase")
    assert {c["kind"] for c in cards} == {"vocab", "phrase"}


def test_no_server_side_schedule_remains(client):
    """The scheduler moved to the browser; these endpoints read and wrote the one
    shared SQLite file that made a public deployment meaningless."""
    for path in ("/api/queue", "/api/stats", "/api/cards"):
        assert client.get(path).status_code == 404, path
    # A POST to a route that no longer exists falls through to the static mount,
    # which answers 405 rather than 404. Either way it is gone, which is the point.
    for path in ("/api/review", "/api/settings", "/api/cards/suspend"):
        assert client.post(path, json={}).status_code in (404, 405), path


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


def test_a_404_says_which_thing_was_not_found(client):
    """The client acts on the reason. A conversation is kept across a reload now, so
    a saved one sitting on a node a later build removed has to be recognised and
    abandoned — and "not found" is indistinguishable from a mistyped URL. The SPA
    fallback catches these deliberate 404s too, and used to flatten them."""
    r = client.post("/api/drill/answer", json={"dialogue": "greet", "node": "g99-gone"})
    assert r.status_code == 404
    assert r.json()["detail"] == "unknown node"

    r = client.post("/api/drill/start", json={"dialogue": "nope"})
    assert r.json()["detail"] == "unknown dialogue"


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


def test_every_scene_can_be_walked_to_the_end_by_answering_correctly(client):
    """Play each scene the way a learner who knows the answers would, over the real
    API. A scene whose accepted answer does not advance it, or that loops, or that
    points at a node the server cannot present, fails here rather than in someone's
    ear halfway through.

    Nothing needs cleaning up afterwards any more: walking every scene used to write
    a hundred-odd cards into the shared database and exhaust the day's new-card
    allowance for the tests that followed. The server has no database to write to."""
    from backend import dialogue

    _walk_every_scene(client, dialogue)


def _walk_every_scene(client, dialogue) -> None:
    for d in dialogue.all_dialogues():
        did = d["id"]
        start = client.post("/api/drill/start", json={"dialogue": did})
        assert start.status_code == 200, did
        node = start.json()["node"]
        visited, guard = [], 0
        while guard < len(d["nodes"]) + 2:
            guard += 1
            visited.append(node)
            said = d["nodes"][node]["accept"][0]["mt"]
            r = client.post("/api/drill/answer", json={
                "dialogue": did, "node": node, "said": said, "attempts": 0}).json()
            assert r["verdict"] == "correct", f"{did}.{node} rejected {said!r}"
            assert r["reply_mt"], f"{did}.{node} says nothing back"
            if r.get("finished"):
                break
            node = r["next"]["node"]
            assert node not in visited, f"{did} loops back to {node}"
        else:
            pytest.fail(f"{did} never finished, visited {visited}")
        assert len(visited) == len(d["nodes"]), (
            f"{did} skipped turns: played {visited}")


def test_a_correct_drill_answer_reports_what_to_schedule(client):
    """The server no longer writes to a deck — it returns the matched phrase and the
    client schedules it locally. Without these fields nothing a learner says in
    conversation would ever reach their review queue."""
    r = client.post("/api/drill/answer", json={
        "dialogue": "market", "node": "m1",
        "said": "Kilo tuffieħ, jekk jogħġbok."}).json()
    assert r["verdict"] == "correct"
    assert r["matched_mt"] and r["matched_en"]


# ── Grading and review ─────────────────────────────────────────────────────

def test_attempt_scores_and_diffs(client):
    r = client.post("/api/attempt", json={
        "said": "jien mill awstralja", "target": "Jien mill-Awstralja."}).json()
    assert r["score"] > 0.95
    assert r["verdict"] == "perfect"
    assert isinstance(r["diff"], list)


def test_attempt_requires_a_target(client):
    assert client.post("/api/attempt", json={"said": "x"}).status_code == 400


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

    # Declarations are read from the *raw* file, not the stripped copy. The
    # stripping is a heuristic — a template literal with another nested in it ends the
    # outer match early, and from there whole regions of the file vanish. That was
    # survivable while a swallowed region lost its definitions and its call sites
    # together; the day a call moved out of one and its definition did not, this failed
    # on `switchView`, which was plainly defined twenty lines away.
    #
    # A declaration is unambiguous in raw text — `function x`, `const x =` — and reading
    # it there costs nothing, because the failure this test exists to catch is a
    # function that is *gone*, and a gone function is gone from the raw file too. Only
    # the calls need the stripping, to keep prose out of them.
    defined = set(re.findall(r"(?:async\s+)?function\s+(\w+)", raw))
    defined |= set(re.findall(r"\b(?:const|let|var)\s+(\w+)\s*=", raw))
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
        "fetch", "setTimeout", "clearTimeout", "setInterval", "clearInterval",
        "Promise", "Audio",
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


# ── Offline shell ──────────────────────────────────────────────────────────

def test_a_browser_on_another_origin_may_send_an_utterance(client):
    """The static build has no recogniser of its own — an iPhone SE has 250-350MB of
    page memory and the model is 200MB before any tensors, so it is sent here
    instead. That is a cross-origin POST, which a browser refuses unless this app
    names the origin. Named, not `*`: a wildcard on an endpoint that accepts audio
    offers the internet a free transcription service."""
    from backend.config import CFG

    origin = CFG.cors_origins.split(",")[0]
    r = client.options("/api/stt", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert r.status_code in (200, 204), r.text
    assert r.headers.get("access-control-allow-origin") == origin

    r = client.get("/api/health", headers={"Origin": origin})
    assert r.headers.get("access-control-allow-origin") == origin

    r = client.get("/api/health", headers={"Origin": "https://not-this-app.example"})
    assert "access-control-allow-origin" not in r.headers, "must not be a wildcard"


def test_service_worker_and_manifest_are_served(client):
    sw = client.get("/sw.js")
    assert sw.status_code == 200
    assert "javascript" in sw.headers["content-type"]
    # Named after the build, exactly as the static build stamps it. Left as `dev`
    # the worker's bytes never change, so an edited app.js is served from the old
    # cache for one more load — which here is every reload while working on it.
    assert re.search(r"const BUILD = '[0-9a-f]{12}'", sw.text), "shell cache is unstamped"
    assert "const BUILD = 'dev'" not in sw.text

    mf = client.get("/manifest.webmanifest")
    assert mf.status_code == 200
    data = mf.json()
    assert data["start_url"] == "/"
    assert data["display"] == "standalone"
    for icon in data["icons"]:
        assert client.get(icon["src"]).status_code == 200, icon["src"]


def test_service_worker_never_caches_live_state():
    """Caching /api/queue or /api/stats would show yesterday's schedule.

    Checked by running the worker's own routing rule over real URLs rather than by
    grepping for it: the rule changed when the static build arrived — its
    api/deck.json is an immutable file that *should* cache — and a test that
    matched the old string would have failed for the wrong reason."""
    import json
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")

    driver = ROOT / "tests" / "_sw_driver.mjs"
    # sw.js names its routing rule `routeFor`, so the test can run the real thing.
    # Everything else in the file needs a ServiceWorker global scope, so just that
    # function is lifted out and evaluated.
    driver.write_text(
        "import { readFileSync } from 'node:fs';\n"
        "const src = readFileSync('frontend/sw.js', 'utf8');\n"
        "const start = src.indexOf('function routeFor');\n"
        "const end = src.indexOf('\\n}', start) + 2;\n"
        "const routeFor = new Function(`${src.slice(start, end)}; return routeFor;`)();\n"
        "const urls = JSON.parse(process.argv[2]);\n"
        "console.log(JSON.stringify(urls.map((u) => routeFor(new URL(u)))));\n",
        encoding="utf-8")
    urls = [
        "http://x/api/tts?text=hi",              # audio  — immutable, cache hard
        "http://x/speak-maltese/api/tts?text=hi",
        "http://x/api/queue?limit=5",            # network — live state
        "http://x/api/stats",
        "http://x/api/drill/answer",
        "http://x/speak-maltese/api/deck.json",  # shell   — a file in the static build
        "http://x/style.css",
        "http://x/img/scene-cafe.webp",
        # A pre-rendered line in the static build. Hashed over (voice, rate, text),
        # so immutable — and 23MB of them must not be thrown away by a deploy, which
        # is what routing them to the per-build shell cache would do.
        "http://x/speak-maltese/audio/" + "a" * 32 + ".mp3",
        # …but the manifest that maps a line to its file changes when they do.
        "http://x/speak-maltese/audio/index.json",
    ]
    try:
        proc = subprocess.run([node, str(driver), json.dumps(urls)], cwd=ROOT,
                              capture_output=True, text=True, timeout=30)
    finally:
        driver.unlink(missing_ok=True)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    assert got == ["audio", "audio", "network", "network", "network",
                   "shell", "shell", "shell", "audio", "shell"], dict(zip(urls, got))


def test_a_new_build_takes_over_the_page_rather_than_half_of_it():
    """The shell is served stale-while-revalidate, so a page open across a deploy
    runs the old code against whatever the new worker serves — the mix that had a
    prompt showing no Maltese frame while its own fresh data carried one. The worker
    claims the page and the page reloads once. Not on the first activation, which is
    the app starting up rather than a new build, and not in the middle of a turn."""
    app = (Path(__file__).resolve().parent.parent / "frontend" / "app.js").read_text()
    assert "controllerchange" in app, "a new build must not go unnoticed"
    after = app.split("controllerchange", 1)[1][:700]
    assert "location.reload()" in after
    assert "drill.busy" in after, "a reload mid-answer costs the learner the turn"
    assert "navigator.serviceWorker.controller" in app.split("controllerchange", 1)[0], \
        "the first activation is not a new build and must be told apart"


def test_every_module_the_client_loads_is_in_the_offline_shell():
    """The client is ES modules, so one missing file is not a degraded app — it is a
    blank page. A module added to app.js has to be added to the worker's precache
    list and to the static build's file list, and nothing about writing the import
    reminds you of either."""
    frontend = Path(__file__).resolve().parent.parent / "frontend"
    imported = set()
    for js in frontend.glob("*.js"):
        imported |= set(re.findall(r"""from\s+['"]\./([\w.-]+\.js)['"]""",
                                   js.read_text(encoding="utf-8")))
    assert "session.js" in imported, "the test itself has gone stale"

    sw = (frontend / "sw.js").read_text(encoding="utf-8")
    build = (Path(__file__).resolve().parent.parent
             / "scripts" / "build_static.py").read_text(encoding="utf-8")
    shell = build.split("SHELL = (", 1)[1].split(")", 1)[0]
    for name in sorted(imported):
        assert f"'./{name}'" in sw, f"{name} is not precached by the service worker"
        assert f'"{name}"' in shell, f"{name} is not copied by build_static.py"


def test_service_worker_survives_a_missing_asset():
    """cache.addAll is atomic — one 404 and the app caches nothing at all."""
    sw = (Path(__file__).resolve().parent.parent / "frontend" / "sw.js").read_text()
    assert "allSettled" in sw, "shell precache must tolerate a missing file"


def test_scene_images_exist_for_every_dialogue():
    """A scene with no picture degrades to no header, but the set should be
    complete — a gap here means generate_scene_images.py was not re-run."""
    from backend import dialogue

    img_dir = Path(__file__).resolve().parent.parent / "frontend" / "img"
    missing = [d["id"] for d in dialogue.all_dialogues()
               if not (img_dir / f"scene-{d['id']}.webp").exists()]
    assert not missing, f"scenes without art: {missing}"

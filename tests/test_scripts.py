"""The scripts in `scripts/` are build tooling, not app code, so nothing exercises
them until someone runs one — and by then the content they produce is already stale.

`prebuild_audio.py --what all` had been broken for some time: it still called
`curriculum.load_scenarios()`, removed along with the free-conversation mode. The
failure was invisible because `--what drills` takes a different branch. These tests
import each script and call the part that reads the app's own data.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCRIPTS = ROOT / "scripts"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", [p.stem for p in sorted(SCRIPTS.glob("*.py"))])
def test_script_imports(name):
    """Importing runs everything above `main()`, which is where the module-level
    constants and prompt tables live."""
    load(name)


@pytest.mark.parametrize("what", ["all", "drills", "deck"])
def test_prebuild_audio_collects_lines(what):
    lines = load("prebuild_audio").lines_for(what)
    assert lines, f"--what {what} found nothing to render"
    assert all(isinstance(line, str) and line.strip() for line in lines)
    assert len(lines) == len(set(lines)), "the same line would be rendered twice"


def test_prebuild_audio_all_is_the_union_of_the_others():
    mod = load("prebuild_audio")
    every = set(mod.lines_for("all"))
    assert set(mod.lines_for("drills")) <= every
    assert set(mod.lines_for("deck")) <= every


def _mini_build(mod, out: Path) -> str:
    """Everything a device caches as the shell, minus the 23MB of MP3s."""
    mod.DIST = out
    out.mkdir(parents=True, exist_ok=True)
    mod.copy_shell()
    mod.write_api("")
    return mod.stamp_shell_version()


def test_every_line_the_app_speaks_has_audio_committed(tmp_path):
    """The MP3s are rendered by edge-tts and committed on purpose: a build that
    depends on an unofficial endpoint being up is a build that breaks without
    warning. Which means editing a line and not re-rendering it is a change that
    passes every other test, builds fine, and fails the deploy — the build refuses
    to ship a line the app cannot say.

    That is exactly how it went: five replies were rewritten, and the deploy stopped
    on missing audio. This is the check that belongs before the push."""
    from backend.config import AUDIO_CACHE, CFG

    mod = load("build_static")
    missing = [line for line in mod.wanted_lines()
               if not (AUDIO_CACHE / f"{mod.cache_key(line, CFG.azure_voice, 0.95)}.mp3").exists()]
    assert not missing, (
        f"{len(missing)} lines have no rendered audio — run "
        f"scripts/prebuild_audio.py --what all:\n  " + "\n  ".join(missing[:8]))


def test_the_build_can_point_the_client_at_a_remote_recogniser(tmp_path):
    """Recognition happens on the device by default, from the 2.1MB model in `stt/`.
    A deployment that would rather centralise it can still name somewhere to post to,
    and when it does the client stops loading a model of its own."""
    mod = load("build_static")

    mod.DIST = tmp_path / "local"
    mod.DIST.mkdir(parents=True)
    mod.write_api("")
    boot = json.loads((mod.DIST / "api" / "bootstrap.json").read_text(encoding="utf-8"))
    assert boot["stt_base"] == "", "no remote recogniser by default"
    assert boot["capabilities"]["stt"] == [], "the UI must not offer what is not there"

    mod.DIST = tmp_path / "remote"
    mod.DIST.mkdir(parents=True)
    mod.write_api("https://example-space.hf.space/")
    boot = json.loads((mod.DIST / "api" / "bootstrap.json").read_text(encoding="utf-8"))
    assert boot["stt_base"] == "https://example-space.hf.space", "trailing slash trimmed"
    assert boot["capabilities"]["stt"] == ["remote"]


def test_the_client_prefers_the_remote_recogniser_where_one_is_named(tmp_path):
    """A build pointed at `stt_base` does not also load a model on the device — not at
    startup, not in the background, and not to answer with.

    This mattered more when the model was 200MB and could kill the tab; at 2.1MB it is
    only about not doing the same work twice. The startup branch is still the startup
    screen's: it asks for the on-device model in the `else`."""
    splash_js = (ROOT / "frontend" / "splash.js").read_text(encoding="utf-8")
    assert "if (boot.stt_base) {" in splash_js
    assert "} else if (onModel) {" in splash_js, "the model load must be the else"

    app_js = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    for guard in ("if (!remoteStt() && state.settings.local_stt && nanostt.isReady())",
                  "if (!remoteStt() && state.settings.local_stt && nanostt.supported()"):
        assert guard in app_js, f"missing: {guard}"
    assert app_js.count("remoteStt()") >= 4


def test_the_static_build_ships_the_recogniser(tmp_path):
    """2.1MB is small enough to serve from our own origin, which is the whole reason
    the model is in the repository. A build that copies the shell and forgets `stt/`
    loads, seeds the deck, and then has a mic that silently never works."""
    mod = load("build_static")
    mod.DIST = tmp_path / "dist"
    mod.DIST.mkdir(parents=True)
    mod.copy_shell()

    assert (mod.DIST / "nanostt.js").exists()
    model = mod.DIST / "stt" / "model.onnx"
    assert model.exists(), "the static build has no recogniser"
    assert (mod.DIST / "stt" / "vocab.txt").exists(), "weights without a vocabulary"
    # Comfortably inside GitHub's 100MB per-file limit, which the old model was not.
    assert model.stat().st_size < 20_000_000, f"{model.stat().st_size / 1e6:.0f}MB"


def test_build_stamps_the_shell_cache_with_the_build(tmp_path):
    """The worker serves the shell stale-while-revalidate, so a page open across a
    deploy mixes builds — the new `api/dialogues.json` against the previous
    `app.js`, which showed up as a prompt whose Maltese frame had gone missing. The
    cache is named after the build so the old one becomes unreachable instead."""
    mod = load("build_static")
    build = _mini_build(mod, tmp_path / "dist")

    assert re.fullmatch(r"[0-9a-f]{12}", build), build
    sw = (tmp_path / "dist" / "sw.js").read_text(encoding="utf-8")
    assert f"const BUILD = '{build}'" in sw
    assert "const BUILD = 'dev'" not in sw, "the placeholder is still there"


def test_build_stamp_moves_only_when_the_build_does(tmp_path):
    """Named after the content, not the commit: a deploy that changes nothing the
    device would notice must leave its cache — and its audio — alone."""
    mod = load("build_static")
    first = _mini_build(mod, tmp_path / "a")
    assert _mini_build(mod, tmp_path / "b") == first, "same input, different stamp"

    for name, edit in (("style.css", "\n/* changed */\n"),
                       ("api/dialogues.json", " ")):
        out = tmp_path / f"c-{name.replace('/', '-')}"
        _mini_build(mod, out)
        target = out / name
        target.write_text(target.read_text(encoding="utf-8") + edit, encoding="utf-8")
        shutil.copy(ROOT / "frontend" / "sw.js", out / "sw.js")   # a fresh placeholder
        mod.DIST = out
        assert mod.stamp_shell_version() != first, f"{name} changed and the stamp did not"


def test_audio_cache_is_not_versioned_per_build():
    """23MB of MP3s, and a sentence at a given voice and rate is the same file
    forever. Naming that cache after the build would re-download all of it every
    time a stylesheet changed."""
    sw = (ROOT / "frontend" / "sw.js").read_text(encoding="utf-8")
    assert "const AUDIO = 'audio-" in sw, "audio cache name must not interpolate BUILD"
    assert "`audio-${BUILD}`" not in sw


def test_every_scene_has_an_image_prompt():
    """A scene added without a prompt silently renders nothing, and the gap only
    shows up as a missing picture in the UI."""
    from backend import dialogue

    prompts = load("generate_scene_images").SCENES
    missing = [d["id"] for d in dialogue.all_dialogues() if d["id"] not in prompts]
    assert not missing, f"scenes with no image prompt: {missing}"


def test_coverage_report_runs():
    cov = load("coverage")
    used, lines = cov.dialogue_words()
    assert lines > 0 and used
    assert cov.deck_words()


def test_every_recording_prompt_has_a_pronunciation_guide():
    """Whoever records the evaluation clips is a Maltese *learner* — that is the whole
    premise of the app — so a prompt without a respelling, a meaning and something to
    listen to is a prompt that gets mispronounced, and a mispronounced clip measures the
    recogniser against the wrong sound.

    This fails when the deck changes and `data/pronunciation.tsv` is not updated with it,
    which is exactly when the gap would otherwise go unnoticed."""
    mod = load("compare_stt")
    en, say = mod._guide()
    prompts = mod._sentences(25)

    missing_say = [s for s in prompts if not say.get(s, "").strip()]
    assert not missing_say, ("no pronunciation in data/pronunciation.tsv for:\n  "
                            + "\n  ".join(missing_say))
    missing_en = [s for s in prompts if not en.get(s, "").strip()]
    assert not missing_en, f"no English gloss for: {missing_en}"

    # And the reference audio, which is what a non-speaker actually copies.
    from backend import tts
    from backend.config import AUDIO_CACHE, CFG
    silent = [s for s in prompts
              if not (AUDIO_CACHE / f"{tts._cache_key(s, CFG.azure_voice, 0.95, 'edge')}.mp3").exists()]
    assert not silent, f"nothing to listen to for: {silent}"

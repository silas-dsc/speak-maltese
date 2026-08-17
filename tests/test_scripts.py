"""The scripts in `scripts/` are build tooling, not app code, so nothing exercises
them until someone runs one — and by then the content they produce is already stale.

`prebuild_audio.py --what all` had been broken for some time: it still called
`curriculum.load_scenarios()`, removed along with the free-conversation mode. The
failure was invisible because `--what drills` takes a different branch. These tests
import each script and call the part that reads the app's own data.
"""

from __future__ import annotations

import importlib.util
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
    mod.write_api("/models/")
    return mod.stamp_shell_version()


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

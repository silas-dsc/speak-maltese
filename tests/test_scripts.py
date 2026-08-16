"""The scripts in `scripts/` are build tooling, not app code, so nothing exercises
them until someone runs one — and by then the content they produce is already stale.

`prebuild_audio.py --what all` had been broken for some time: it still called
`curriculum.load_scenarios()`, removed along with the free-conversation mode. The
failure was invisible because `--what drills` takes a different branch. These tests
import each script and call the part that reads the app's own data.
"""

from __future__ import annotations

import importlib.util
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

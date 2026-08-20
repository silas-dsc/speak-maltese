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


def test_the_field_is_ranked_on_the_rank_and_the_floor_on_the_confidence():
    """Two numbers doing two jobs, and it matters which is used where.

    `rank` carries the duration prior and is what the field is compared on. `confidence`
    is the acoustic fit alone and is what `MIN_CONFIDENCE` tests. Rank the field on
    `confidence` and the prior stops reaching the decision — which was worth four of the
    learner's twenty-five answers. Floor on `rank` and the bar for "is there speech here"
    starts moving with the length of whichever line was asked for."""
    app_js = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "r.rank > r.runnerUp + need" in app_js
    # `need` is the margin, and it must still bottom out at the swept constant however
    # the per-field scaling is set.
    assert "Math.max(MIN_MARGIN, MARGIN_SIGMAS * (r.fieldSd || 0))" in app_js
    assert "r.confidence >= MIN_CONFIDENCE" in app_js
    assert "r.confidence > r.runnerUp" not in app_js, (
        "the field is being ranked on the un-priored confidence again")


def test_the_margin_and_the_field_are_unchanged_until_swept():
    """`MARGIN_SIGMAS` and `FIELD_LOCAL` both change which answers are accepted, and
    neither has been priced against the 25 recordings and 90 negatives that chose every
    other constant in that block. At zero they are exactly the deployed rule: the margin
    is `MIN_MARGIN` alone, and the field is drawn from the whole script as it always was.

    Pinned because the failure mode is silent — a grader that has quietly become stricter
    marks correct answers wrong and feeds that into the FSRS scheduler."""
    app_js = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "const MARGIN_SIGMAS = 0;" in app_js
    assert "const FIELD_LOCAL = 0;" in app_js
    assert "const MIN_MARGIN = 0.02;" in app_js


def test_the_floor_came_down_with_the_prior_and_not_alone():
    """0.35 is only safe because `rankScore` charges for an implausible length. Lowering
    the floor without the prior admits hiss and buys nothing — measured, in the comment
    beside it. So the two are pinned together here: if the prior ever goes, this fails."""
    app_js = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    nano_js = (ROOT / "frontend" / "nanostt.js").read_text(encoding="utf-8")
    assert "const MIN_CONFIDENCE = 0.35;" in app_js
    assert "export function rankScore(" in nano_js
    assert "export function durationPrior(" in nano_js


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


def test_every_turn_can_show_and_say_its_answer():
    """The drill offers **Show me**, which puts a model answer on screen and speaks
    it. Two ways for that to become a button that does nothing, both silent:

    * a node whose accepted answers are *all* open frames (`Jisimni …`) has no line
      to show — a gap with an ellipsis in it is not a model of anything;
    * a line that exists but was never rendered to speech, which in the static build
      means the audio manifest has no entry and the player has nothing to play.

    `present()` picks a non-open answer for exactly the second reason — those are the
    entries `every_line()` synthesises — so this checks the guarantee end to end
    rather than trusting the comment that states it."""
    from backend import dialogue
    from backend.config import AUDIO_CACHE, CFG

    mod = load("build_static")
    silent, missing = [], []
    for d in dialogue.all_dialogues():
        for nid in (d.get("nodes") or {}):
            answer = (dialogue.present(d["id"], nid).get("answer") or {}).get("mt")
            if not answer:
                missing.append(f"{d['id']}/{nid}")
                continue
            key = mod.cache_key(answer, CFG.azure_voice, 0.95)
            if not (AUDIO_CACHE / f"{key}.mp3").exists():
                silent.append(f"{d['id']}/{nid}: {answer}")

    assert not missing, ("nodes with no answer to show — every `accept` entry is "
                        "open:\n  " + "\n  ".join(missing[:8]))
    assert not silent, ("answers with nothing to play — run "
                       "scripts/prebuild_audio.py --what all:\n  "
                       + "\n  ".join(silent[:8]))


def test_every_turn_has_a_picture_and_every_picture_a_turn():
    """The conversation shows a square beside each question, named after the turn:
    `img/turn-<scene>-<node>.webp`. A node without one degrades to no picture, which
    is right at runtime and wrong to ship — the whole point of a picture per turn is
    that it shows *this* moment, and a gap reads as a broken scene rather than as a
    scene with nothing to draw.

    The reverse matters too. Rename a node and its old picture is orphaned: 10KB
    nothing will ever request, and the next person to count them is misled about
    which turns are covered."""
    from backend import dialogue

    art = ROOT / "frontend" / "img"
    wanted = {f"turn-{d['id']}-{nid}.webp"
              for d in dialogue.all_dialogues() for nid in (d.get("nodes") or {})}
    present = {p.name for p in art.glob("turn-*.webp")}

    missing = sorted(wanted - present)
    assert not missing, (
        f"{len(missing)} turns have no picture — run scripts/generate_scene_images.py "
        f"--what turns:\n  " + "\n  ".join(missing[:8]))

    orphans = sorted(present - wanted)
    assert not orphans, (
        f"{len(orphans)} pictures belong to turns that no longer exist:\n  "
        + "\n  ".join(orphans[:8]))

    # Small enough that 113 of them are worth committing: the whole set is ~1.2MB,
    # against 23MB of audio already in here.
    biggest = max(art.glob("turn-*.webp"), key=lambda p: p.stat().st_size)
    assert biggest.stat().st_size < 60_000, \
        f"{biggest.name} is {biggest.stat().st_size / 1024:.0f}KB"


def test_every_turn_picture_was_asked_for_something_specific():
    """The prompts are generated once from the scene's setting and the turn's own
    English gloss, then committed to `data/turn_prompts.json` — so a re-render
    reproduces the same set rather than a new interpretation, and a prompt that came
    out silly can be edited by hand.

    Which makes the file the thing to check: a turn falling back to its scene's
    setting is a turn whose picture says "you are in a café" four times in a row,
    which is the picture-per-scene this replaced."""
    import json

    from backend import dialogue

    mod = load("generate_scene_images")
    prompts = json.loads(mod.PROMPTS_FILE.read_text(encoding="utf-8"))

    keys = {f"{d['id']}/{nid}"
            for d in dialogue.all_dialogues() for nid in (d.get("nodes") or {})}
    assert keys <= set(prompts), f"no prompt for: {sorted(keys - set(prompts))[:8]}"

    settings = set(mod.SCENES.values())
    generic = sorted(k for k in keys if prompts[k] in settings)
    assert not generic, f"{len(generic)} turns fell back to the scene: {generic[:6]}"

    # The image model renders lettering as garbage, so the brief forbids it.
    banned = sorted(k for k in keys
                    if any(w in prompts[k].lower()
                           for w in ("speech bubble", "lettering", "written words")))
    assert not banned, f"prompts that ask for text: {banned[:6]}"


def test_the_line_to_say_back_always_has_audio():
    """A near miss shows a line to say back, and that line now has its own Play button —
    so it has to be one that was rendered. It is always `_best_match`'s pick, and that
    skips open entries, which is what makes the guarantee hold: open entries are frames
    with a gap in them (`Jisimni …`) and are deliberately never synthesised.

    Checked rather than asserted from the code, because the two facts live in different
    files and neither mentions the other."""
    from backend import dialogue
    from backend.config import AUDIO_CACHE, CFG

    mod = load("build_static")
    silent = []
    for d in dialogue.all_dialogues():
        for nid, n in (d.get("nodes") or {}).items():
            for a in n.get("accept", []):
                if a.get("open"):
                    continue
                key = mod.cache_key(a["mt"], CFG.azure_voice, 0.95)
                if not (AUDIO_CACHE / f"{key}.mp3").exists():
                    silent.append(f"{d['id']}/{nid}: {a['mt']}")

    assert not silent, ("accepted answers with nothing to play — run "
                       "scripts/prebuild_audio.py --what all:\n  "
                       + "\n  ".join(silent[:8]))

    # And the matcher really does skip the open ones, which is the half of the
    # guarantee that lives in the engine.
    src = (ROOT / "backend" / "dialogue.py").read_text(encoding="utf-8")
    best = src.split("def _best_match(")[1].split("\ndef ")[0]
    assert 'if candidate.get("open"):' in best and "continue" in best


def test_no_rendered_line_was_cut_short():
    """"Some audio seems cut off" had two possible causes and only one was in the
    player. The other would be a truncated MP3, so: every line the app speaks, against
    how long a line of that length takes.

    Fitted rather than guessed, because a flat characters-per-second rule flags a third
    of the corpus — a one-word vocabulary card and a fourteen-word sentence do not read
    at the same rate. Over the 820 spoken lines:

        seconds ≈ 0.435 + 0.0368 × characters
        observed / predicted:  min 0.65 · median 0.97 · max 1.70

    and the distribution is continuous, with no gap at either end. The slowest are
    sentences of short exclamations with pauses between them (`Perfett. Grazzi ħafna!
    Ċaw!`, 1.70×) and the fastest are long flowing ones (0.65×). Nothing is truncated —
    a half-written file would sit at 0.2× and stand well clear of that floor, which is
    where the bar goes."""
    from backend import dialogue, tts
    from backend.config import AUDIO_CACHE, CFG

    BITRATES = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
    SAMPLE_RATES = {0: 44100, 1: 48000, 2: 32000}

    def seconds(path):
        """Sum the frame durations of an MPEG audio file — what a decoder would play,
        rather than what a header claims."""
        data = path.read_bytes()
        i, total = 0, 0.0
        if data[:3] == b"ID3":
            i = 10 + int.from_bytes(data[6:10], "big")
        while i + 4 <= len(data):
            if data[i] != 0xFF or (data[i + 1] & 0xE0) != 0xE0:
                i += 1
                continue
            bitrate = BITRATES[(data[i + 2] >> 4) & 0xF]
            rate = SAMPLE_RATES.get((data[i + 2] >> 2) & 0x3)
            if not bitrate or not rate:
                i += 1
                continue
            total += 1152 / rate
            i += max(144000 * bitrate // rate + ((data[i + 2] >> 1) & 1), 1)
        return total

    rows = []
    for line in set(dialogue.every_line()):
        path = AUDIO_CACHE / f"{tts._cache_key(line, CFG.azure_voice, 0.95, 'edge')}.mp3"
        if path.exists():
            rows.append((len(line), seconds(path), line))

    assert len(rows) > 500, f"only {len(rows)} lines measured"

    # Least squares over the corpus itself, so the expectation moves with the content.
    n = len(rows)
    sx = sum(r[0] for r in rows)
    sy = sum(r[1] for r in rows)
    sxx = sum(r[0] * r[0] for r in rows)
    sxy = sum(r[0] * r[1] for r in rows)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n

    short = [(secs / (intercept + slope * chars), secs, chars, line)
             for chars, secs, line in rows
             if secs < 0.45 * (intercept + slope * chars) or secs < 0.2]
    assert not short, (
        "lines with less audio than their length can account for — re-render with "
        "scripts/prebuild_audio.py --what all:\n  "
        + "\n  ".join(f"{r[0]:.2f}× ({r[1]:.2f}s for {r[2]} chars) {r[3]!r}"
                      for r in sorted(short)[:8]))


def test_duration_reads_a_wav_and_shrugs_off_anything_else(tmp_path):
    """A stalled microphone leaves a file behind, so length has to be measured.

    avfoundation opens the device before it delivers, and the first open of an iPhone
    over Continuity came back with 0.01 seconds of a 2-second request — one packet. The
    peak of that is a plausible-looking number, which is how a dead input gets reported
    as a quiet one and sends the speaker to the gain knob."""
    import sys
    import wave

    sys.path.insert(0, "scripts")
    import compare_stt as C

    good = tmp_path / "good.wav"
    with wave.open(str(good), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x01" * 8000)
    assert abs(C._duration(good) - 0.5) < 1e-6

    stalled = tmp_path / "stalled.wav"
    with wave.open(str(stalled), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x01" * 160)
    assert C._duration(stalled) < 0.5, "0.01s of audio must not read as a usable take"

    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not a wav")
    assert C._duration(junk) == 0.0, "an unreadable take is a take that failed"


def test_trim_caps_silence_and_never_loses_the_original(tmp_path):
    """Leading silence is measured in frames by the duration prior, so it is not cosmetic.

    Waiting for the input device to start put 0.86s in front of every take — 43 frames at
    50fps, larger than the prior's whole 28.28 intercept — while the clips recorded before
    that wait existed have 0.009s. Left alone, the two halves of the eval set differ by an
    artefact of the recorder."""
    import sys
    import wave

    sys.path.insert(0, "scripts")
    import compare_stt as C

    clips = tmp_path / "clips"
    clips.mkdir()
    quiet = b"\x00\x00" * 16000          # 1.0s of silence
    loud = b"\x00\x40" * 8000            # 0.5s of signal
    with wave.open(str(clips / "me_001.wav"), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(quiet + loud + quiet)

    before = C.CLIPS, C.MANIFEST
    try:
        C.use_clips_dir(clips)
        C.trim(0.20, 0.25)
        with wave.open(str(clips / "me_001.wav")) as w:
            trimmed = w.getnframes() / w.getframerate()
        with wave.open(str(clips / C.RAW / "me_001.wav")) as w:
            original = w.getnframes() / w.getframerate()
    finally:
        C.CLIPS, C.MANIFEST = before

    assert abs(original - 2.5) < 1e-3, "the untouched take has to survive somewhere"
    assert abs(trimmed - 0.95) < 0.02, f"0.5s of speech plus the pads, got {trimmed:.2f}s"

    # Reading from raw/ every time is what makes a second pass with a different pad
    # meaningful rather than compounding.
    try:
        C.use_clips_dir(clips)
        C.trim(0.05, 0.05)
        with wave.open(str(clips / "me_001.wav")) as w:
            again = w.getnframes() / w.getframerate()
    finally:
        C.CLIPS, C.MANIFEST = before
    assert abs(again - 0.60) < 0.02, f"trimming must not compound, got {again:.2f}s"

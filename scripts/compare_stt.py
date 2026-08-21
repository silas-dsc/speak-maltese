#!/usr/bin/env python3
"""A/B Maltese speech recognisers on the same clips.

Generic Whisper is weak on Maltese — there is very little of it in the training mix —
so a Maltese fine-tune should win. "Should" is not evidence, hence this.

Two ways to get an eval set:

    # 1. Synthetic: speak deck sentences with the app's own mt-MT voice.
    #    Zero effort, but TTS audio is cleaner and more regular than real speech,
    #    so treat the absolute numbers as optimistic and the *ranking* as the result.
    python scripts/compare_stt.py --synth 25

    # 2. Your own voice — what actually matters, since the app has to understand YOU.
    python scripts/compare_stt.py --record 20        # prompts you, records via ffmpeg
    python scripts/compare_stt.py                    # re-uses whatever is on disk

Then compare any set of CTranslate2 models:

    python scripts/compare_stt.py --models small,large-v3,\\
        carlosdanielhernandezmena/whisper-large-maltese-8k-steps-64h-ct2

Metrics
    WER / CER    standard, on normalised text
    folded WER   ignores the diacritics and the silent għ that recognisers always
                 drop — closer to "did it hear the right words"
    app score    `text.score`, the tolerant grader that actually decides whether the
                 learner is marked correct. This is the number that matters.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import csv
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import curriculum, text  # noqa: E402
from backend.config import DATA_DIR  # noqa: E402

CLIPS = DATA_DIR / "eval_clips"
MANIFEST = CLIPS / "manifest.tsv"


def use_clips_dir(path: Path) -> None:
    """Point the harness at a different evaluation set.

    Every number in this project was measured on the app's own TTS voices, which is the
    optimistic case and says nothing about a real learner. A directory of real speech
    with a matching manifest scores through exactly this code instead of a parallel
    script with its own subtly different metrics."""
    global CLIPS, MANIFEST
    CLIPS = path
    MANIFEST = path / "manifest.tsv"


# ── Metrics ────────────────────────────────────────────────────────────────

def _edit(a: list, b: list) -> int:
    """Levenshtein distance between two sequences."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(hyp: str, ref: str, folded: bool = False) -> float:
    r = (text.fold(ref) if folded else text.normalise(ref).lower()).split()
    h = (text.fold(hyp) if folded else text.normalise(hyp).lower()).split()
    return _edit(h, r) / len(r) if r else (0.0 if not h else 1.0)


def cer(hyp: str, ref: str) -> float:
    r = text.normalise(ref).lower()
    h = text.normalise(hyp).lower()
    return _edit(list(h), list(r)) / len(r) if r else (0.0 if not h else 1.0)


# ── Eval sets ──────────────────────────────────────────────────────────────

def _sentences(n: int) -> list[str]:
    """Phrases first — they are full utterances; then vocab example sentences."""
    raw = [r["mt"] for r in curriculum._read_tsv(curriculum.PHRASES_TSV)]
    raw += [r["ex_mt"] for r in curriculum._read_tsv(curriculum.VOCAB_TSV) if r.get("ex_mt")]
    # The phrase deck and the vocab examples overlap; duplicates would silently
    # weight a few sentences and shrink the eval set.
    seen, out = set(), []
    for s in raw:
        key = text.fold(s)
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    # spread across the deck rather than taking the first n, which are all greetings
    step = max(1, len(out) // n)
    return out[::step][:n]


async def synth(n: int, voice: str | None) -> None:
    from backend import tts

    CLIPS.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, sentence in enumerate(_sentences(n), 1):
        # Name by content hash, never by index: --synth 8 and --synth 25 select
        # different sentences, so an index-named cache would silently pair old audio
        # with new reference text and quietly corrupt every score.
        digest = hashlib.sha256(sentence.encode()).hexdigest()[:12]
        path = CLIPS / f"synth_{digest}.mp3"
        if not path.exists():
            audio, _ = await tts.synthesize(sentence, voice, rate=1.0)
            path.write_bytes(audio)
        rows.append({"file": path.name, "text": sentence})
        print(f"  {i:>3}/{n}  {sentence}")
    _write_manifest(rows)
    print(f"\n✓ {len(rows)} synthetic clips in {CLIPS}")


def bad_takes() -> list[tuple[str, str]]:
    """Recordings that measure the microphone rather than the speaker.

    The same two faults that cost 9 of the first 25 clips: one so far below the others as
    to be unusable, and eight clipped flat at full scale. Selecting them by measurement
    rather than by remembering which ones went wrong."""
    out = []
    for row in _read_manifest("voice"):
        peak = _peak(CLIPS / row["file"])
        if peak < 0.10:
            out.append((row["file"], f"too quiet (peak {peak:.2f})"))
        elif peak >= 0.99:
            out.append((row["file"], f"clipping (peak {peak:.2f})"))
    return out


def redo(spec: str, n: int) -> None:
    """Throw away recordings so they can be made again.

    `all`, `bad`, or a list like `2` / `2,7,19`. A take that came out wrong has to be
    removable or the only way back is deleting files by hand and hand-editing a TSV — and
    the numbering has to survive it, because `me_007.wav` is what pairs a clip with prompt
    7. So the file and its manifest row go, and nothing renumbers."""
    rows = _read_manifest()
    if spec.strip().lower() == "bad":
        faults = bad_takes()
        if not faults:
            print("no clips fail the level checks — nothing to redo")
            return
        for name, why in faults:
            print(f"  {name}  {why}")
        drop = {name for name, _ in faults}
    elif spec.strip().lower() == "all":
        drop = {r["file"] for r in rows if not r["file"].startswith("synth_")}
    else:
        try:
            wanted = {int(x) for x in spec.replace(" ", "").split(",") if x}
        except ValueError:
            sys.exit(f"--redo takes 'all' or numbers like 2,7,19 — got {spec!r}")
        bad = [i for i in wanted if not 1 <= i <= n]
        if bad:
            sys.exit(f"--redo out of range for {n} prompts: {sorted(bad)}")
        drop = {f"me_{i:03d}.wav" for i in wanted}

    gone = 0
    for name in sorted(drop):
        path = CLIPS / name
        if path.exists():
            path.unlink()
            gone += 1
    kept = [r for r in rows if r["file"] not in drop]
    _write_manifest(kept)
    print(f"cleared {gone} recording(s); {len([r for r in kept if not r['file'].startswith('synth_')])} left on disk")


def _guide() -> tuple[dict, dict]:
    """English glosses and an English-speaker respelling for each line.

    The respellings follow the sound table in `data/grammar_notes.md`, which is the same
    one the app's Guide tab shows: `x`=sh, `ġ`=j, `ż`=z, `z`=ts, `ħ`=a strong throaty h,
    `q`=a glottal stop (written `'`), `għ`=silent but lengthens the vowel beside it, `j`=y,
    `ie`=one long ee-eh. CAPITALS mark the stressed syllable, which in Maltese is usually
    the second to last.

    Kept in `data/pronunciation.tsv` rather than in this file so the phrasing can be fixed
    by somebody who actually speaks Maltese without touching the harness."""
    from backend import curriculum

    en = {}
    for tsv in (curriculum.PHRASES_TSV, curriculum.VOCAB_TSV):
        for r in curriculum._read_tsv(tsv):
            if r.get("mt"):
                en.setdefault(r["mt"], r.get("en", ""))
            if r.get("ex_mt"):
                en.setdefault(r["ex_mt"], r.get("ex_en") or r.get("en", ""))

    say = {}
    path = DATA_DIR / "pronunciation.tsv"
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if row.get("mt"):
                    say[row["mt"]] = row.get("say", "")
    return en, say


def _play(line: str) -> bool:
    """Play the app's own recording of the line, so there is something to mimic.

    Whoever is reading these prompts may not speak Maltese — that is the normal case for
    this app — and a phonetic respelling only goes so far. The audio is already on disk
    from `prebuild_audio.py`."""
    from backend import tts
    from backend.config import AUDIO_CACHE, CFG

    path = AUDIO_CACHE / f"{tts._cache_key(line, CFG.azure_voice, 0.95, 'edge')}.mp3"
    if not path.exists() or not shutil.which("afplay"):
        return False
    subprocess.run(["afplay", str(path)], check=False)
    return True


def print_guide(n: int) -> None:
    """The whole sheet at once, for reading before starting."""
    en, say = _guide()
    print(f"\n{n} lines to record. CAPITALS mark the stressed syllable.\n")
    for i, line in enumerate(_sentences(n), 1):
        print(f"{i:2}. {line}")
        print(f"    say:     {say.get(line, '(no guide yet)')}")
        print(f"    meaning: {en.get(line, '?')}\n")


ERRORS = "errors"

# What to actually do, per kind of mistake. The generator emits a code; a person needs a
# sentence. Kept here rather than in the prompt file so the wording can be fixed without
# regenerating a set that has already been half recorded.
HOW = {
    "geminate": "say that doubled letter ONCE, short",
    "ghajn": "sound the g in għ as a hard g — it should be silent",
    "hkbira": "say ħ as a plain English h, not from the throat",
    "ie": "shorten ie to a plain i",
    "zeta": "say ż as the z in zokkor, a ts sound",
    "xin": "say x as s, not sh",
    "dropped": "leave that word out — this is meant to be WRONG",
    "other": "say this different line instead — this is meant to be WRONG",
}


def record_errors(prompts: Path, device: str = ":default",
                  say_only: bool = False) -> None:
    """Record deliberate mispronunciations, labelled with the line they were meant to be.

    Every clip in `eval_clips` is an honest attempt, which makes one question unanswerable:
    a learner who drops a doubled consonant — is that caught, or does the audio still pass
    as the correct line? Scoring two spellings against one recording asks only whether the
    model *can* hear the difference. Detection needs audio of the error itself.

    So the prompt is the wrong spelling and the label is the right one, which is the whole
    point and also the one thing the ordinary recorder cannot express: it pairs clip N with
    deck line N by construction."""
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required for --record-errors (brew install ffmpeg)")
    with prompts.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        sys.exit(f"no prompts in {prompts} — write some with\n"
                 f"  python scripts/gemination.py --models frontend/stt "
                 f"--write-prompts {prompts}")

    out_dir = CLIPS / ERRORS
    out_dir.mkdir(parents=True, exist_ok=True)
    en, say_as = _guide()
    # `_guide` is keyed by the deck's own text; the prompt file stores the normalised
    # lowercase form the scorer uses. Looking one up with the other silently finds nothing,
    # which reads as "no guide exists" rather than as a mismatch.
    def _key(x: str) -> str:
        return text.normalise(x).lower().strip()
    say_as = {_key(k): v for k, v in say_as.items()}
    en = {_key(k): v for k, v in en.items()}
    # Before the microphone opens, so nothing stalls mid-session waiting on a network call.
    _ensure_audio([r["say"] for r in rows] + [r["intended"] for r in rows])
    lead = _input_lead(device)
    manifest = out_dir / "manifest.tsv"
    done = {}
    if manifest.exists():
        with manifest.open(encoding="utf-8") as fh:
            done = {r["file"]: r for r in csv.DictReader(fh, delimiter="\t")}

    kept = list(done.values())
    quiet = 0
    for i, row in enumerate(rows, 1):
        name = f"err_{i:03d}.wav"
        if name in done:
            continue
        print(f"\n[{i}/{len(rows)}]  say it WRONG:  {row['say']}")
        print(f"          the real line is:  {row['intended']}")
        # The instruction, not the audio, is what to follow where the two differ. For a
        # halved geminate the rendering differs from the truth by one to three percent of
        # duration — on two of twelve prompts the halved version came out *longer* — so
        # imitating it would be imitating noise. For the other kinds the mistake is a
        # different phoneme and the audio carries it.
        kind = row.get("kind") or ("geminate" if row.get("halved") else "")
        how = HOW.get(kind) or row.get("detail") or "say it as written above"
        mark = "  ← should be WRONG" if row.get("class") == "reject" else ""
        print(f"          {how}{mark}")
        guide = say_as.get(row["intended"])
        if guide:
            print(f"          the real line sounds like:  {guide}")
            print(f"          meaning: {en.get(row['intended'], '?')}")
        # Playing the intended line is help when the mistake is a sound inside it. On an
        # `other` prompt the intended line is a *different sentence* — unrelated by
        # construction — so it is delay rather than help, and 40 prompts of it is a wasted
        # quarter of an hour.
        pair = not say_only and kind != "other"
        print("          listen: the correct line, then a rough render of the error"
              if pair else "          listen: just the line to say")
        if pair:
            if not _play(row["intended"]):
                print("          (no audio for the correct line — run prebuild_audio.py)")
            time.sleep(0.4)
        if not _play(row["say"]):
            print(f"          ! no audio for {row['say']!r} — say it from the spelling, or "
                  f"skip this prompt")
        input("       Enter to hear it again, or arm the microphone with the next "
              "Enter… ")
        if pair:
            _play(row["intended"])
            time.sleep(0.4)
        _play(row["say"])
        input("       Enter to arm the microphone, then wait for 'speak now'… ")
        proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "avfoundation", "-i", device,
             "-ar", "16000", "-ac", "1", str(out_dir / name)],
            stdin=subprocess.PIPE,
        )
        time.sleep(lead)
        print("       ▶ speak now, then Enter to stop")
        input()
        proc.communicate(b"q")

        level = _peak(out_dir / name)
        held = _duration(out_dir / name)
        if held < 0.5:
            quiet += 1
            print(f"       ! only {held:.2f}s recorded — the input stalled, delete and redo")
        elif level < 0.10:
            quiet += 1
            print(f"       ! too quiet to use (peak {level:.2f}) — delete and redo")
        else:
            print(f"       ok (peak {level:.2f})")
        # `text` is what the grader will be asked to confirm, so it is the *intended*
        # line. `said` records what was actually spoken, which is what makes the clip a
        # negative rather than a recording with a typo in its label.
        # `class` decides what a correct grader does with the clip, so it has to survive
        # into the recording's own manifest — otherwise the set is 200 clips with no record
        # of which ones were supposed to be accepted.
        kept.append({"file": name, "text": row["intended"], "said": row["say"],
                     "halved": row.get("halved", ""),
                     "class": row.get("class", "accept"), "kind": kind})
        with manifest.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, delimiter="\t",
                               fieldnames=["file", "text", "said", "halved",
                                           "class", "kind"])
            w.writeheader()
            w.writerows(kept)

    print(f"\n✓ {len(kept)} deliberate errors in {out_dir}")
    if quiet:
        print(f"! {quiet} unusable — delete the file and its manifest row, then re-run")
    print("  measure whether the grader catches them:\n"
          "    python scripts/gemination.py --models frontend/stt --errors")


def _ensure_audio(lines: list[str]) -> int:
    """Render any of these lines the audio cache does not already hold.

    `_play` says why this matters: whoever reads these prompts may not speak Maltese, which
    is the normal case for this app. For deck lines `prebuild_audio.py` has already done it.
    A deliberately misspelled line has never been synthesised by anything, and it is the one
    that has to be heard — nobody can produce `kolox` rather than `kollox` from a spelling
    they cannot pronounce in the first place. Rendering the wrong spelling gives them the
    error to imitate instead of a rule to apply.

    Synthesised at the same voice and rate `_play` looks up, or the file would land under a
    different cache key and play nothing."""
    import asyncio

    from backend import tts
    from backend.config import AUDIO_CACHE, CFG

    made = 0
    todo = [ln for ln in dict.fromkeys(lines) if ln.strip() and not (
        AUDIO_CACHE / f"{tts._cache_key(ln, CFG.azure_voice, 0.95, 'edge')}.mp3").exists()]
    if not todo:
        return 0
    print(f"rendering {len(todo)} prompt(s) that have never been synthesised…")

    async def go() -> int:
        n = 0
        for ln in todo:
            try:
                await tts.synthesize(ln, CFG.azure_voice, rate=0.95)
                n += 1
            except Exception as exc:  # noqa: BLE001 — one bad line must not lose the rest
                print(f"  ! could not render {ln!r}: {exc}")
        return n

    made = asyncio.run(go())
    print(f"  rendered {made}/{len(todo)}")
    return made


def _input_lead(device: str) -> float:
    """Open the input once and measure how much of a take it swallows before samples flow.

    avfoundation hands ffmpeg the device before the device is ready, so the front of
    every recording is missing: a 3-second request comes back as 2.68 seconds on the
    built-in microphone, and the *first* open of an iPhone over Continuity is worse
    still — a 2-second request came back as 0.01 seconds, one packet, which reads as a
    dead device rather than a slow one. Both are the same fault, so measure it instead
    of guessing: this call pays the cold open, and what it returns is how long to wait
    after arming the microphone before telling anyone to speak.

    It doubles as a check on the device itself, which is worth having before fifty
    prompts rather than after: exact silence means the wrong input (BlackHole, Teams),
    and no audio at all means the device never started."""
    want = 3.0
    probe = CLIPS / ".probe.wav"

    def capture() -> tuple[int, str]:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "avfoundation", "-i", device,
             "-ar", "16000", "-ac", "1", "-t", str(want), str(probe)],
            capture_output=True, text=True,
        )
        return r.returncode, r.stderr
    rc, err = capture()
    # A cold Continuity session hands back silence rather than refusing: the phone is
    # advertised as an input whenever it is nearby, but samples only flow once the session
    # actually engages, and the first open is what engages it. Measured this morning, the
    # first open of the same microphone lost about two seconds; a little later it lost 0.3.
    # So a silent first capture means "not awake yet" far more often than "wrong device",
    # and telling someone to pick another input is the wrong advice.
    if rc == 0 and probe.exists() and _peak(probe) == 0.0:
        print(f"{device} returned silence on the first open — waking it and trying again")
        time.sleep(1.0)
        rc, err = capture()
    if rc != 0 or not probe.exists():
        sys.exit(f"could not record from {device}: {err.strip() or 'no output'}\n"
                 f"run --list-inputs to see what this machine has, and check the "
                 f"microphone is granted under System Settings > Privacy & Security")

    got = _duration(probe)
    level = _peak(probe)
    probe.unlink(missing_ok=True)
    if got < 0.5:
        sys.exit(f"{device} produced {got:.2f}s of a {want:.0f}s recording — it is not "
                 f"starting. If it is an iPhone over Continuity, wake and unlock the "
                 f"phone and try again; otherwise pick another --input.")
    if level == 0.0:
        sys.exit(f"{device} produced pure silence twice, not even a noise floor.\n"
                 f"  If it is an iPhone over Continuity: wake and unlock the phone, keep "
                 f"it near the Mac, and check it says it is connected — the microphone is "
                 f"advertised whenever the phone is nearby but only delivers once the "
                 f"session engages.\n"
                 f"  Otherwise this is a virtual device or a muted input rather than a "
                 f"microphone: run --list-inputs and pick another --input. Note the "
                 f"indices renumber whenever anything is plugged in or removed.")

    lead = max(0.5, (want - got) + 0.3)
    print(f"{device}: ready after {want - got:.2f}s, noise floor {level:.3f} — "
          f"waiting {lead:.1f}s before each take")
    return lead


def record(n: int, device: str = ":default") -> None:
    """Prompt for each sentence and record it from a microphone via ffmpeg.

    `:default` is whatever macOS currently calls the default input, which is not
    always a microphone — a machine with BlackHole or Teams audio installed can have
    a virtual device as its default and record twenty-five files of silence. Pass
    `--input :1` (see `--list-inputs`) to name one, and each clip is level-checked as
    it lands rather than at scoring time."""
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required for --record (brew install ffmpeg)")
    CLIPS.mkdir(parents=True, exist_ok=True)
    lead = _input_lead(device)
    en, say = _guide()
    rows = _read_manifest()
    existing = {r["file"] for r in rows}
    quiet = clipped = 0
    for i, sentence in enumerate(_sentences(n), 1):
        name = f"me_{i:03d}.wav"
        if name in existing:
            continue
        print(f"\n[{i}/{n}]  {sentence}")
        print(f"          say:     {say.get(sentence, '(no guide yet)')}")
        print(f"          meaning: {en.get(sentence, '?')}")
        if not _play(sentence):
            print("          (no reference audio — run scripts/prebuild_audio.py)")
        input("       Enter to hear it again, or arm the microphone with the next Enter… ")
        _play(sentence)
        input("       Enter to arm the microphone, then wait for 'speak now'… ")
        proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "avfoundation", "-i", device,
             "-ar", "16000", "-ac", "1", str(CLIPS / name)],
            stdin=subprocess.PIPE,
        )
        # The device is open but not yet delivering. Speaking into this gap loses the
        # first word, which is where the hardest sounds live — għ- and x- and ħ-.
        time.sleep(lead)
        print("       ▶ speak now, then Enter to stop")
        input()
        proc.communicate(b"q")

        level = _peak(CLIPS / name)
        held = _duration(CLIPS / name)
        # 0.01 was far too lenient: a take at peak 0.03 passed it, sat 30dB under every
        # other clip, and was the single worst-scoring recording in the set — both Whisper
        # models hallucinated on it. Anything this far down is unusable, not merely quiet.
        if held < 0.5:
            # Not a quiet take — a take that never happened. Saying "too quiet" here
            # sends someone to the gain knob for a fault in the device handshake.
            quiet += 1
            print(f"       ! only {held:.2f}s recorded — the input stalled, --redo {i}")
        elif level < 0.10:
            quiet += 1
            print(f"       ! too quiet to use (peak {level:.2f}) — move closer or raise "
                  f"the input gain, then --redo {i}")
        elif level >= 0.99:
            # Digital clipping. It cost 8 of the first 25 clips, all late in the run, as
            # the speaker leaned in — so it is worth saying at the time rather than
            # discovering it in the scores.
            clipped += 1
            print(f"       ! clipping (peak {level:.2f}) — lower the input gain, "
                  f"then --redo {i}")
        else:
            print(f"       ok (peak {level:.2f})")
        rows.append({"file": name, "text": sentence})
        _write_manifest(rows)
    if quiet or clipped:
        print(f"\n! {quiet} too quiet, {clipped} clipping. Re-record just those with "
              f"--redo <numbers> --record {n}; a bad clip measures the microphone rather "
              f"than the recogniser.")
    print(f"\n✓ {len(rows)} clips in {CLIPS}")
    print(f"  score them with:  --clips voice --models <model>")


RAW = "raw"
LEAD_PAD = 0.20
TRAIL_PAD = 0.25


def trim(lead_pad: float = LEAD_PAD, trail_pad: float = TRAIL_PAD) -> None:
    """Cap the silence at both ends of every recording, keeping the originals.

    Waiting for the input device to start puts about 0.86s of silence in front of every
    take. At 50 frames per second that is 43 frames — larger than the duration prior's
    whole intercept of 28.28 — so a clip recorded this way looks far too long for its
    token count and the prior penalises the right answer for it. The clips recorded
    before the wait existed have 0.009s of lead, so the two halves of the set would
    otherwise differ by an artefact of the recorder rather than by anything about speech.

    Nothing production sees looks like either extreme: a learner taps a button and
    speaks, and the app trims nothing. So both ends are capped at a plausible pad rather
    than shaved to the waveform, and the untouched files stay in `raw/` — the audio is
    the one thing here that cannot be regenerated."""
    import wave as wave_mod

    raw_dir = CLIPS / RAW
    raw_dir.mkdir(parents=True, exist_ok=True)
    changed = kept = 0
    for path in sorted(CLIPS.glob("*.wav")):
        backup = raw_dir / path.name
        if not backup.exists():
            shutil.copy2(path, backup)
        with wave_mod.open(str(backup)) as w:
            params = w.getparams()
            rate, width = w.getframerate(), w.getsampwidth()
            frames = w.readframes(w.getnframes())
        if width != 2:
            kept += 1
            continue
        samples = array.array("h")
        samples.frombytes(frames)
        if not samples:
            kept += 1
            continue

        peak = max(abs(s) for s in samples)
        if peak == 0:
            kept += 1
            continue
        # A tenth of the clip's own peak, so the cut follows the recording's own level
        # rather than an absolute number the quiet iPhone takes would fail.
        floor = peak * 0.10
        first = next((i for i, s in enumerate(samples) if abs(s) >= floor), 0)
        last = next((i for i in range(len(samples) - 1, -1, -1)
                     if abs(samples[i]) >= floor), len(samples) - 1)
        start = max(0, first - int(rate * lead_pad))
        end = min(len(samples), last + int(rate * trail_pad))
        if start == 0 and end == len(samples):
            kept += 1
            continue

        with wave_mod.open(str(path), "w") as w:
            w.setparams(params)
            w.setnframes(0)
            w.writeframes(samples[start:end].tobytes())
        changed += 1
        print(f"  {path.name}  {len(samples) / rate:.2f}s -> "
              f"{(end - start) / rate:.2f}s")

    print(f"\n✓ trimmed {changed}, left {kept} alone. Originals in {raw_dir}")
    print("  re-run with a different --trim-pad to change it; it always reads from raw/")


def _duration(path: Path) -> float:
    """Seconds of audio actually on disk.

    Deliberately stdlib `wave` rather than the decoder `_peak` uses: this is asked the
    same question about a device that may be broken, and it must not depend on anything
    CI does not install. A file that cannot be read counts as no audio, which is what a
    stalled input leaves behind."""
    import wave as wave_mod

    try:
        with wave_mod.open(str(path)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:  # noqa: BLE001 — an unreadable take is a take that failed
        return 0.0


def _peak(path: Path) -> float:
    """Loudest sample, 0..1. A recording of nothing is the one failure worth catching
    while the microphone is still open.

    numpy is imported here rather than at the top of the file on purpose: CI installs
    only what the app itself needs, and `tests/test_scripts.py` imports every script to
    check the parts above `main()` still work. A module-level import of anything from
    the modelling side turns that into a collection error."""
    try:
        import numpy as np

        from faster_whisper.audio import decode_audio
        wave = decode_audio(str(path), sampling_rate=16000)
        return float(np.abs(np.asarray(wave)).max()) if len(wave) else 0.0
    except Exception:  # noqa: BLE001 — a level check must not lose the recording
        return 1.0


def list_inputs() -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation",
                    "-list_devices", "true", "-i", ""], check=False)


def _write_manifest(rows: list[dict]) -> None:
    seen, uniq = set(), []
    for r in rows:
        if r["file"] not in seen:
            seen.add(r["file"])
            uniq.append(r)
    with MANIFEST.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, delimiter="\t", fieldnames=["file", "text"])
        w.writeheader()
        w.writerows(uniq)


def _read_manifest(which: str = "all") -> list[dict]:
    """`which`: all · synth (TTS clips) · voice (recorded ones).

    One manifest holds both, because `--record` appends to whatever is already there.
    Scoring them together would average a synthetic voice with a real one and report a
    single number for neither — and the synthetic clips are the optimistic half, so the
    mix would quietly flatter whatever is being tested."""
    if not MANIFEST.exists():
        return []
    with MANIFEST.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if r.get("file")]
    if which == "synth":
        return [r for r in rows if r["file"].startswith("synth_")]
    if which == "voice":
        return [r for r in rows if not r["file"].startswith("synth_")]
    return rows


# ── Comparison ─────────────────────────────────────────────────────────────

def _run_wav2vec2(name: str, rows: list[dict]) -> tuple[list[dict], float, float]:
    """CTC path. One forward pass over the real audio length — no decoder loop and
    no 30-second padding, which is where the order-of-magnitude comes from."""
    import torch
    from faster_whisper.audio import decode_audio
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.time()
    processor = Wav2Vec2Processor.from_pretrained(name)
    model = Wav2Vec2ForCTC.from_pretrained(name).to(device).eval()
    load_s = time.time() - t0

    results, start = [], time.time()
    for i, row in enumerate(rows, 1):
        path = CLIPS / row["file"]
        if not path.exists():
            continue
        wave = decode_audio(str(path), sampling_rate=16000)
        inputs = processor(wave, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        hyp = processor.batch_decode(torch.argmax(logits, dim=-1))[0]
        results.append(_score_row(hyp, row["text"]))
        print(f"  {i:>3}/{len(rows)}  score {results[-1]['score']:.2f}  {hyp[:58]}", flush=True)
    return results, load_s, time.time() - start


def _score_row(hyp: str, ref: str) -> dict:
    return {
        "ref": ref, "hyp": hyp,
        "wer": wer(hyp, ref), "fwer": wer(hyp, ref, folded=True),
        "cer": cer(hyp, ref), "score": text.score(hyp, ref),
    }


# ── NeMo CTC via ONNX Runtime ──────────────────────────────────────────────
# The interesting candidate for the browser is QuartzNet15x5, the same author's
# Maltese model at 18.9M parameters against wav2vec2-large's 315M — 76MB of fp32
# ONNX rather than 201MB of 4-bit, and convolution-only, so it does not need WebGPU.
#
# It is measured here rather than through `onnx-asr` because that library has no
# 64-mel preprocessor (only 80 and 128), and because the feature extraction written
# out longhand below is exactly what a browser port would have to reimplement. If it
# is wrong the transcripts are noise, so this doubles as the feasibility check.
#
# Every constant comes from the checkpoint's own `model_config.yaml`, read out of the
# .nemo archive, not from NeMo's defaults: `window_size: 0.02` is 320 samples, where
# the Conformer models everything else is written for use 400. Getting that one wrong
# costs accuracy quietly instead of failing.
_NEMO = {
    "sample_rate": 16000, "n_fft": 512, "win_length": 320, "hop_length": 160,
    "preemph": 0.97, "log_guard": float(2 ** -24),
}
# Slaney mel scale, as librosa builds it with htk=False — NeMo's default.
_F_SP = 200.0 / 3.0
_BREAK_HZ = 1000.0
_BREAK_MEL = _BREAK_HZ / _F_SP
_LOGSTEP = 0.0690875477931522   # log(6.4) / 27


def _mel_filters(n_freqs: int, n_mels: int, sample_rate: int):
    """Triangular mel filterbank with Slaney normalisation.

    Verified against `onnx-asr`'s reference implementation at 64, 80 and 128 bins:
    identical to 3e-8, which is below float32 resolution here."""
    import numpy as np

    def to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        # `where` evaluates both arms, so guard the log against f = 0 rather than
        # letting it warn and be discarded.
        safe = np.maximum(f, 1e-9)
        return np.where(f < _BREAK_HZ, f / _F_SP,
                        _BREAK_MEL + np.log(safe / _BREAK_HZ) / _LOGSTEP)

    def to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        return np.where(m < _BREAK_MEL, m * _F_SP,
                        _BREAK_HZ * np.exp(_LOGSTEP * (m - _BREAK_MEL)))

    pts = to_hz(np.linspace(to_mel(0), to_mel(sample_rate / 2), n_mels + 2))
    freqs = np.linspace(0, sample_rate / 2, n_freqs)
    fb = np.zeros((n_freqs, n_mels))
    for i in range(n_mels):
        lo, mid, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[:, i] = np.maximum(0.0, np.minimum((freqs - lo) / (mid - lo),
                                              (hi - freqs) / (hi - mid)))
    fb *= 2.0 / (pts[2:n_mels + 2] - pts[:n_mels])
    return fb.astype(np.float32)


def _nemo_features(wave, n_mels: int, fb, window):
    """Waveform → log-mel, normalised per feature. NeMo's `AudioToMelSpectrogram`."""
    import numpy as np

    cfg = _NEMO
    x = np.concatenate([wave[:1], wave[1:] - cfg["preemph"] * wave[:-1]])
    # torch.stft(center=True, pad_mode="reflect") — NeMo's default, and the edges of
    # a two-word answer are a real share of it.
    x = np.pad(x.astype(np.float32), cfg["n_fft"] // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(x, cfg["n_fft"])[::cfg["hop_length"]]
    spec = np.abs(np.fft.rfft(frames * window, n=cfg["n_fft"])) ** 2
    mel = np.log(spec.astype(np.float32) @ fb + cfg["log_guard"])
    # `normalize: per_feature` — per mel bin over time, sample variance as NeMo takes it
    mean = mel.mean(axis=0, keepdims=True)
    std = np.sqrt(mel.var(axis=0, keepdims=True, ddof=1))
    return ((mel - mean) / (std + 1e-5)).T[None].astype(np.float32)


def _run_nemo_ctc(name: str, rows: list[dict]) -> tuple[list[dict], float, float]:
    import json

    import numpy as np
    import onnxruntime as rt
    from faster_whisper.audio import decode_audio
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    src = Path(name)
    if src.is_dir():
        paths = {f: src / f for f in ("model.onnx", "vocab.txt", "config.json")}
    else:
        paths = {f: Path(hf_hub_download(name, f))
                 for f in ("model.onnx", "vocab.txt", "config.json")}
    cfg = json.loads(paths["config.json"].read_text(encoding="utf-8"))
    n_mels = int(cfg.get("features_size", 64))

    vocab, blank = {}, None
    for line in paths["vocab.txt"].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tok, idx = line.rsplit(" ", 1)
        vocab[int(idx)] = tok
        if tok == "<blk>":
            blank = int(idx)

    fb = _mel_filters(_NEMO["n_fft"] // 2 + 1, n_mels, _NEMO["sample_rate"])
    win = np.hanning(_NEMO["win_length"]).astype(np.float32)   # periodic=False
    pad = (_NEMO["n_fft"] - _NEMO["win_length"]) // 2
    window = np.pad(win, (pad, pad))
    sess = rt.InferenceSession(str(paths["model.onnx"]), providers=["CPUExecutionProvider"])
    load_s = time.time() - t0

    def decode(ids) -> str:
        """Merge repeated frames, *then* drop blanks. The other order degeminates,
        which in Maltese is the difference between `irrid` and `irid`."""
        out, prev = [], -1
        for i in ids:
            if i != prev:
                if i != blank:
                    out.append(vocab[int(i)])
                prev = i
        return "".join(out).replace("▁", " ").strip()

    results, start = [], time.time()
    for i, row in enumerate(rows, 1):
        path = CLIPS / row["file"]
        if not path.exists():
            continue
        wave = np.asarray(decode_audio(str(path), sampling_rate=_NEMO["sample_rate"]),
                          dtype=np.float32)
        feats = _nemo_features(wave, n_mels, fb, window)
        logprobs, = sess.run(["logprobs"], {"audio_signal": feats})
        hyp = decode(logprobs[0].argmax(-1))
        results.append(_score_row(hyp, row["text"]))
        print(f"  {i:>3}/{len(rows)}  score {results[-1]['score']:.2f}  {hyp[:58]}",
              flush=True)
    return results, load_s, time.time() - start


def _report(name: str, results: list[dict], load_s: float, elapsed: float) -> dict:
    """One shape for every backend, so the table compares like with like."""
    n = len(results) or 1
    return {
        "model": name, "n": len(results), "load_s": load_s,
        "sec_per_clip": elapsed / n,
        "wer": sum(r["wer"] for r in results) / n,
        "fwer": sum(r["fwer"] for r in results) / n,
        "cer": sum(r["cer"] for r in results) / n,
        "score": sum(r["score"] for r in results) / n,
        # The app's own threshold: below this a learner is told to try again.
        "pass_rate": sum(1 for r in results if r["score"] >= 0.78) / n,
        "results": results,
    }


def run_model(name: str, rows: list[dict], device: str, beam: int) -> dict:
    # wav2vec2 checkpoints are CTC, not Whisper — different loader entirely.
    if "wav2vec2" in name or "w2v" in name:
        print(f"\n▸ loading {name}  (CTC)", flush=True)
        return _report(name, *_run_wav2vec2(name, rows))

    # NeMo CTC exports: ONNX Runtime, and the features built here rather than by a
    # processor that ships with the checkpoint. Recognised by what is in the directory
    # rather than by what it is called, so a distilled student under data/distill scores
    # through the same path as the Hub export it is being compared against.
    if (Path(name).is_dir() and (Path(name) / "model.onnx").exists()) \
            or "quartznet" in name.lower() or "nemo" in name.lower():
        print(f"\n▸ loading {name}  (NeMo CTC, ONNX)", flush=True)
        return _report(name, *_run_nemo_ctc(name, rows))

    from faster_whisper import WhisperModel

    print(f"\n▸ loading {name}  (first run downloads it)", flush=True)
    t0 = time.time()
    compute = "int8" if device == "cpu" else "float16"
    model = WhisperModel(name, device=device, compute_type=compute)
    load_s = time.time() - t0

    results, t_start = [], time.time()
    for i, row in enumerate(rows, 1):
        path = CLIPS / row["file"]
        if not path.exists():
            continue
        segments, _ = model.transcribe(str(path), language="mt", beam_size=beam,
                                       vad_filter=False)
        hyp = " ".join(s.text for s in segments).strip()
        results.append(_score_row(hyp, row["text"]))
        print(f"  {i:>3}/{len(rows)}  score {results[-1]['score']:.2f}  {hyp[:58]}",
              flush=True)
    elapsed = time.time() - t_start
    del model
    return _report(name, results, load_s, elapsed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="small,carlosdanielhernandezmena/"
                                        "whisper-large-maltese-8k-steps-64h-ct2")
    ap.add_argument("--synth", type=int, metavar="N",
                    help="build N clips with the app's own Maltese TTS voice")
    ap.add_argument("--record", type=int, metavar="N",
                    help="record N clips in your own voice")
    ap.add_argument("--input", default=":default",
                    help="ffmpeg avfoundation input for --record, e.g. ':1'")
    ap.add_argument("--list-inputs", action="store_true",
                    help="show the microphones ffmpeg can see, then exit")
    ap.add_argument("--guide", type=int, metavar="N", default=None,
                    help="print the N lines to record, with pronunciation, then exit")
    ap.add_argument("--record-errors", type=Path, default=None, metavar="PROMPTS",
                    help="record deliberate mispronunciations from a prompt TSV")
    ap.add_argument("--say-only", action="store_true",
                    help="play only the line to say, never the intended one. Automatic for "
                         "prompts whose mistake *is* a different line")
    ap.add_argument("--trim", action="store_true",
                    help="cap silence at both ends of every clip (originals kept in "
                         "eval_clips/raw)")
    ap.add_argument("--trim-pad", type=float, nargs=2, metavar=("LEAD", "TRAIL"),
                    default=(LEAD_PAD, TRAIL_PAD),
                    help="seconds of silence to leave at each end")
    ap.add_argument("--redo", metavar="WHICH", default=None,
                    help="discard recordings before recording: 'bad' (fails the "
                         "level checks), 'all', or '2,7,19'")
    ap.add_argument("--clips-dir", type=Path, default=None,
                    help="score a different eval set, e.g. data/fleurs/eval")
    ap.add_argument("--clips", choices=["all", "synth", "voice"], default="all",
                    help="which clips to score: synthetic, recorded, or both")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--worst", type=int, default=5, help="show N worst clips per model")
    args = ap.parse_args()
    # Whether --models was passed explicitly, as opposed to defaulted.
    args.models_given = any(a.startswith("--models") for a in sys.argv[1:])

    if args.list_inputs:
        list_inputs()
        return 0
    if args.guide:
        print_guide(args.guide)
        return 0
    if args.clips_dir:
        use_clips_dir(args.clips_dir)
    if args.synth:
        asyncio.run(synth(args.synth, args.voice))
    if args.record_errors:
        record_errors(args.record_errors, args.input, args.say_only)
        return

    if args.trim:
        trim(*args.trim_pad)
        return

    if args.redo:
        redo(args.redo, args.record or 25)
        if not args.record:
            # Clearing takes and then scoring whatever is left — against the default
            # model list, which downloads Whisper — is nobody intent.
            print("Nothing to record. Add --record 25 to record them now.")
            return 0
    if args.record:
        record(args.record, args.input)
        # Recording used to fall straight into scoring against the *default* model list,
        # which is two Whisper builds — so finishing a take session downloaded 3GB and
        # graded the new clips mixed in with the synthetic ones, against models the app
        # does not use. Say what to run instead.
        if not args.models_given:
            return 0

    rows = _read_manifest(args.clips)
    if not rows:
        print(f"No {args.clips} clips. Run with --synth 25 or --record 25 first.",
              file=sys.stderr)
        return 2

    kinds = {"synth" if r["file"].startswith("synth_") else "voice" for r in rows}
    if kinds == {"synth"}:
        note = "  (synthetic — ranking is the result, not the absolute numbers)"
    elif kinds == {"voice"}:
        note = "  (your voice — the numbers that actually matter)"
    else:
        # Averaging a synthetic voice with a real one reports a number for neither, and
        # the synthetic half is the optimistic one, so the mix flatters whatever is being
        # tested. Say so rather than printing it as a single figure.
        note = ("  (MIXED synthetic and real — pass --clips voice or --clips synth to "
                "separate them)")
    print(f"\nComparing on {len(rows)} clips{note}")

    reports = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        try:
            reports.append(run_model(name, rows, args.device, args.beam))
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {name} failed: {exc}", file=sys.stderr)

    if not reports:
        return 1

    print("\n" + "═" * 92)
    print(f"{'model':<52}{'WER':>7}{'fWER':>7}{'CER':>7}{'score':>8}{'pass':>7}{'s/clip':>8}")
    print("─" * 92)
    best = min(reports, key=lambda r: r["fwer"])
    for r in sorted(reports, key=lambda r: r["fwer"]):
        mark = "★" if r is best else " "
        name = r["model"] if len(r["model"]) <= 50 else "…" + r["model"][-49:]
        print(f"{mark}{name:<51}{r['wer']:>6.1%}{r['fwer']:>7.1%}{r['cer']:>7.1%}"
              f"{r['score']:>8.2f}{r['pass_rate']:>7.0%}{r['sec_per_clip']:>8.1f}")
    print("═" * 92)
    print("  fWER ignores diacritics and the silent għ · pass = share the app would "
          "mark correct")

    if len(reports) > 1:
        # Lower fWER is better, so a negative delta means `a` won. This read the
        # other way round and printed the loser as the winner — the table above it
        # sorts independently, so the summary line contradicted its own table.
        a, b = reports[0], reports[-1]
        delta = a["fwer"] - b["fwer"]
        better, worse = (a, b) if delta < 0 else (b, a)
        print(f"\n  {better['model'].split('/')[-1]} beats "
              f"{worse['model'].split('/')[-1]} by "
              f"{abs(delta):.1%} fWER and {abs(a['pass_rate']-b['pass_rate']):.0%} pass rate.")

    for r in reports:
        bad = sorted(r["results"], key=lambda x: x["score"])[:args.worst]
        if not bad or bad[0]["score"] > 0.9:
            continue
        print(f"\n  worst for {r['model'].split('/')[-1]}:")
        for x in bad:
            print(f"    {x['score']:.2f}  want: {x['ref']}")
            print(f"          got : {x['hyp']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

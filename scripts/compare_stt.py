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
    en, say = _guide()
    rows = _read_manifest()
    existing = {r["file"] for r in rows}
    quiet = 0
    for i, sentence in enumerate(_sentences(n), 1):
        name = f"me_{i:03d}.wav"
        if name in existing:
            continue
        print(f"\n[{i}/{n}]  {sentence}")
        print(f"          say:     {say.get(sentence, '(no guide yet)')}")
        print(f"          meaning: {en.get(sentence, '?')}")
        if not _play(sentence):
            print("          (no reference audio — run scripts/prebuild_audio.py)")
        input("       Enter to hear it again, or just speak after the next Enter… ")
        _play(sentence)
        input("       Enter to start recording, speak, then Enter again to stop… ")
        proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "avfoundation", "-i", device,
             "-ar", "16000", "-ac", "1", str(CLIPS / name)],
            stdin=subprocess.PIPE,
        )
        input()
        proc.communicate(b"q")

        level = _peak(CLIPS / name)
        if level < 0.01:
            quiet += 1
            print(f"       ! silent (peak {level:.4f}) — wrong input device? "
                  f"try --list-inputs")
        else:
            print(f"       ok (peak {level:.2f})")
        rows.append({"file": name, "text": sentence})
        _write_manifest(rows)
    if quiet:
        print(f"\n! {quiet} clips came back silent. Delete data/eval_clips/me_*.wav "
              f"and the me_ rows in manifest.tsv, then retry with --input.")
    print(f"\n✓ {len(rows)} clips in {CLIPS}")


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
    ap.add_argument("--clips-dir", type=Path, default=None,
                    help="score a different eval set, e.g. data/fleurs/eval")
    ap.add_argument("--clips", choices=["all", "synth", "voice"], default="all",
                    help="which clips to score: synthetic, recorded, or both")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--worst", type=int, default=5, help="show N worst clips per model")
    args = ap.parse_args()

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
    if args.record:
        record(args.record, args.input)

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

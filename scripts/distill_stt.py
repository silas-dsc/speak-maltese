#!/usr/bin/env python3
"""Distil the 315M Maltese recogniser into something a phone can hold.

Everything else has been tried and measured. The 201MB model is the only thing that
passes — 0.98 app score, and 92% of right answers kept at the threshold that rejects
95% of near-misses — and it does not fit in a WebKit page. The 18.9M QuartzNet fits and
fails: 80% free-decode pass rate, 8% under the near-miss threshold, because its frame
posteriors are too blurry for a likelihood ratio to mean anything. Quantising below
76MB is not available, and matching stored audio with DTW tops out at 44% for reasons
that are structural rather than about size.

What has not been tried is training a small model *for this app* rather than picking a
small general one off the Hub, and two things here make that unusually promising:

  * **A teacher.** The 315M model produces frame-level posteriors over 39 characters
    for any audio, so the student learns a soft distribution per frame rather than a
    hard label per utterance. That is far more signal than the 64h corpus carried.
  * **Unlimited matched data.** The app speaks with `edge-tts`, so the exact
    distribution it must recognise can be synthesised — every line it will ever ask
    for, in both voices, at several rates. `prebuild_audio.py` already renders it.

The student is a QuartzNet-shaped depthwise-separable conv stack, deliberately: no
attention, so it runs on WASM and needs no WebGPU, which was the other half of what
made 200MB unusable on an iPhone.

    python scripts/distill_stt.py teacher     # mel + teacher posteriors → memmaps
    python scripts/distill_stt.py train       # distil
    python scripts/distill_stt.py export      # ONNX, in the layout the eval harness reads

Deliberately excluded from training: the 25 sentences in `data/eval_clips/manifest.tsv`.
They are deck lines, so they would otherwise be trained on and every number reported by
`compare_stt.py` and `constrained_ctc.py` would be measured on the training set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import text as mtext  # noqa: E402
from backend.config import AUDIO_CACHE, CFG, DATA_DIR  # noqa: E402
from compare_stt import _NEMO, _mel_filters, _nemo_features  # noqa: E402

WORK = DATA_DIR / "distill"
CLIPS = DATA_DIR / "eval_clips"
TEACHER = "carlosdanielhernandezmena/wav2vec2-large-xlsr-53-maltese-64h"

# (voice, rate) renders to learn from. The app's own voice and rate first; the second
# voice is a different gender, which is the only speaker variation available without
# recording people.
VARIANTS = [
    (CFG.azure_voice, 0.95),
    (CFG.azure_voice_alt, 0.95),
    (CFG.azure_voice, 1.10),
    (CFG.azure_voice_alt, 0.85),
]

N_MELS = 64


# ── Corpus ─────────────────────────────────────────────────────────────────

def held_out() -> set[str]:
    """The evaluation sentences, normalised. Never trained on."""
    import csv

    manifest = CLIPS / "manifest.tsv"
    if not manifest.exists():
        return set()
    with manifest.open(encoding="utf-8") as fh:
        return {mtext.normalise(r["text"]).lower().strip()
                for r in csv.DictReader(fh, delimiter="\t") if r.get("text")}


def corpus() -> list[str]:
    from prebuild_audio import lines_for

    keep, seen = [], set()
    skip = held_out()
    for line in lines_for("all"):
        flat = mtext.normalise(line).lower().strip()
        if not flat or flat in seen or flat in skip:
            continue
        seen.add(flat)
        keep.append(line)
    return keep


def clip_path(line: str, voice: str, rate: float) -> Path:
    from backend import tts

    return AUDIO_CACHE / f"{tts._cache_key(line, voice, rate, 'edge')}.mp3"


# ── Augmentation ───────────────────────────────────────────────────────────
# The first student was trained on clean TTS and nothing else, and it showed: it works
# on the app's own voices and falls apart on a person. Three mismatches stack up between
# the training audio and what actually reaches the model, and each is fixable here.
#
#   * **Speakers.** Two synthetic voices, both `mt-MT`. Resampling shifts pitch and
#     formants together, which is crude vocal-tract-length perturbation — it turns two
#     apparent speakers into as many as we ask for.
#   * **The codec.** `MediaRecorder` hands us Opus in a WebM container; training saw
#     edge-tts MP3. Different artefacts entirely, and the only honest fix is to put the
#     training audio through the same codec.
#   * **The room and the microphone.** No noise, no reverb, no clipping, always the same
#     level.
#
# Augmenting in the time domain is why the teacher has to be re-run per variant: its
# posteriors describe the audio it was given, frame for frame, and anything that shifts
# the waveform in time leaves them pointing at the wrong frames. That is exactly the
# mistake the first round avoided by only masking features — and exactly why it could
# not fix the domain gap.

AUGMENTS = ("identity", "opus", "vtlp_up", "vtlp_down", "noisy_room")

# FLEURS reads Wikipedia prose: 15-30 seconds a clip, where the app asks for two-second
# phrases. Training on whole clips would teach the model a length distribution it will
# never meet, and cost a fortune in teacher time per example. Real speech carries no
# transcript through this pipeline — it is supervised on the teacher's posteriors — so it
# can be cut anywhere without breaking anything, and short chunks are both cheaper and a
# closer match to what a learner actually says.
CHUNK_S = 3.0
CHUNK_OVERLAP_S = 0.25
MIN_CHUNK_S = 0.8


def chunk(wave, sample_rate: int = 16000):
    """Split a long recording into utterance-length pieces, with a little overlap so a
    word cut in half still appears whole somewhere."""
    span = int(CHUNK_S * sample_rate)
    step = span - int(CHUNK_OVERLAP_S * sample_rate)
    if len(wave) <= span:
        return [wave]
    out = []
    for start in range(0, len(wave), step):
        piece = wave[start:start + span]
        if len(piece) >= int(MIN_CHUNK_S * sample_rate):
            out.append(piece)
    return out or [wave]


def _resample(wave, factor: float):
    """Linear resample by `factor`. Playing audio at a different rate moves pitch and
    formants together, which is the cheap approximation to a different vocal tract."""
    import numpy as np

    n = int(len(wave) / factor)
    if n < 400:
        return wave
    idx = np.arange(n, dtype=np.float32) * factor
    lo = np.floor(idx).astype(np.int32).clip(0, len(wave) - 2)
    frac = (idx - lo).astype(np.float32)
    return ((1 - frac) * wave[lo] + frac * wave[lo + 1]).astype(np.float32)


def _opus_roundtrip(wave):
    """Through the codec the browser actually records with. Nothing else simulates what
    Opus at a low bitrate does to fricatives, which is where Maltese `ħ`, `x` and `għ`
    live."""
    import subprocess
    import tempfile

    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "in.f32"
        enc = Path(tmp) / "mid.opus"
        out = Path(tmp) / "out.f32"
        raw.write_bytes(wave.astype("<f4").tobytes())
        base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        fmt = ["-f", "f32le", "-ar", "16000", "-ac", "1"]
        try:
            subprocess.run(base + fmt + ["-i", str(raw), "-c:a", "libopus",
                                         "-b:a", "24k", str(enc)], check=True)
            subprocess.run(base + ["-i", str(enc)] + fmt + [str(out)], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return wave                     # no libopus: skip rather than fail the run
        got = np.frombuffer(out.read_bytes(), dtype="<f4").astype(np.float32)
        return got if got.size >= 400 else wave


def augment(wave, kind: str, rng):
    """One variant of one clip. Deliberately mild: the point is a plausible microphone
    in a plausible room, not a stress test the teacher itself cannot label."""
    import numpy as np

    if kind == "identity":
        return wave
    if kind == "opus":
        return _opus_roundtrip(wave)
    if kind in ("vtlp_up", "vtlp_down"):
        factor = rng.uniform(1.04, 1.12) if kind == "vtlp_up" else rng.uniform(0.89, 0.96)
        return _resample(wave, factor)
    if kind == "noisy_room":
        out = wave.copy()
        # A short early-reflection tail. Not a measured impulse response, but enough
        # that the model stops assuming an anechoic studio.
        delay = int(rng.uniform(0.01, 0.05) * _NEMO["sample_rate"])
        if delay < len(out):
            out[delay:] += 0.25 * rng.uniform(0.5, 1.0) * wave[:-delay]
        out += rng.normal(0, rng.uniform(0.001, 0.01), len(out)).astype(np.float32)
        out *= rng.uniform(0.5, 1.4)                        # level varies per speaker
        return np.clip(out, -1.0, 1.0).astype(np.float32)
    raise ValueError(kind)


# ── Real speech ────────────────────────────────────────────────────────────

def fleurs_unpack(parquet: Path, out_dir: Path, limit: int | None = None,
                  skip: set[str] | None = None) -> list[dict]:
    """Unpack FLEURS Maltese into 16kHz mono FLAC, with its transcripts.

    Distillation itself needs no transcripts — the teacher labels whatever it is given,
    so any Maltese audio counts, and what this buys is the one thing synthesis cannot:
    real speakers, real rooms, real microphones. The transcripts are kept anyway, because
    they make a *real-speech* evaluation possible, and every number this project has
    reported so far was measured on the app's own TTS voices."""
    import io
    import subprocess

    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    skip = skip or set()
    rows: list[dict] = []
    if not parquet.exists():
        return rows
    try:
        table = pq.read_table(parquet, columns=["audio", "id", "transcription", "gender"])
    except Exception as exc:  # noqa: BLE001
        # Most often a download still in flight. Carry on with whatever else is readable
        # rather than losing the run to it.
        print(f"  ! {parquet.name} unreadable ({exc.__class__.__name__}) — skipped",
              file=sys.stderr)
        return rows

    print(f"  {parquet.name}: {table.num_rows} rows")
    for i in range(table.num_rows):
        if limit is not None and len(rows) >= limit:
            break
        uid = f"{parquet.stem}_{i:05d}"
        if uid in skip:
            continue
        cell = table.column("audio")[i].as_py()
        raw = cell.get("bytes") if isinstance(cell, dict) else None
        if not raw:
            continue
        dest = out_dir / f"{uid}.flac"
        if not dest.exists():
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", "pipe:0",
                 "-ar", "16000", "-ac", "1", str(dest)],
                input=raw, capture_output=True)
            if proc.returncode != 0 or not dest.exists():
                continue
        rows.append({"file": dest.name, "uid": uid,
                     "text": table.column("transcription")[i].as_py() or "",
                     "gender": table.column("gender")[i].as_py()})
        if len(rows) % 250 == 0:
            print(f"    {len(rows)} clips", flush=True)
    return rows


def write_manifest(rows: list[dict], path: Path) -> None:
    """The two-column manifest `compare_stt.py` already reads, so a real-speech set is
    scored by the same harness and the same metrics as the synthetic one."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, delimiter="\t", fieldnames=["file", "text"])
        w.writeheader()
        for r in rows:
            if r["text"].strip():
                w.writerow({"file": r["file"], "text": r["text"]})


FLEURS = DATA_DIR / "fleurs"
REAL_EVAL_N = 150


def fleurs_split(eval_n: int = REAL_EVAL_N, train_limit: int | None = None):
    """Real speech, split so the evaluation half is never trained on.

    The first `eval_n` validation clips become the held-out real-speech set; everything
    else — the rest of validation, plus train — is training material."""
    eval_rows = fleurs_unpack(FLEURS / "validation.parquet", FLEURS / "eval", eval_n)
    held = {r["uid"] for r in eval_rows}
    write_manifest(eval_rows, FLEURS / "eval" / "manifest.tsv")

    train_rows = fleurs_unpack(FLEURS / "validation.parquet", FLEURS / "clips",
                               None, held)
    train_rows += fleurs_unpack(FLEURS / "train.parquet", FLEURS / "clips",
                                train_limit, held)
    write_manifest(train_rows, FLEURS / "clips" / "manifest.tsv")
    print(f"real speech: {len(train_rows)} for training · {len(eval_rows)} held out")
    return train_rows, eval_rows


# ── Stage 1: features and teacher posteriors ───────────────────────────────

def stage_teacher(limit: int | None, augments: list[str], real_limit: int | None,
                  shard: str, sources: list[str]) -> int:
    """Precompute once: the student's input and the teacher's answer, side by side.

    Both go into flat memmaps with an index, so training never touches audio or the
    315M model again — which turns an epoch from twenty minutes into seconds."""
    import torch
    from faster_whisper.audio import decode_audio
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    WORK.mkdir(parents=True, exist_ok=True)
    lines = corpus()[:limit]
    # Every clip is (audio to load, the text the teacher will be checked against). Real
    # speech has no text here on purpose: distillation supervises on the teacher's
    # posteriors, so a FLEURS clip needs no transcript to be useful — and its Wikipedia
    # sentences are nothing like the deck anyway. CTC is applied only where a target is
    # known.
    jobs = []
    if "tts" in sources:
        jobs += [(clip_path(ln, v, r), mtext.normalise(ln).lower().strip(), f"{v}@{r}")
                 for ln in lines for (v, r) in VARIANTS if clip_path(ln, v, r).exists()]
        print(f"{len(lines)} lines · {len(jobs)} rendered TTS clips")

    if "real" in sources:
        train_rows, _held = fleurs_split(train_limit=real_limit)
        jobs += [(FLEURS / "clips" / r["file"], "", "fleurs") for r in train_rows]
        print(f"{len(train_rows)} real-speech clips")

    if "accent" in sources:
        # Deck lines read by voices that mispronounce Maltese. The label is the deck line
        # regardless of what came out of the speaker — that is the entire point, and it is
        # why these carry a target where FLEURS does not.
        from render_accents import VOICES, clip_path as accent_path, corpus as accent_corpus

        n = 0
        for voice in VOICES:
            for line in accent_corpus(None):
                path = accent_path(line, voice)
                if path.exists():
                    jobs.append((path, mtext.normalise(line).lower().strip(),
                                 f"accent:{voice}"))
                    n += 1
        print(f"{n} accented clips")
    print(f"{len(jobs)} clips × {len(augments)} variants = "
          f"{len(jobs) * len(augments)} passes → shard {shard!r}")
    if not jobs:
        print("Nothing rendered yet — run scripts/prebuild_audio.py", file=sys.stderr)
        return 2

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(TEACHER)
    model = Wav2Vec2ForCTC.from_pretrained(TEACHER).to(device).eval()
    vocab = processor.tokenizer.get_vocab()
    (WORK / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    v_size = model.config.vocab_size

    fb = _mel_filters(_NEMO["n_fft"] // 2 + 1, N_MELS, _NEMO["sample_rate"])
    pad = (_NEMO["n_fft"] - _NEMO["win_length"]) // 2
    window = np.pad(np.hanning(_NEMO["win_length"]).astype(np.float32), (pad, pad))

    mels: list[np.ndarray] = []
    posts: list[np.ndarray] = []
    index: list[dict] = []
    rng = np.random.default_rng(3)
    t0 = time.time()
    passes = 0
    for i, (path, flat, source) in enumerate(jobs, 1):
        try:
            base_wave = np.asarray(decode_audio(str(path), sampling_rate=16000),
                                   dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 — a truncated render is not fatal
            print(f"  ! {path.name}: {exc}", file=sys.stderr)
            continue
        if base_wave.size < 1600:                 # under 0.1s: a failed render
            continue

        # Only real speech is chunked; a TTS clip is already one phrase.
        pieces = chunk(base_wave) if not flat else [base_wave]
        for piece, kind in ((pc, k) for pc in pieces for k in augments):
            wave = augment(piece, kind, rng)
            passes += 1
            mel = _nemo_features(wave, N_MELS, fb, window)[0]          # (64, T)
            inputs = processor(wave, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                logits = model(inputs.input_values.to(device)).logits
            post = torch.log_softmax(logits, dim=-1)[0].float().cpu().numpy()   # (T', V)

            # The student subsamples its 100fps mel by 2 and the teacher's conv stack
            # subsamples 16kHz by 320, so both land on 50fps and differ by at most a
            # frame. Recomputed per variant, which is the whole reason time-domain
            # augmentation is possible at all.
            keep = min(mel.shape[1] // 2, post.shape[0])
            if keep < 4:
                continue
            mels.append(mel[:, :keep * 2].T.astype(np.float16))
            posts.append(post[:keep].astype(np.float16))
            index.append({"text": flat, "source": source, "augment": kind,
                          "frames": keep})
        if i % 100 == 0 or i == len(jobs):
            print(f"  {i:>5}/{len(jobs)} clips · {passes} passes · kept {len(index)} · "
                  f"{(time.time() - t0) / max(1, passes):.2f}s/pass", flush=True)

    if not index:
        print("nothing kept", file=sys.stderr)
        return 1
    mel_all = np.concatenate(mels)
    post_all = np.concatenate(posts)
    # One shard per source, so real speech can be added to a run without recomputing the
    # teacher over everything that was already done. `train` reads whatever shards exist.
    np.save(WORK / f"mel_{shard}.npy", mel_all)
    np.save(WORK / f"post_{shard}.npy", post_all)
    (WORK / f"index_{shard}.json").write_text(json.dumps(
        {"vocab_size": v_size, "n_mels": N_MELS, "items": index}), encoding="utf-8")
    print(f"\n{len(index)} passes · mel {mel_all.nbytes / 1e6:.0f}MB "
          f"· posteriors {post_all.nbytes / 1e6:.0f}MB → {WORK}/*_{shard}.npy")
    return 0


# ── Pseudo-labels for untranscribed speech ─────────────────────────────────
# The first attempt at adding real speech collapsed: 60% of the data had no transcript,
# so it trained on frame-level KD alone, and blank frames outnumber character frames by
# an order of magnitude in any CTC posterior. Matching the teacher per frame is then
# satisfied most cheaply by predicting blank everywhere — which is exactly what happened.
# The student went silent on real audio (100% blank over 535 frames) while still reading
# the synthetic clips it had a CTC target for.
#
# The fix does not need the teacher again. Its posteriors are already on disk, so decoding
# them gives the sequence it would have transcribed, and that becomes a target for the
# same CTC term the TTS clips use. Per-frame agreement stops being the only pressure, and
# the student has to actually emit the characters.

def stage_pseudo(shard: str) -> int:
    """Greedy-decode a shard's teacher posteriors into text targets, in place."""
    meta_path = WORK / f"index_{shard}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    vocab = json.loads((WORK / "vocab.json").read_text(encoding="utf-8"))
    inv = {i: t for t, i in vocab.items()}
    blank = vocab.get("<pad>", vocab.get("[PAD]"))
    space = "|" if "|" in vocab else " "

    post = np.load(WORK / f"post_{shard}.npy", mmap_mode="r")
    off, filled, empty = 0, 0, 0
    for it in meta["items"]:
        n = it["frames"]
        ids = np.asarray(post[off:off + n], dtype=np.float32).argmax(-1)
        off += n
        # Merge repeats, then drop blanks — the order that keeps geminates, same as
        # everywhere else in this project.
        out, prev = [], -1
        for i in ids:
            if i != prev:
                if i != blank:
                    out.append(inv[int(i)])
                prev = i
        text = "".join(out).replace(space, " ")
        text = " ".join(text.split())
        if text:
            it["text"] = text
            filled += 1
        else:
            # The teacher heard nothing here — usually a chunk that landed on silence.
            # Leaving it untargeted would reintroduce exactly the imbalance above, so it
            # is marked for the trainer to drop.
            it["text"] = ""
            empty += 1
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    print(f"shard {shard}: {filled} pseudo-labelled, {empty} silent")
    if filled:
        print("  examples:")
        for it in meta["items"][:3]:
            print(f"    {it['text'][:70]!r}")
    return 0


# ── The student ────────────────────────────────────────────────────────────

def build_student(vocab_size: int, width: int, blocks: int, kernel: int):
    """QuartzNet-shaped: a strided stem, then depthwise-separable residual blocks.

    Convolution only — no attention anywhere. That is not a stylistic choice: WASM has
    no fast attention kernel and WebGPU is what an iPhone could not afford, so a stack
    of convolutions is the only shape that runs everywhere this has to run."""
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, c: int, k: int):
            super().__init__()
            self.dw = nn.Conv1d(c, c, k, padding=k // 2, groups=c, bias=False)
            self.pw = nn.Conv1d(c, c, 1, bias=False)
            self.bn = nn.BatchNorm1d(c)
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(self.bn(self.pw(self.dw(x))) + x)

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                # stride 2: 100fps mel → 50fps, which is the teacher's frame rate
                nn.Conv1d(N_MELS, width, 11, stride=2, padding=5, bias=False),
                nn.BatchNorm1d(width), nn.ReLU(),
            )
            self.blocks = nn.Sequential(*[Block(width, kernel) for _ in range(blocks)])
            self.head = nn.Conv1d(width, vocab_size, 1)

        def forward(self, mel):                       # (B, 64, T) → (B, T/2, V)
            x = self.blocks(self.stem(mel))
            return torch.log_softmax(self.head(x).transpose(1, 2), dim=-1)

    return Student()


def param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ── Stage 2: distil ────────────────────────────────────────────────────────

def stage_train(width: int, blocks: int, kernel: int, epochs: int, batch: int,
                lr: float, kd_weight: float, tag: str) -> int:
    import torch
    from torch import nn

    shards = sorted(WORK.glob("index_*.json"))
    if not shards:
        print("no shards — run `distill_stt.py teacher` first", file=sys.stderr)
        return 2
    vocab = json.loads((WORK / "vocab.json").read_text(encoding="utf-8"))
    blank = vocab.get("<pad>", vocab.get("[PAD]"))
    space = "|" if "|" in vocab else " "

    # Each shard is its own pair of memmaps, so offsets are per shard rather than global.
    items, mels, posts, offs = [], [], [], []
    v_size = None
    for meta_path in shards:
        name = meta_path.stem.removeprefix("index_")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        v_size = meta["vocab_size"]
        which = len(mels)
        mels.append(np.load(WORK / f"mel_{name}.npy", mmap_mode="r"))
        posts.append(np.load(WORK / f"post_{name}.npy", mmap_mode="r"))
        m_off = p_off = 0
        for it in meta["items"]:
            items.append(it)
            offs.append((which, m_off, p_off, it["frames"]))
            m_off += it["frames"] * 2
            p_off += it["frames"]
        print(f"  shard {name}: {len(meta['items'])} passes")

    def encode(s: str) -> list[int]:
        return [vocab[space if ch == " " else ch] for ch in s
                if (space if ch == " " else ch) in vocab]

    # Real speech carries no transcript here, so it trains on the teacher's posteriors
    # alone. That is not a compromise — the posteriors are the richer signal, and the
    # CTC term only exists to keep the output sequence honest where a target is known.
    targets = [encode(it.get("text") or "") for it in items]

    # How much to trust the teacher, per item. On accented speech it is guessing — it
    # transcribes a learner at 74% fWER — so its posteriors there describe its own
    # confusion, not the sound. The text label, by contrast, is exactly right: we know
    # which line was asked for. So accented items lean on CTC and barely on KD, the
    # opposite balance from FLEURS, where there is no label and the posteriors are all
    # there is. Getting this backwards is what silenced the first attempt at real speech.
    kd_scale = np.array([0.2 if (it.get("source") or "").startswith("accent") else 1.0
                         for it in items], dtype=np.float32)

    # Split by *line*, so a dev sentence is never seen in another voice either.
    rng = np.random.default_rng(11)
    # Split by line so a dev sentence is unseen in every voice and every augmentation.
    # Real speech has no line, so it is keyed by its own clip instead — the same
    # guarantee, applied to the only identity it has.
    def key(i: int, it: dict) -> str:
        return it.get("text") or f"clip:{offs[i][0]}:{offs[i][1]}"

    # A pass the teacher transcribed as nothing teaches only "stay silent", which is the
    # failure this pipeline already walked into once. Dropped rather than down-weighted.
    silent = [i for i, it in enumerate(items) if not (it.get("text") or "").strip()]
    if silent:
        print(f"  dropping {len(silent)} passes the teacher heard nothing in")
    usable = set(range(len(items))) - set(silent)

    keys = sorted({key(i, it) for i, it in enumerate(items) if i in usable})
    rng.shuffle(keys)
    dev_keys = set(keys[:max(1, len(keys) // 10)])
    train_ix = [i for i, it in enumerate(items)
                if i in usable and key(i, it) not in dev_keys]
    dev_ix = [i for i, it in enumerate(items)
              if i in usable and key(i, it) in dev_keys]
    print(f"{len(train_ix)} train passes · {len(dev_ix)} dev passes "
          f"({len(keys) - len(dev_keys)}/{len(dev_keys)} distinct utterances)")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_student(v_size, width, blocks, kernel).to(device)
    n_par = param_count(model)
    print(f"student: width={width} blocks={blocks} k={kernel} · "
          f"{n_par / 1e6:.2f}M params · {n_par * 4 / 1e6:.1f}MB fp32")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps = max(1, len(train_ix) // batch) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.15)
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True)

    def batch_of(ix: list[int], augment: bool):
        frames = [offs[i][3] for i in ix]
        tmax = max(frames)
        x = np.zeros((len(ix), N_MELS, tmax * 2), dtype=np.float32)
        y = np.zeros((len(ix), tmax, v_size), dtype=np.float32)
        for row, i in enumerate(ix):
            which, m_off, p_off, n = offs[i]
            x[row, :, :n * 2] = mels[which][m_off:m_off + n * 2].T.astype(np.float32)
            y[row, :n] = posts[which][p_off:p_off + n].astype(np.float32)
        xt = torch.from_numpy(x)
        if augment:
            # SpecAugment, in the feature domain on purpose: the teacher's posteriors
            # were computed on the clean audio, and anything that shifted the audio in
            # time would leave them pointing at the wrong frames.
            for row in range(xt.shape[0]):
                for _ in range(2):
                    f = np.random.randint(0, 9)
                    f0 = np.random.randint(0, max(1, N_MELS - f))
                    xt[row, f0:f0 + f] = 0
                    t = np.random.randint(0, max(1, int(0.10 * frames[row] * 2)))
                    t0 = np.random.randint(0, max(1, frames[row] * 2 - t))
                    xt[row, :, t0:t0 + t] = 0
        flat = torch.cat([torch.tensor(targets[i], dtype=torch.long) for i in ix])
        return (xt.to(device), torch.from_numpy(y).to(device),
                torch.tensor(frames, dtype=torch.long),
                flat.to(device),
                torch.tensor([len(targets[i]) for i in ix], dtype=torch.long),
                torch.from_numpy(kd_scale[list(ix)]).to(device))

    def run_epoch(ix: list[int], train: bool):
        model.train(train)
        order = list(ix)
        if train:
            rng.shuffle(order)
        # Length-bucketed so padding does not dominate: sort within large chunks.
        chunks = [order[i:i + batch * 16] for i in range(0, len(order), batch * 16)]
        order = [i for ch in chunks for i in sorted(ch, key=lambda k: offs[k][3])]
        tot_kd = tot_ctc = n = 0
        for s in range(0, len(order) - batch + 1, batch):
            bx, by, blen, flat, tlen, kscale = batch_of(order[s:s + batch], augment=train)
            with torch.set_grad_enabled(train):
                out = model(bx)                                # (B, T, V)
                keep = min(out.shape[1], by.shape[1])
                out, by_ = out[:, :keep], by[:, :keep]
                mask = (torch.arange(keep)[None, :] < blen[:, None]).to(device)
                # KL(teacher ‖ student) over valid frames. The teacher's full
                # distribution is the point — a hard label per frame would throw away
                # exactly the confidence that near-miss rejection depends on.
                kd = (by_.exp() * (by_ - out)).sum(-1)
                kd = (kd * mask * kscale[:, None]).sum() / mask.sum()
                # `aten::_ctc_loss` has no MPS kernel, so this one term crosses to the
                # CPU and back. `.cpu()` is differentiable, so the gradient still
                # reaches the GPU weights; the copy is ~600KB a step and does not show
                # up against the convolutions.
                # Clips with no target contribute nothing to CTC; dropping them keeps
                # the loss from being divided by utterances it cannot describe.
                if int(tlen.sum()) > 0:
                    ctc_loss = ctc(out.cpu().transpose(0, 1), flat.cpu(),
                                   torch.clamp(blen, max=keep), tlen)
                else:
                    ctc_loss = torch.zeros((), dtype=out.dtype)
                loss = kd_weight * kd + (1 - kd_weight) * ctc_loss
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    sched.step()
            tot_kd += float(kd.detach())
            tot_ctc += float(ctc_loss.detach())
            n += 1
        return tot_kd / max(1, n), tot_ctc / max(1, n)

    out_dir = WORK / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        kd, c = run_epoch(train_ix, True)
        dkd, dc = run_epoch(dev_ix, False)
        flag = ""
        if dkd + dc < best:
            best = dkd + dc
            torch.save({"state": model.state_dict(), "width": width, "blocks": blocks,
                        "kernel": kernel, "vocab_size": v_size}, out_dir / "student.pt")
            flag = "  ←"
        print(f"  ep {ep:>3}/{epochs}  kd {kd:.4f} ctc {c:.3f} │ "
              f"dev kd {dkd:.4f} ctc {dc:.3f}  {time.time() - t0:.0f}s{flag}", flush=True)
    print(f"\nbest dev {best:.4f} → {out_dir / 'student.pt'}")
    return 0


# ── Stage 3: export ────────────────────────────────────────────────────────

def stage_export(tag: str) -> int:
    """Write the student in the layout `compare_stt.py` and `constrained_ctc.py`
    already read — the same `model.onnx` / `vocab.txt` / `config.json` triple the
    QuartzNet export uses — so both harnesses score it with no new code."""
    import torch

    ckpt = torch.load(WORK / tag / "student.pt", map_location="cpu", weights_only=False)
    model = build_student(ckpt["vocab_size"], ckpt["width"], ckpt["blocks"],
                          ckpt["kernel"])
    model.load_state_dict(ckpt["state"])
    model.eval()

    out = WORK / tag / "onnx"
    out.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, N_MELS, 200)
    torch.onnx.export(
        model, (dummy,), str(out / "model.onnx"),
        input_names=["audio_signal"], output_names=["logprobs"],
        dynamic_axes={"audio_signal": {0: "batch", 2: "time"},
                      "logprobs": {0: "batch", 1: "frames"}},
        opset_version=17,
        # One file. The default puts the weights in a sibling `.onnx.data`, which is
        # fine on disk and useless as something to fetch from a CDN.
        external_data=False,
    )

    vocab = json.loads((WORK / "vocab.json").read_text(encoding="utf-8"))
    # The harness expects QuartzNet's conventions: `▁` for a space and `<blk>` for the
    # CTC blank. Same ids, renamed.
    lines = []
    for tok, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
        name = "▁" if tok == "|" else ("<blk>" if tok in ("<pad>", "[PAD]") else tok)
        lines.append(f"{name} {idx}")
    (out / "vocab.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "config.json").write_text(json.dumps(
        {"model_type": "nemo-conformer-ctc", "features_size": N_MELS,
         "subsampling_factor": 2}, indent=2), encoding="utf-8")

    size = (out / "model.onnx").stat().st_size
    print(f"{param_count(model) / 1e6:.2f}M params · {size / 1e6:.1f} MB → {out}")
    print(f"\nevaluate with:\n  .venv/bin/python scripts/compare_stt.py --models "
          f"{out}\n  .venv/bin/python scripts/constrained_ctc.py --models {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["teacher", "pseudo", "train", "export"])
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of deck lines (for a quick pass)")
    ap.add_argument("--real-limit", type=int, default=None,
                    help="cap the number of real-speech clips")
    ap.add_argument("--augments", default=",".join(AUGMENTS),
                    help=f"comma-separated, from {','.join(AUGMENTS)}")
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--blocks", type=int, default=15)
    ap.add_argument("--kernel", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--kd-weight", type=float, default=0.9)
    ap.add_argument("--tag", default="student")
    ap.add_argument("--shard", default="tts", help="name for this teacher shard")
    ap.add_argument("--sources", default="tts,real",
                    help="which audio to run the teacher over: tts, real, accent")
    args = ap.parse_args()

    if args.stage == "teacher":
        kinds = [a.strip() for a in args.augments.split(",") if a.strip()]
        bad = [k for k in kinds if k not in AUGMENTS]
        if bad:
            print(f"unknown augmentations: {bad}", file=sys.stderr)
            return 2
        srcs = [x.strip() for x in args.sources.split(',') if x.strip()]
        return stage_teacher(args.limit, kinds, args.real_limit,
                             args.shard, srcs)
    if args.stage == "pseudo":
        return stage_pseudo(args.shard)
    if args.stage == "train":
        return stage_train(args.width, args.blocks, args.kernel, args.epochs,
                           args.batch, args.lr, args.kd_weight, args.tag)
    return stage_export(args.tag)


if __name__ == "__main__":
    raise SystemExit(main())

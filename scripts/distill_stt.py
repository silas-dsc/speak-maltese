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


# ── Stage 1: features and teacher posteriors ───────────────────────────────

def stage_teacher(limit: int | None) -> int:
    """Precompute once: the student's input and the teacher's answer, side by side.

    Both go into flat memmaps with an index, so training never touches audio or the
    315M model again — which turns an epoch from twenty minutes into seconds."""
    import torch
    from faster_whisper.audio import decode_audio
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    WORK.mkdir(parents=True, exist_ok=True)
    lines = corpus()[:limit]
    jobs = [(ln, v, r) for ln in lines for (v, r) in VARIANTS
            if clip_path(ln, v, r).exists()]
    print(f"{len(lines)} lines · {len(jobs)} rendered clips "
          f"({len(lines) * len(VARIANTS) - len(jobs)} missing)")
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
    t0 = time.time()
    for i, (line, voice, rate) in enumerate(jobs, 1):
        try:
            wave = np.asarray(decode_audio(str(clip_path(line, voice, rate)),
                                           sampling_rate=16000), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 — a truncated render is not fatal
            print(f"  ! {line[:40]!r} {voice} {rate}: {exc}", file=sys.stderr)
            continue
        if wave.size < 1600:                      # under 0.1s: a failed render
            continue

        mel = _nemo_features(wave, N_MELS, fb, window)[0]          # (64, T)
        inputs = processor(wave, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        post = torch.log_softmax(logits, dim=-1)[0].float().cpu().numpy()   # (T', V)

        # The student subsamples its 100fps mel by 2 and the teacher's conv stack
        # subsamples 16kHz by 320, so both land on 50fps and differ by at most a frame.
        keep = min(mel.shape[1] // 2, post.shape[0])
        if keep < 4:
            continue
        mels.append(mel[:, :keep * 2].T.astype(np.float16))
        posts.append(post[:keep].astype(np.float16))
        index.append({"text": mtext.normalise(line).lower().strip(),
                      "voice": voice, "rate": rate, "frames": keep})
        if i % 200 == 0 or i == len(jobs):
            done = len(index)
            print(f"  {i:>5}/{len(jobs)}  kept {done}  "
                  f"{(time.time() - t0) / max(1, i):.2f}s/clip", flush=True)

    mel_all = np.concatenate(mels)
    post_all = np.concatenate(posts)
    np.save(WORK / "mel.npy", mel_all)
    np.save(WORK / "post.npy", post_all)
    (WORK / "index.json").write_text(json.dumps(
        {"vocab_size": v_size, "n_mels": N_MELS, "items": index}), encoding="utf-8")
    print(f"\n{len(index)} clips · mel {mel_all.nbytes / 1e6:.0f}MB "
          f"· posteriors {post_all.nbytes / 1e6:.0f}MB → {WORK}")
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

    meta = json.loads((WORK / "index.json").read_text(encoding="utf-8"))
    items = meta["items"]
    v_size = meta["vocab_size"]
    vocab = json.loads((WORK / "vocab.json").read_text(encoding="utf-8"))
    blank = vocab.get("<pad>", vocab.get("[PAD]"))
    space = "|" if "|" in vocab else " "

    mel = np.load(WORK / "mel.npy", mmap_mode="r")
    post = np.load(WORK / "post.npy", mmap_mode="r")

    # Offsets: mel runs at twice the frame rate of the posteriors.
    offs, m_off, p_off = [], 0, 0
    for it in items:
        offs.append((m_off, p_off, it["frames"]))
        m_off += it["frames"] * 2
        p_off += it["frames"]

    def encode(s: str) -> list[int]:
        return [vocab[space if ch == " " else ch] for ch in s
                if (space if ch == " " else ch) in vocab]

    targets = [encode(it["text"]) for it in items]

    # Split by *line*, so a dev sentence is never seen in another voice either.
    rng = np.random.default_rng(11)
    lines = sorted({it["text"] for it in items})
    rng.shuffle(lines)
    dev_lines = set(lines[:max(1, len(lines) // 10)])
    train_ix = [i for i, it in enumerate(items) if it["text"] not in dev_lines]
    dev_ix = [i for i, it in enumerate(items) if it["text"] in dev_lines]
    print(f"{len(train_ix)} train clips · {len(dev_ix)} dev clips "
          f"({len(lines) - len(dev_lines)}/{len(dev_lines)} lines)")

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
        frames = [offs[i][2] for i in ix]
        tmax = max(frames)
        x = np.zeros((len(ix), N_MELS, tmax * 2), dtype=np.float32)
        y = np.zeros((len(ix), tmax, v_size), dtype=np.float32)
        for row, i in enumerate(ix):
            m_off, p_off, n = offs[i]
            x[row, :, :n * 2] = mel[m_off:m_off + n * 2].T.astype(np.float32)
            y[row, :n] = post[p_off:p_off + n].astype(np.float32)
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
                torch.tensor([len(targets[i]) for i in ix], dtype=torch.long))

    def run_epoch(ix: list[int], train: bool):
        model.train(train)
        order = list(ix)
        if train:
            rng.shuffle(order)
        # Length-bucketed so padding does not dominate: sort within large chunks.
        chunks = [order[i:i + batch * 16] for i in range(0, len(order), batch * 16)]
        order = [i for ch in chunks for i in sorted(ch, key=lambda k: offs[k][2])]
        tot_kd = tot_ctc = n = 0
        for s in range(0, len(order) - batch + 1, batch):
            bx, by, blen, flat, tlen = batch_of(order[s:s + batch], augment=train)
            with torch.set_grad_enabled(train):
                out = model(bx)                                # (B, T, V)
                keep = min(out.shape[1], by.shape[1])
                out, by_ = out[:, :keep], by[:, :keep]
                mask = (torch.arange(keep)[None, :] < blen[:, None]).to(device)
                # KL(teacher ‖ student) over valid frames. The teacher's full
                # distribution is the point — a hard label per frame would throw away
                # exactly the confidence that near-miss rejection depends on.
                kd = (by_.exp() * (by_ - out)).sum(-1)
                kd = (kd * mask).sum() / mask.sum()
                # `aten::_ctc_loss` has no MPS kernel, so this one term crosses to the
                # CPU and back. `.cpu()` is differentiable, so the gradient still
                # reaches the GPU weights; the copy is ~600KB a step and does not show
                # up against the convolutions.
                ctc_loss = ctc(out.cpu().transpose(0, 1), flat.cpu(),
                               torch.clamp(blen, max=keep), tlen)
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
    ap.add_argument("stage", choices=["teacher", "train", "export"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--blocks", type=int, default=15)
    ap.add_argument("--kernel", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--kd-weight", type=float, default=0.9)
    ap.add_argument("--tag", default="student")
    args = ap.parse_args()

    if args.stage == "teacher":
        return stage_teacher(args.limit)
    if args.stage == "train":
        return stage_train(args.width, args.blocks, args.kernel, args.epochs,
                           args.batch, args.lr, args.kd_weight, args.tag)
    return stage_export(args.tag)


if __name__ == "__main__":
    raise SystemExit(main())

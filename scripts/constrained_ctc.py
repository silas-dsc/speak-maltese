#!/usr/bin/env python3
"""Does knowing the answer let a small recogniser keep up with a large one?

The app never has to transcribe. It shows a line, the learner says it, and the only
question is whether what they said was that line. Free decoding answers a harder
question than the one being asked, and `compare_stt.py` measured what that costs: the
18.9M-parameter QuartzNet returns confident wrong words — `mingħajr zokkor` heard as
`min għajd sokkor` — and lands at 80% of answers marked correct against the 315M
model's 96%. None of those mistakes is a plausible *alignment* of the target, which is
the observation this script tests.

So: score the target sequence directly against the audio with the CTC forward
algorithm, and compare that likelihood to the alternatives.

    python scripts/constrained_ctc.py

**The trap.** A decoder that only ever scores the target accepts everything, silence
included, and reports 100%. Any honest measurement of this needs answers that are
*wrong*, so every clip is scored against every target in the eval set — a full matrix.
Three numbers come out of it:

    rank-1        how often the true line beats all 24 others. The discriminative
                  question, and free of thresholds.
    app pass      the winner is fed to `text.score`, the same grader and the same 0.78
                  threshold as compare_stt.py, so the column means what it does there.
    TPR@95        share of true answers accepted at the threshold that rejects 95% of
                  wrong ones. This is the number that maps onto the app, where the
                  alternatives are not 24 known lines but anything a learner might say.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import text  # noqa: E402
from backend.config import DATA_DIR  # noqa: E402
from compare_stt import _NEMO, _mel_filters, _nemo_features  # noqa: E402

CLIPS = DATA_DIR / "eval_clips"
NEG = -1e30


# ── CTC scoring ────────────────────────────────────────────────────────────

def ctc_logp(logprobs: np.ndarray, ids: list[int], blank: int) -> float:
    """log P(ids | audio), summed over every alignment — the CTC forward algorithm.

    The extended sequence interleaves blanks (`b s1 b s2 … b`) so that a path may or
    may not emit one between tokens, and a repeated token *must* have one between its
    two copies. That last rule is why `irrid` and `irid` are different hypotheses here
    rather than the same one: exactly the distinction greedy decoding loses when it
    collapses repeats in the wrong order.

    Vectorised across the sequence axis — 625 alignments per model in this script, and
    a Python loop over both axes is minutes rather than seconds."""
    if not ids:
        return NEG
    ext = np.full(2 * len(ids) + 1, blank, dtype=np.int64)
    ext[1::2] = ids
    s = len(ext)

    # A path may skip from s-2 to s only when that does not merge two identical
    # tokens, and never onto a blank.
    skip = np.zeros(s, dtype=bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])

    a = np.full(s, NEG)
    a[0] = logprobs[0, ext[0]]
    if s > 1:
        a[1] = logprobs[0, ext[1]]

    for t in range(1, len(logprobs)):
        stay = a
        one = np.concatenate(([NEG], a[:-1]))
        two = np.where(skip, np.concatenate(([NEG, NEG], a[:-2])), NEG)
        a = np.logaddexp(np.logaddexp(stay, one), two) + logprobs[t, ext]

    return float(np.logaddexp(a[-1], a[-2]) if s > 1 else a[-1])


def greedy_logp(logprobs: np.ndarray) -> float:
    """log-probability of the single best path — the ceiling any sequence is measured
    against. Not a sequence likelihood, deliberately: it is the most the audio could
    have scored, so the gap to it is `how much worse than the best explanation`."""
    return float(logprobs.max(axis=-1).sum())


def confidence(logprobs: np.ndarray, ids: list[int], blank: int) -> float:
    """`exp` of the per-frame likelihood gap to the best path.

    Per *frame*, because a long sentence accumulates more log-probability than a short
    one and an unnormalised total would rank `Bonġu` above every full sentence in the
    deck regardless of what was said.

    Near 1 means "as good an explanation of this audio as anything". It can exceed 1,
    and does: the numerator sums over every alignment of the sequence while the
    denominator is one path, so a sequence the model is sure of beats the greedy path
    it agrees with. That is not a bug to clamp — the ordering is what is used."""
    n = len(logprobs)
    gap = (ctc_logp(logprobs, ids, blank) - greedy_logp(logprobs)) / max(1, n)
    return float(np.exp(gap))


# ── Hard negatives ─────────────────────────────────────────────────────────
# Other lines in the deck are easy negatives: `Bonġu` against `In-nanna tagħmel
# il-pastizzi` is not a decision anything gets wrong, and a matrix of those flatters
# the method. What the app has to survive is a learner who *nearly* said it — a
# dropped word, a lost geminate, the wrong assimilated article. Those share most of
# their audio with the target, so they are where a likelihood ratio earns its keep or
# does not.

_ARTICLES = [("mal-", "mill-"), ("mill-", "mal-"), ("tas-", "ta' "), ("il-", "l-"),
             ("fis-", "f'"), ("ix-", "is-"), ("is-", "ix-")]


def near_misses(target: str) -> list[str]:
    """A few plausible wrong answers for one line. Not exhaustive — each is a mistake
    a learner actually makes, and each keeps most of the target's sound."""
    flat = text.normalise(target).lower().strip()
    words = flat.split()
    out = []

    if len(words) > 1:
        out.append(" ".join(words[:-1]))                      # trailed off
        out.append(" ".join(words[1:]))                       # missed the opening
    if len(words) > 2:
        out.append(" ".join(words[:1] + words[2:]))           # dropped a middle word

    # Degemination — `irrid` → `irid`. The failure this project keeps chasing, and the
    # one a decoder that collapses repeats in the wrong order cannot even represent.
    for i in range(1, len(flat)):
        if flat[i] == flat[i - 1] and flat[i].isalpha():
            out.append(flat[:i] + flat[i + 1:])
            break

    for a, b in _ARTICLES:
        if a in flat:
            out.append(flat.replace(a, b, 1))
            break

    seen, uniq = {flat}, []
    for s in out:
        s = " ".join(s.split())
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


# ── Vocabularies ───────────────────────────────────────────────────────────

def encode(target: str, vocab: dict[str, int], space: str) -> list[int]:
    """Target text → token ids, dropping what the model has no token for.

    Both recognisers are character CTC and neither has punctuation, so `Mingħajr
    zokkor.` has to lose its full stop before it can be aligned to anything."""
    flat = text.normalise(target).lower().strip()
    ids = []
    for ch in flat:
        tok = space if ch == " " else ch
        if tok in vocab:
            ids.append(vocab[tok])
    return ids


# ── Models: audio → log-probabilities ──────────────────────────────────────

def load_nemo(name: str):
    import json

    import onnxruntime as rt
    from huggingface_hub import hf_hub_download

    src = Path(name)
    if src.is_dir():
        paths = {f: src / f for f in ("model.onnx", "vocab.txt", "config.json")}
    else:
        paths = {f: Path(hf_hub_download(name, f))
                 for f in ("model.onnx", "vocab.txt", "config.json")}
    n_mels = int(json.loads(paths["config.json"].read_text(encoding="utf-8"))
                 .get("features_size", 64))

    vocab = {}
    for line in paths["vocab.txt"].read_text(encoding="utf-8").splitlines():
        if line.strip():
            tok, idx = line.rsplit(" ", 1)
            vocab[tok] = int(idx)
    blank = vocab["<blk>"]

    fb = _mel_filters(_NEMO["n_fft"] // 2 + 1, n_mels, _NEMO["sample_rate"])
    pad = (_NEMO["n_fft"] - _NEMO["win_length"]) // 2
    window = np.pad(np.hanning(_NEMO["win_length"]).astype(np.float32), (pad, pad))
    sess = rt.InferenceSession(str(paths["model.onnx"]),
                               providers=["CPUExecutionProvider"])

    def logprobs(wave: np.ndarray) -> np.ndarray:
        feats = _nemo_features(wave, n_mels, fb, window)
        out, = sess.run(["logprobs"], {"audio_signal": feats})
        return out[0]                      # already log-softmaxed by the export

    return logprobs, vocab, blank, "▁"


def load_wav2vec2(name: str):
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(name)
    model = Wav2Vec2ForCTC.from_pretrained(name).to(device).eval()
    vocab = processor.tokenizer.get_vocab()
    blank = vocab.get("[PAD]", vocab.get("<pad>"))

    def logprobs(wave: np.ndarray) -> np.ndarray:
        inputs = processor(wave, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        return torch.log_softmax(logits, dim=-1)[0].float().cpu().numpy()

    return logprobs, vocab, blank, "|"


def load(name: str):
    if "wav2vec2" in name or "w2v" in name:
        return load_wav2vec2(name)
    return load_nemo(name)


# ── The measurement ────────────────────────────────────────────────────────

def evaluate(name: str, rows: list[dict]) -> dict:
    from faster_whisper.audio import decode_audio

    print(f"\n▸ {name}", flush=True)
    t0 = time.time()
    logprobs_for, vocab, blank, space = load(name)
    load_s = time.time() - t0

    targets = [r["text"] for r in rows]
    encoded = [encode(t, vocab, space) for t in targets]
    missing = [t for t, ids in zip(targets, encoded) if not ids]
    if missing:
        print(f"  ! {len(missing)} targets encode to nothing", file=sys.stderr)

    hard = [[encode(v, vocab, space) for v in near_misses(t)] for t in targets]
    print(f"  {sum(len(h) for h in hard)} near-misses across {len(targets)} lines")

    conf = np.zeros((len(rows), len(rows)))
    hard_conf: list[list[float]] = []
    start = time.time()
    for i, row in enumerate(rows):
        wave = np.asarray(decode_audio(str(CLIPS / row["file"]), sampling_rate=16000),
                          dtype=np.float32)
        lp = logprobs_for(wave)
        ceiling = greedy_logp(lp)
        n = len(lp)
        for j, ids in enumerate(encoded):
            conf[i, j] = np.exp((ctc_logp(lp, ids, blank) - ceiling) / max(1, n))
        hard_conf.append([float(np.exp((ctc_logp(lp, ids, blank) - ceiling) / max(1, n)))
                          for ids in hard[i]])
        chosen = targets[int(conf[i].argmax())]
        worst_near = max(hard_conf[i], default=0.0)
        mark = "✓" if chosen == row["text"] and conf[i, i] > worst_near else "✗"
        print(f"  {i + 1:>3}/{len(rows)} {mark} conf {conf[i, i]:.3f} "
              f"(other lines {np.delete(conf[i], i).max():.3f} · "
              f"near-miss {worst_near:.3f})  {chosen[:36]}", flush=True)
    elapsed = time.time() - start

    true_conf = np.diag(conf)
    wrong = conf[~np.eye(len(rows), dtype=bool)]
    near = np.array([c for row_c in hard_conf for c in row_c])
    rank1 = float(np.mean([conf[i].argmax() == i for i in range(len(rows))]))
    # Beating the other lines is not enough: the target has to beat every near-miss of
    # itself too, which is the decision the app is actually making.
    beats_near = float(np.mean([true_conf[i] > max(hard_conf[i], default=0.0)
                                for i in range(len(rows))]))
    scores = [text.score(targets[int(conf[i].argmax())], rows[i]["text"])
              for i in range(len(rows))]

    # The threshold that rejects 95% of wrong answers, and what it costs in accepted
    # right ones. This is the pair the app actually has to live with.
    def tpr_at(neg: np.ndarray) -> tuple[float, float]:
        cut = float(np.quantile(neg, 0.95))
        return cut, float(np.mean(true_conf >= cut))

    cut, tpr = tpr_at(wrong)
    hcut, htpr = tpr_at(near) if near.size else (float("nan"), float("nan"))
    return {
        "model": name, "load_s": load_s, "sec_per_clip": elapsed / max(1, len(rows)),
        "rank1": rank1, "beats_near": beats_near,
        "score": float(np.mean(scores)),
        "pass_rate": float(np.mean([s >= 0.78 for s in scores])),
        "true_conf": float(true_conf.mean()),
        "wrong_conf": float(wrong.mean()),
        "near_conf": float(near.mean()) if near.size else float("nan"),
        "cut": cut, "tpr_at_95": tpr,
        "hard_cut": hcut, "hard_tpr_at_95": htpr,
        "conf": conf, "hard_conf": hard_conf, "targets": targets,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="OpenVoiceOS/carlosdanielhernandezmena-"
                                        "stt_mt_quartznet15x5_sp_ep255_64h_onnx,"
                                        "carlosdanielhernandezmena/"
                                        "wav2vec2-large-xlsr-53-maltese-64h")
    ap.add_argument("--worst", type=int, default=5)
    args = ap.parse_args()

    manifest = CLIPS / "manifest.tsv"
    if not manifest.exists():
        print("No clips. Run compare_stt.py --synth 25 first.", file=sys.stderr)
        return 2
    with manifest.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if r.get("file")]
    rows = [r for r in rows if (CLIPS / r["file"]).exists()]

    print(f"Scoring {len(rows)} clips against all {len(rows)} targets "
          f"({len(rows) ** 2} alignments per model)")

    reports = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        try:
            reports.append(evaluate(name, rows))
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {name} failed: {exc}", file=sys.stderr)
    if not reports:
        return 1

    print("\n" + "═" * 100)
    print(f"{'model':<38}{'rank-1':>8}{'>near':>7}{'pass':>6}"
          f"{'conf ✓':>9}{'other':>8}{'near':>8}{'TPR@95':>8}{'hard':>7}{'s/clip':>8}")
    print("─" * 100)
    for r in sorted(reports, key=lambda r: -r["beats_near"]):
        name = r["model"] if len(r["model"]) <= 37 else "…" + r["model"][-36:]
        print(f"{name:<38}{r['rank1']:>8.0%}{r['beats_near']:>7.0%}"
              f"{r['pass_rate']:>6.0%}{r['true_conf']:>9.3f}{r['wrong_conf']:>8.3f}"
              f"{r['near_conf']:>8.3f}{r['tpr_at_95']:>8.0%}"
              f"{r['hard_tpr_at_95']:>7.0%}{r['sec_per_clip']:>8.2f}")
    print("═" * 100)
    for r in reports:
        # The number the app has to hard-code, so print it rather than making someone
        # re-derive it: accept at or above this and 95% of near-misses are turned away.
        print(f"  accept threshold for {r['model'].split('/')[-1]}: "
              f"{r['hard_cut']:.4f}  (other-lines cut {r['cut']:.4f})")
    print("  rank-1 = beats the other 24 lines · >near = also beats every near-miss of "
          "itself\n  TPR@95 / hard = right answers kept at the cut that rejects 95% of "
          "other lines / near-misses")

    for r in reports:
        order = np.argsort(np.diag(r["conf"]))[:args.worst]
        print(f"\n  tightest margins for {r['model'].split('/')[-1]}:")
        for i in order:
            j = int(r["conf"][i].argmax())
            near = max(r["hard_conf"][i], default=0.0)
            print(f"    {r['conf'][i, i]:.3f} vs {near:.3f} near-miss  "
                  f"{r['targets'][i]}"
                  + ("" if j == i else f"   → picked: {r['targets'][j]}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Can the learner's audio be matched against stored reference audio, with no model?

The app knows every line it will ever ask for and already ships a recording of each
one — 1,584 pre-rendered MP3s. So the recogniser could be replaced by a comparison:
encode what was said, encode what should have been said, and see whether they line up.
If plain cepstral features are enough for that, the 200MB model and the server both go
away.

The catch is that the stored side is synthetic and the learner is not, so anything
measured with one voice against itself is measuring nothing. Here the reference is the
app's own `mt-MT-GraceNeural` clips and the query is the same sentence in
`mt-MT-JosephNeural` — a different synthetic speaker, and a different gender. Not a
human, but a real speaker mismatch rather than none.

    python scripts/dtw_match.py --synth        # render the query voice, once
    python scripts/dtw_match.py

Negatives are the same near-misses `constrained_ctc.py` uses — a dropped word, a lost
geminate, a swapped article — spoken in the query voice and matched against the
target's reference. Those share most of their sound with the target and are the only
negatives that decide anything.

Three encoders, same metrics:

    mfcc            13 cepstra + CMVN. No model at all, a few lines of arithmetic.
    quartznet       44-dim CTC posteriors from the 18.9M model (76MB).
    wav2vec2        39-dim CTC posteriors from the 315M model (201MB) — the ceiling.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import text  # noqa: E402
from backend.config import CFG, DATA_DIR  # noqa: E402
from compare_stt import _NEMO, _mel_filters, _nemo_features  # noqa: E402
from constrained_ctc import load_nemo, load_wav2vec2, near_misses  # noqa: E402

CLIPS = DATA_DIR / "eval_clips"
QUERIES = CLIPS / "xvoice"
INF = np.inf


# ── The query voice ────────────────────────────────────────────────────────

def _query_path(line: str) -> Path:
    return QUERIES / f"{hashlib.sha256(line.encode()).hexdigest()[:12]}.mp3"


async def render(lines: list[str], voice: str) -> None:
    from backend import tts

    QUERIES.mkdir(parents=True, exist_ok=True)
    todo = [ln for ln in lines if not _query_path(ln).exists()]
    print(f"{len(lines)} lines, {len(todo)} to render in {voice}")
    for i, line in enumerate(todo, 1):
        audio, _ = await tts.synthesize(line, voice, rate=1.0)
        _query_path(line).write_bytes(audio)
        print(f"  {i:>4}/{len(todo)}  {line[:60]}", flush=True)


# ── Encoders ───────────────────────────────────────────────────────────────

def _dct_matrix(n_in: int, n_out: int) -> np.ndarray:
    """DCT-II, orthonormal — the mel → cepstrum step, written out rather than pulled
    from scipy so this script needs nothing that is not already installed."""
    k = np.arange(n_out)[:, None]
    n = np.arange(n_in)[None, :]
    m = np.sqrt(2.0 / n_in) * np.cos(np.pi * k * (2 * n + 1) / (2 * n_in))
    m[0] *= np.sqrt(0.5)
    return m.astype(np.float32)


def mfcc_encoder(n_cep: int = 13):
    fb = _mel_filters(_NEMO["n_fft"] // 2 + 1, 64, _NEMO["sample_rate"])
    pad = (_NEMO["n_fft"] - _NEMO["win_length"]) // 2
    window = np.pad(np.hanning(_NEMO["win_length"]).astype(np.float32), (pad, pad))
    dct = _dct_matrix(64, n_cep)

    def encode(wave: np.ndarray) -> np.ndarray:
        mel = _nemo_features(wave, 64, fb, window)[0]        # (64, T), already CMVN'd
        cep = dct @ mel                                       # (n_cep, T)
        # c0 is loudness, not content — a louder learner is not a different word.
        cep = cep[1:]
        cep -= cep.mean(axis=1, keepdims=True)
        cep /= cep.std(axis=1, keepdims=True) + 1e-5
        return cep.T                                          # (T, n_cep-1)

    return encode


def posterior_encoder(name: str):
    logprobs_for, _vocab, _blank, _space = (
        load_wav2vec2(name) if ("wav2vec2" in name or "w2v" in name) else load_nemo(name))

    def encode(wave: np.ndarray) -> np.ndarray:
        # Probabilities, not log-probabilities: the frames are compared by angle, and
        # in log space every near-zero class contributes a large negative number that
        # swamps the one class that was actually recognised.
        return np.exp(logprobs_for(wave))

    return encode


# ── DTW ────────────────────────────────────────────────────────────────────

def dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised DTW cost between two frame sequences, cosine local distance.

    Step pattern is the symmetric (1,1) (1,2) (2,1) — both indices always advance, so
    each row depends only on the two before it and the whole thing vectorises. A
    plain (1,0) (0,1) (1,1) pattern has a within-row dependency and would need a
    Python loop per cell, which is minutes rather than seconds over these pairs.

    The pattern constrains the slope to between ½ and 2, so a pair whose lengths differ
    by more than 2× has no valid path and comes back as infinite. For near-misses of
    the same line that never bites; across different lines it sometimes does, and
    `rank-1` below is generous by exactly that much."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    cost = 1.0 - an @ bn.T                                    # (N, M)
    n, m = cost.shape

    prev2 = np.full(m, INF)
    prev1 = np.full(m, INF)
    for i in range(n):
        if i == 0:
            cur = np.full(m, INF)
            cur[0] = cost[0, 0]
        else:
            d1 = np.concatenate(([INF], prev1[:-1]))           # (i-1, j-1)
            d2 = np.concatenate(([INF, INF], prev1[:-2]))      # (i-1, j-2)
            d3 = np.concatenate(([INF], prev2[:-1]))           # (i-2, j-1)
            cur = np.minimum(np.minimum(d1, d2), d3) + cost[i]
        prev2, prev1 = prev1, cur
    return float(prev1[-1] / (n + m))


# ── The measurement ────────────────────────────────────────────────────────

def evaluate(label: str, encode, rows: list[dict], hard: dict) -> dict:
    from faster_whisper.audio import decode_audio

    def wave(path: Path) -> np.ndarray:
        return np.asarray(decode_audio(str(path), sampling_rate=16000), dtype=np.float32)

    print(f"\n▸ {label}", flush=True)
    t0 = time.time()
    targets = [r["text"] for r in rows]
    refs = [encode(wave(CLIPS / r["file"])) for r in rows]
    queries = [encode(wave(_query_path(t))) for t in targets]
    encode_s = time.time() - t0

    start = time.time()
    cost = np.array([[dtw(q, r) for r in refs] for q in queries])   # query i vs ref j
    near_cost = []
    for i, t in enumerate(targets):
        near_cost.append([dtw(encode(wave(_query_path(v))), refs[i]) for v in hard[t]])
        best_other = np.delete(cost[i], i).min()
        worst_near = min(near_cost[i], default=INF)
        ok = cost[i, i] < best_other and cost[i, i] < worst_near
        print(f"  {i + 1:>3}/{len(rows)} {'✓' if ok else '✗'} "
              f"cost {cost[i, i]:.4f} (other lines {best_other:.4f} · "
              f"near-miss {worst_near:.4f})  {t[:34]}", flush=True)
    elapsed = time.time() - start

    true_cost = np.diag(cost)
    other = cost[~np.eye(len(rows), dtype=bool)]
    other = other[np.isfinite(other)]
    near = np.array([c for row in near_cost for c in row])
    near = near[np.isfinite(near)]

    # Lower is better here, so the accept threshold is a *low* cut and 95% of negatives
    # must sit above it.
    def tpr_at(neg: np.ndarray) -> float:
        if not neg.size:
            return float("nan")
        return float(np.mean(true_cost <= float(np.quantile(neg, 0.05))))

    return {
        "model": label, "encode_s": encode_s,
        "sec_per_pair": elapsed / max(1, cost.size + near.size),
        "rank1": float(np.mean([cost[i].argmin() == i for i in range(len(rows))])),
        "beats_near": float(np.mean([true_cost[i] < min(near_cost[i], default=INF)
                                     for i in range(len(rows))])),
        "true_cost": float(true_cost.mean()),
        "other_cost": float(other.mean()) if other.size else float("nan"),
        "near_cost": float(near.mean()) if near.size else float("nan"),
        "tpr_at_95": tpr_at(other), "hard_tpr_at_95": tpr_at(near),
        "cost": cost, "near": near_cost, "targets": targets,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", action="store_true",
                    help="render the query voice (needed once)")
    ap.add_argument("--voice", default=CFG.azure_voice_alt)
    ap.add_argument("--encoders", default="mfcc,quartznet,wav2vec2")
    ap.add_argument("--worst", type=int, default=5)
    args = ap.parse_args()

    manifest = CLIPS / "manifest.tsv"
    if not manifest.exists():
        print("No clips. Run compare_stt.py --synth 25 first.", file=sys.stderr)
        return 2
    with manifest.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if r.get("file")]
    rows = [r for r in rows if (CLIPS / r["file"]).exists()]
    hard = {r["text"]: near_misses(r["text"]) for r in rows}

    wanted = [r["text"] for r in rows] + [v for vs in hard.values() for v in vs]
    if args.synth:
        asyncio.run(render(wanted, args.voice))
    missing = [ln for ln in wanted if not _query_path(ln).exists()]
    if missing:
        print(f"{len(missing)} query clips missing — run with --synth", file=sys.stderr)
        return 2

    print(f"\n{len(rows)} lines · {sum(len(v) for v in hard.values())} near-misses · "
          f"reference {CFG.azure_voice} vs query {args.voice}")

    builders = {
        "mfcc": lambda: ("mfcc (no model)", mfcc_encoder()),
        "quartznet": lambda: ("quartznet posteriors (76MB)", posterior_encoder(
            "OpenVoiceOS/carlosdanielhernandezmena-stt_mt_quartznet15x5_sp_ep255_64h_onnx")),
        "wav2vec2": lambda: ("wav2vec2 posteriors (201MB)", posterior_encoder(
            "carlosdanielhernandezmena/wav2vec2-large-xlsr-53-maltese-64h")),
    }
    reports = []
    for key in [k.strip() for k in args.encoders.split(",") if k.strip()]:
        try:
            label, encode = builders[key]()
            reports.append(evaluate(label, encode, rows, hard))
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {key} failed: {exc}", file=sys.stderr)
    if not reports:
        return 1

    print("\n" + "═" * 94)
    print(f"{'encoder':<30}{'rank-1':>8}{'<near':>7}{'cost ✓':>9}{'other':>9}"
          f"{'near':>9}{'TPR@95':>8}{'hard':>7}")
    print("─" * 94)
    for r in sorted(reports, key=lambda r: -r["beats_near"]):
        print(f"{r['model']:<30}{r['rank1']:>8.0%}{r['beats_near']:>7.0%}"
              f"{r['true_cost']:>9.4f}{r['other_cost']:>9.4f}{r['near_cost']:>9.4f}"
              f"{r['tpr_at_95']:>8.0%}{r['hard_tpr_at_95']:>7.0%}")
    print("═" * 94)
    print("  <near = the true reference is closer than every near-miss · hard = right "
          "answers\n  kept at the cut that rejects 95% of near-misses · lower cost is "
          "closer")

    for r in reports:
        order = np.argsort(-np.diag(r["cost"]))[:args.worst]
        print(f"\n  worst matches for {r['model']}:")
        for i in order:
            j = int(r["cost"][i].argmin())
            print(f"    {r['cost'][i, i]:.4f} vs {min(r['near'][i], default=INF):.4f} "
                  f"near-miss  {r['targets'][i]}"
                  + ("" if j == i else f"   → picked: {r['targets'][j]}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

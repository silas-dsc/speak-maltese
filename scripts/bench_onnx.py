#!/usr/bin/env python3
"""Does the Maltese recogniser survive being quantized for the browser?

The app's speech recognition is the last thing keeping a server in the loop: the
schedule, the deck and the progress all run client-side now. transformers.js can
run this same wav2vec2 CTC model in the browser through ONNX Runtime Web, which
would make the whole app static — free hosting anywhere, no cold starts, no
sleeping container, and it would work offline.

The cost is the download: 1.26GB of fp32 weights, or ~355MB quantized to int8.
Quantization is where a low-resource-language model is most likely to lose
accuracy, and "it still produces Maltese-looking text" is not a measurement. So
all three are run over the same clips and scored the same way:

    torch      — what the app serves today, the reference
    onnx-fp32  — the export, to separate export loss from quantization loss
    onnx-int8  — what would actually ship to a browser

Scores are word and character error rate against the reference transcript, both
raw and folded. Folded is the one that matters: recognisers routinely drop
Maltese diacritics and split the fused article, and the app's grading already
forgives both, so a raw WER counts mistakes the learner never sees.

    python scripts/bench_onnx.py --onnx /tmp/onnx-mt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLIPS = ROOT / "data" / "eval_clips"
MANIFEST = CLIPS / "manifest.tsv"


def _metrics():
    spec = importlib.util.spec_from_file_location(
        "compare_stt", ROOT / "scripts" / "compare_stt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resample(audio: np.ndarray, src_hz: int, dst_hz: int = 16000) -> np.ndarray:
    """Linear resample. The clips are 24kHz TTS output and wav2vec2 wants 16kHz.

    Deliberately not librosa: it drags in numba, which does not import on this
    Python, and the point of this script is to measure the model rather than to
    litigate an audio stack. Linear interpolation is applied identically to every
    run, so it cannot favour one over another.
    """
    if src_hz == dst_hz:
        return audio.astype(np.float32)
    n = int(round(len(audio) * dst_hz / src_hz))
    x = np.linspace(0, len(audio) - 1, n, dtype=np.float64)
    return np.interp(x, np.arange(len(audio)), audio).astype(np.float32)


def load_clips() -> list[tuple[str, str, np.ndarray]]:
    import soundfile as sf

    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        name, text = line.split("\t", 1)
        audio, sr = sf.read(CLIPS / name, dtype="float32", always_2d=True)
        rows.append((name, text, _resample(audio.mean(axis=1), sr)))
    return rows


def run_torch(model_id: str, clips) -> tuple[list[str], float]:
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    proc = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).eval()

    out, elapsed = [], 0.0
    for _name, _ref, audio in clips:
        inputs = proc(audio, sampling_rate=16000, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        elapsed += time.perf_counter() - t0
        ids = torch.argmax(logits, dim=-1)
        out.append(proc.batch_decode(ids)[0])
    return out, elapsed


def run_onnx(onnx_dir: Path, filename: str, clips) -> tuple[list[str], float]:
    import onnxruntime as ort
    from transformers import Wav2Vec2Processor

    proc = Wav2Vec2Processor.from_pretrained(str(onnx_dir))
    # One thread: a browser gets one WASM thread by default unless the page is
    # cross-origin isolated, so this is closer to what a learner would feel than
    # letting onnxruntime use every core on this laptop.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_dir / filename), opts,
                                providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    out, elapsed = [], 0.0
    for _name, _ref, audio in clips:
        feats = proc(audio, sampling_rate=16000, return_tensors="np")
        t0 = time.perf_counter()
        logits = sess.run(None, {input_name: feats.input_values.astype(np.float32)})[0]
        elapsed += time.perf_counter() - t0
        ids = np.argmax(logits, axis=-1)
        out.append(proc.batch_decode(ids)[0])
    return out, elapsed


def score(m, hyps, clips) -> dict:
    refs = [ref for _n, ref, _a in clips]
    return {
        "wer": sum(m.wer(h, r) for h, r in zip(hyps, refs)) / len(refs),
        "wer_folded": sum(m.wer(h, r, folded=True) for h, r in zip(hyps, refs)) / len(refs),
        "cer": sum(m.cer(h, r) for h, r in zip(hyps, refs)) / len(refs),
        "exact": sum(1 for h, r in zip(hyps, refs)
                     if m.wer(h, r, folded=True) == 0) / len(refs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, default=Path("/tmp/onnx-mt"))
    ap.add_argument("--model", default="carlosdanielhernandezmena/wav2vec2-large-xlsr-53-maltese-64h")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    m = _metrics()
    clips = load_clips()
    audio_seconds = sum(len(a) for _n, _r, a in clips) / 16000
    print(f"{len(clips)} clips · {audio_seconds:.0f}s of audio\n")

    runs = [("torch", lambda: run_torch(args.model, clips))]
    for label, fn in (("onnx-fp32", "model.onnx"), ("onnx-int8", "model_quantized.onnx")):
        if (args.onnx / fn).exists():
            runs.append((label, (lambda f=fn: run_onnx(args.onnx, f, clips))))

    results = {}
    hyps_by_run = {}
    for label, fn in runs:
        hyps, elapsed = fn()
        hyps_by_run[label] = hyps
        results[label] = {**score(m, hyps, clips),
                          "seconds": elapsed,
                          "realtime_factor": audio_seconds / elapsed}
        print(f"{label:10s} done in {elapsed:.1f}s")

    print(f"\n{'':10s} {'WER':>7} {'WER↓fold':>9} {'CER':>7} {'exact':>7} {'xRT':>7}")
    for label, r in results.items():
        print(f"{label:10s} {r['wer']:7.3f} {r['wer_folded']:9.3f} {r['cer']:7.3f} "
              f"{r['exact']:7.0%} {r['realtime_factor']:6.1f}x")

    base = results.get("torch")
    if base and "onnx-int8" in results:
        d = results["onnx-int8"]["wer_folded"] - base["wer_folded"]
        print(f"\nint8 vs torch, folded WER: {d:+.4f}")

    # Every disagreement, so the damage can be read rather than inferred from a mean.
    if "onnx-int8" in hyps_by_run:
        print("\nWhere int8 differs from torch:")
        shown = 0
        for i, (_name, ref, _a) in enumerate(clips):
            a, b = hyps_by_run["torch"][i], hyps_by_run["onnx-int8"][i]
            if a.strip() != b.strip():
                shown += 1
                print(f"  ref   {ref}")
                print(f"  torch {a}")
                print(f"  int8  {b}")
        if not shown:
            print("  (identical on every clip)")

    if args.out:
        args.out.write_text(json.dumps(
            {"results": results, "hyps": hyps_by_run,
             "refs": [r for _n, r, _a in clips]}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

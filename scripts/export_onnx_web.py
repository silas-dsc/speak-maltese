#!/usr/bin/env python3
"""Build the browser copies of the Maltese recogniser.

Speech recognition is the last thing keeping a server in the loop. This produces
the files that would let it run in the page instead — ONNX, in the precisions
ONNX Runtime Web can actually accelerate, laid out the way transformers.js expects:

    web/models/mt-w2v2/
        config.json  preprocessor_config.json  vocab.json  tokenizer_config.json
        onnx/model.onnx           fp32   1262 MB
        onnx/model_fp16.onnx      fp16    631 MB
        onnx/model_q4.onnx        4-bit   246 MB
        onnx/model_q4f16.onnx     4-bit    201 MB   ← the one to ship

Measured on an M-series GPU over 25 clips (scripts/bench_onnx.py has the table):
q4f16 runs at ~30x realtime and q4 at ~26x, against 23x for fp16 and 0.22x for
int8 on WASM. int8 is deliberately not built here: onnxruntime-web has no int8 GPU
kernel, so a quantized model falls back to the CPU path and the size saving buys
nothing. 4-bit uses MatMulNBits, which does have one.

    python scripts/export_onnx_web.py                 # everything
    python scripts/export_onnx_web.py --only q4f16    # just the shipping build

Needs `pip install optimum onnx onnxruntime onnxconverter-common onnx_ir`, none of
which the app itself uses.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "web" / "models" / "mt-w2v2"
DEFAULT_MODEL = "carlosdanielhernandezmena/wav2vec2-large-xlsr-53-maltese-64h"

# Files transformers.js reads besides the weights.
SIDECARS = ("config.json", "preprocessor_config.json", "tokenizer_config.json",
            "vocab.json", "special_tokens_map.json", "added_tokens.json")


def mb(p: Path) -> str:
    return f"{p.stat().st_size / 1e6:.0f} MB"


def export_fp32(model_id: str, work: Path) -> Path:
    if (work / "model.onnx").exists():
        print(f"fp32   already exported ({mb(work / 'model.onnx')})")
        return work / "model.onnx"
    print(f"fp32   exporting {model_id} …")
    subprocess.run(
        [sys.executable, "-m", "optimum.exporters.onnx", "--model", model_id,
         "--task", "automatic-speech-recognition", str(work)],
        check=True,
    )
    return work / "model.onnx"


def build_fp16(src: Path, dst: Path) -> None:
    import onnx
    from onnxconverter_common import float16

    print("fp16   converting …")
    model = onnx.load(str(src))
    # keep_io_types: the page feeds float32 audio in and reads float32 logits out,
    # so only the interior runs at half precision.
    out = float16.convert_float_to_float16(model, keep_io_types=True,
                                           disable_shape_infer=True)
    onnx.save(out, str(dst))


def build_4bit(src: Path, dst: Path, block_size: int = 32) -> None:
    import onnx
    from onnxruntime.quantization import matmul_nbits_quantizer as M

    print(f"{dst.stem[6:] or '4bit':6} quantizing …")
    model = onnx.load(str(src))
    cfg = M.DefaultWeightOnlyQuantConfig(block_size=block_size, is_symmetric=False,
                                         bits=4)
    q = M.MatMulNBitsQuantizer(model, algo_config=cfg)
    q.process()
    onnx.save(q.model.model if hasattr(q.model, "model") else q.model, str(dst))


def write_tokenizer(dirpath: Path) -> None:
    """transformers.js wants a fast-tokenizer file; Wav2Vec2CTCTokenizer has none.

    The decoder is deliberately *not* CTC. transformers.js's ASR pipeline already
    collapses repeated frames, and a CTC decoder here runs the collapse a second
    time over the decoded characters — which eats every Maltese geminate, turning
    `grazzi` into `grazi` and `kollox` into `kolox`. Anything reading these files
    should do its own argmax and CTC collapse (merge repeats, *then* drop blanks —
    that order matters for the same reason). web/stt-test.html shows it.
    """
    vocab = json.loads((dirpath / "vocab.json").read_text(encoding="utf-8"))
    cfg = json.loads((dirpath / "tokenizer_config.json").read_text(encoding="utf-8"))
    pad = cfg.get("pad_token", "<pad>")
    unk = cfg.get("unk_token", "<unk>")
    delim = cfg.get("word_delimiter_token", "|")
    specials = [cfg.get("bos_token"), cfg.get("eos_token"), unk, pad]

    tokenizer = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [
            {"id": vocab[t], "content": t, "single_word": False, "lstrip": False,
             "rstrip": False, "normalized": False, "special": True}
            for t in dict.fromkeys(x for x in specials if x and x in vocab)
        ],
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": delim},
        "pre_tokenizer": {"type": "Split", "pattern": {"String": ""},
                          "behavior": "Isolated", "invert": False},
        "post_processor": None,
        "decoder": {"type": "Replace", "pattern": {"String": delim}, "content": " "},
        "model": {"type": "WordPiece", "unk_token": unk,
                  "continuing_subword_prefix": "", "max_input_chars_per_word": 100,
                  "vocab": vocab},
    }
    (dirpath / "tokenizer.json").write_text(
        json.dumps(tokenizer, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--work", type=Path, default=Path("/tmp/onnx-mt"),
                    help="where the raw fp32 export is cached between runs")
    ap.add_argument("--only", choices=["fp32", "fp16", "q4", "q4f16"], default=None)
    args = ap.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    (OUT / "onnx").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    fp32 = export_fp32(args.model, args.work)

    for name in SIDECARS:
        src = args.work / name
        if src.exists():
            shutil.copy(src, OUT / name)
    write_tokenizer(OUT)

    want = {args.only} if args.only else {"fp32", "fp16", "q4", "q4f16"}
    onnx_dir = OUT / "onnx"

    if "fp32" in want:
        shutil.copy(fp32, onnx_dir / "model.onnx")
    if {"fp16", "q4f16"} & want:
        fp16 = args.work / "model_fp16.onnx"
        if not fp16.exists():
            build_fp16(fp32, fp16)
        if "fp16" in want:
            shutil.copy(fp16, onnx_dir / "model_fp16.onnx")
        if "q4f16" in want:
            build_4bit(fp16, onnx_dir / "model_q4f16.onnx")
    if "q4" in want:
        build_4bit(fp32, onnx_dir / "model_q4.onnx")

    print(f"\n{OUT.relative_to(ROOT)}")
    for p in sorted(onnx_dir.glob("*.onnx")):
        note = "  ← ships" if p.name == "model_q4f16.onnx" else ""
        over = "  ⚠ over GitHub's 100MB file limit" if p.stat().st_size > 100e6 else ""
        print(f"  onnx/{p.name:22} {mb(p):>8}{note}{over}")
    print(f"\ndone in {time.time() - t0:.0f}s")
    print("test it with:  python -m http.server 8000  →  /web/stt-test.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

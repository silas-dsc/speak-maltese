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


def record(n: int) -> None:
    """Prompt for each sentence and record from the default input via ffmpeg."""
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required for --record (brew install ffmpeg)")
    CLIPS.mkdir(parents=True, exist_ok=True)
    rows = _read_manifest()
    existing = {r["file"] for r in rows}
    for i, sentence in enumerate(_sentences(n), 1):
        name = f"me_{i:03d}.wav"
        if name in existing:
            continue
        print(f"\n[{i}/{n}]  Say:  {sentence}")
        input("       press Enter, speak, then press Enter again to stop… ")
        proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "avfoundation", "-i", ":default",
             "-ar", "16000", "-ac", "1", str(CLIPS / name)],
            stdin=subprocess.PIPE,
        )
        input()
        proc.communicate(b"q")
        rows.append({"file": name, "text": sentence})
        _write_manifest(rows)
    print(f"\n✓ {len(rows)} clips in {CLIPS}")


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


def _read_manifest() -> list[dict]:
    if not MANIFEST.exists():
        return []
    with MANIFEST.open(encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh, delimiter="\t") if r.get("file")]


# ── Comparison ─────────────────────────────────────────────────────────────

def run_model(name: str, rows: list[dict], device: str, beam: int) -> dict:
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
        results.append({
            "ref": row["text"], "hyp": hyp,
            "wer": wer(hyp, row["text"]),
            "fwer": wer(hyp, row["text"], folded=True),
            "cer": cer(hyp, row["text"]),
            "score": text.score(hyp, row["text"]),
        })
        print(f"  {i:>3}/{len(rows)}  score {results[-1]['score']:.2f}  {hyp[:58]}",
              flush=True)
    elapsed = time.time() - t_start
    del model

    n = len(results) or 1
    return {
        "model": name, "n": len(results), "load_s": load_s,
        "sec_per_clip": elapsed / n,
        "wer": sum(r["wer"] for r in results) / n,
        "fwer": sum(r["fwer"] for r in results) / n,
        "cer": sum(r["cer"] for r in results) / n,
        "score": sum(r["score"] for r in results) / n,
        "pass_rate": sum(1 for r in results if r["score"] >= 0.78) / n,
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="small,carlosdanielhernandezmena/"
                                        "whisper-large-maltese-8k-steps-64h-ct2")
    ap.add_argument("--synth", type=int, metavar="N",
                    help="build N clips with the app's own Maltese TTS voice")
    ap.add_argument("--record", type=int, metavar="N",
                    help="record N clips in your own voice")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--worst", type=int, default=5, help="show N worst clips per model")
    args = ap.parse_args()

    if args.synth:
        asyncio.run(synth(args.synth, args.voice))
    if args.record:
        record(args.record)

    rows = _read_manifest()
    if not rows:
        print("No clips. Run with --synth 25 or --record 20 first.", file=sys.stderr)
        return 2

    synthetic = any(r["file"].startswith("synth_") for r in rows)
    print(f"\nComparing on {len(rows)} clips"
          + ("  (synthetic — ranking is the result, not the absolute numbers)"
             if synthetic else "  (your voice)"))

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
        a, b = reports[0], reports[-1]
        delta = a["fwer"] - b["fwer"]
        better, worse = (b, a) if delta < 0 else (a, b)
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

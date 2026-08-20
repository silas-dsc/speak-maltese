#!/usr/bin/env python3
"""MASRI-SYNTHETIC parquet → audio files the teacher pass can read.

99 hours over **210 distinct voices**, against the two the `tts` shard holds now. Speaker
diversity is what the low-resource literature credits the synthetic half of a data mix with
contributing — *Flavors of Moonshine* reaches within 3.2% of a 28× larger model on 19.6
hours of Ukrainian and names diversity rather than volume as the reason — so the sampling
here is stratified by speaker on purpose. Taking the first N rows would spend the GPU on a
handful of voices and throw the one thing this corpus has away.

Transcripts come with it, so this can feed `constrain` as well as the teacher.

    python scripts/extract_masri.py --per-speaker 40        # ~8400 utterances

Licence: CC-BY-NC-SA-4.0, attribution to the University of Malta. NonCommercial and
ShareAlike plausibly reach a model trained on this. See RUNBOOK.md.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "data" / "corpora" / "masri_raw"
DST = ROOT / "data" / "corpora" / "masri"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=DST)
    ap.add_argument("--per-speaker", type=int, default=40,
                    help="utterances to take from each voice; the point of the corpus is "
                         "how many voices there are, so this is the axis to spend on")
    ap.add_argument("--max-seconds", type=float, default=12.0,
                    help="skip anything longer; the shard holds passes of about two "
                         "seconds and a long pass drifts")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    files = sorted(args.src.rglob("*.parquet"))
    if not files:
        print(f"no parquet under {args.src}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    taken: dict[str, int] = defaultdict(int)
    rows: list[dict] = []
    secs = 0.0
    for path in files:
        pf = pq.ParquetFile(str(path))
        for batch in pf.iter_batches(batch_size=256):
            d = batch.to_pydict()
            for i in range(len(d["audio_id"])):
                spk = d["speaker_id"][i]
                if taken[spk] >= args.per_speaker:
                    continue
                dur = float(d["duration"][i] or 0.0)
                if dur <= 0 or dur > args.max_seconds:
                    continue
                blob = (d["audio"][i] or {}).get("bytes")
                if not blob:
                    continue
                name = f"{d['audio_id'][i]}.wav"
                (args.out / name).write_bytes(blob)
                taken[spk] += 1
                secs += dur
                rows.append({"file": name, "text": d["normalized_text"][i] or "",
                             "speaker": spk, "gender": d["gender"][i] or "",
                             "seconds": f"{dur:.2f}"})
        print(f"  {path.name}: {len(rows)} utterances, {len(taken)} voices, "
              f"{secs / 3600:.2f}h", flush=True)

    with (args.out / "manifest.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, delimiter="\t",
                           fieldnames=["file", "text", "speaker", "gender", "seconds"])
        w.writeheader()
        w.writerows(rows)

    full = sum(1 for v in taken.values() if v >= args.per_speaker)
    print(f"\n✓ {len(rows)} utterances, {secs / 3600:.2f}h, {len(taken)} voices "
          f"({full} of them full) → {args.out}")
    print(f"  mean {secs / max(1, len(rows)):.1f}s"
          f" · transcripts in manifest.tsv, so `constrain` can use this too")
    print(f"  next:  python scripts/distill_stt.py teacher --sources corpus "
          f"--corpus-name {args.out.name} --shard masri")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

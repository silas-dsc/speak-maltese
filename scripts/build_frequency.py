#!/usr/bin/env python3
"""Derive a real Maltese frequency list, and retier the deck from it.

The deck's tiers were hand-assigned by usefulness. This measures them instead.

Sources, in preference order:

* **MLRS Korpus Malti** (467M tokens, 19 genres, University of Malta) is the right
  source and the script will use it — but the dataset is *gated*, and accepting its
  terms is the repo owner's decision, not something a script should do. Accept at
  huggingface.co/datasets/MLRS/korpus_malti and re-run with `--korpus`. Note it is
  CC BY-NC-SA 4.0, so anything derived from it inherits non-commercial + share-alike.
* **Maltese Wikipedia** (CC BY-SA 3.0/4.0) is the ungated default. Encyclopedic
  register, which under-counts conversational words — `jekk jogħġbok` is rare in an
  encyclopedia and constant in a café — so Tatoeba is blended in to pull the everyday
  vocabulary back up.

Frequency alone is also not the same as usefulness. A learner needs `jekk jogħġbok`
early however rare it is in print, so retiering is *advisory*: it writes a proposal
you review, and never silently rewrites the deck.

    python scripts/build_frequency.py                  # build data/frequency_mt.tsv
    python scripts/build_frequency.py --retier         # + propose new deck tiers
    python scripts/build_frequency.py --korpus         # use Korpus Malti if granted
"""

from __future__ import annotations

import argparse
import bz2
import csv
import html
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from backend import curriculum, text  # noqa: E402
from backend.config import DATA_DIR  # noqa: E402

WIKI_DUMP = "https://dumps.wikimedia.org/mtwiki/latest/mtwiki-latest-pages-articles.xml.bz2"
UA = {"user-agent": "speak-maltese/1.0 (personal language-learning tool)"}

SURFACE: dict[str, Counter] = {}
# How often each token appeared capitalised, to spot proper nouns. Tatoeba's Maltese
# sentences use "Ziri" as their stand-in person the way the English ones use "Tom" —
# 328 occurrences in 646 sentences, which lands it at rank 10 of a vocabulary list
# unless names are excluded.
CASED: Counter = Counter()
TOTALS: Counter = Counter()

CACHE = DATA_DIR / ".corpus_cache"
FREQ_OUT = DATA_DIR / "frequency_mt.tsv"
RETIER_OUT = DATA_DIR / "retier_proposal.tsv"

# Maltese letters only; digits, punctuation and Latin-script noise are dropped.
WORD = re.compile(r"[a-zàèìòùċġħżA-ZÀÈÌÒÙĊĠĦŻ']+")
# MediaWiki markup, templates, refs, tables — strip before counting.
MARKUP = [
    (re.compile(r"<ref[^>]*>.*?</ref>", re.S | re.I), " "),
    (re.compile(r"<[^>]+>"), " "),
    (re.compile(r"\{\{.*?\}\}", re.S), " "),
    (re.compile(r"\{\|.*?\|\}", re.S), " "),
    (re.compile(r"\[\[(?:File|Image|Stampa):[^\]]*\]\]", re.I), " "),
    (re.compile(r"\[\[[^\|\]]*\|"), " "),
    (re.compile(r"[\[\]']{2,}"), " "),
    (re.compile(r"https?://\S+"), " "),
    (re.compile(r"==+[^=]*==+"), " "),
    (re.compile(r"\[\[(?:Kategorija|Category|en|fr|it|de):[^\]]*\]\]", re.I), " "),
]

# MediaWiki furniture and stray English that would otherwise rank in the top 50.
STOP_NOISE = {"kategorija", "category", "the", "of", "and", "in", "to", "is",
              "redirect", "ref", "thumb", "px", "file", "image", "stampa"}


def fetch(url: str, name: str) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / name
    if cached.exists():
        return cached.read_bytes()
    print(f"  ↓ {url}")
    with httpx.stream("GET", url, timeout=900, follow_redirects=True, headers=UA) as r:
        r.raise_for_status()
        buf = bytearray()
        for chunk in r.iter_bytes(1 << 20):
            buf += chunk
            print(f"\r    {len(buf)/1e6:6.1f} MB", end="", flush=True)
    print()
    cached.write_bytes(bytes(buf))
    return bytes(buf)


def _bump(counts: Counter, surface: dict, word: str) -> None:
    """Group by folded key, but keep the commonest surface spelling for display.

    Folding is a matching device, not an orthography: `tiegħu` folds to `tieu` and
    `għall` to `all`, which is meaningless in a word list and collides with English.
    """
    f = text.fold(word)
    if 2 <= len(f) <= 24:
        counts[f] += 1
        surface.setdefault(f, Counter())[word.lower()] += 1
        TOTALS[f] += 1
        if word[:1].isupper():
            CASED[f] += 1


def wikipedia_counts() -> tuple[Counter, int]:
    raw = fetch(WIKI_DUMP, "mtwiki.xml.bz2")
    print("  decompressing and counting…")
    xml = bz2.decompress(raw).decode("utf-8", errors="replace")
    counts: Counter = Counter()
    SURFACE.clear()
    pages = 0
    for m in re.finditer(r"<text[^>]*>(.*?)</text>", xml, re.S):
        body = html.unescape(m.group(1))
        if body.strip().lower().startswith("#redirect"):
            continue
        pages += 1
        for pat, sub in MARKUP:
            body = pat.sub(sub, body)
        for w in WORD.findall(body):
            _bump(counts, SURFACE, w)
    return counts, pages


def korpus_counts() -> tuple[Counter, int]:
    """Korpus Malti, if the account has been granted access."""
    import json
    import pathlib

    tok_file = pathlib.Path.home() / ".cache/huggingface/token"
    token = tok_file.read_text().strip() if tok_file.exists() else ""
    if not token:
        sys.exit("No Hugging Face token found; run `huggingface-cli login` first.")

    base = "https://huggingface.co/api/datasets/MLRS/korpus_malti"
    hdr = {**UA, "authorization": f"Bearer {token}"}
    meta = httpx.get(base, params={"full": "true"}, timeout=120,
                     follow_redirects=True, headers=hdr).json()
    files = [s["rfilename"] for s in meta.get("siblings", [])
             if s["rfilename"].endswith(".jsonl")]
    # Conversational and general genres only. Legal and parliamentary Maltese would
    # dominate the counts and rank `regolament` above `ħobż`.
    wanted = ("blogs/", "comics/", "web_general/", "speeches/", "press_mt/", "wiki/")
    files = [f for f in files if any(w in f for w in wanted)]
    print(f"  {len(files)} files across conversational genres")

    counts: Counter = Counter()
    docs = 0
    for i, f in enumerate(files, 1):
        url = f"https://huggingface.co/datasets/MLRS/korpus_malti/resolve/main/{f}"
        r = httpx.get(url, timeout=300, follow_redirects=True, headers=hdr)
        if r.status_code != 200:
            sys.exit(f"\n  {r.status_code} on {f} — accept the dataset terms at "
                     "https://huggingface.co/datasets/MLRS/korpus_malti")
        for line in r.text.splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            body = obj.get("text") or ""
            if isinstance(body, list):
                body = " ".join(body)
            docs += 1
            for w in WORD.findall(body):
                _bump(counts, SURFACE, w)
        print(f"\r    {i}/{len(files)} files · {sum(counts.values())/1e6:.1f}M tokens",
              end="", flush=True)
    print()
    return counts, docs


def tatoeba_counts() -> Counter:
    """Everyday sentences, to offset Wikipedia's encyclopedic bias."""
    from importlib import util as iutil

    spec = iutil.spec_from_file_location(
        "import_corpus", Path(__file__).parent / "import_corpus.py")
    mod = iutil.module_from_spec(spec)
    spec.loader.exec_module(mod)
    counts: Counter = Counter()
    for mt, _en, _sid in mod.load_pairs():
        for w in WORD.findall(mt):
            _bump(counts, SURFACE, w)
    return counts


def _cached_counts(name: str, build):
    """Counting 700 remote files takes a quarter of an hour; do it once."""
    import json

    cache = CACHE / f"counts_{name}.json"
    if cache.exists():
        blob = json.loads(cache.read_text(encoding="utf-8"))
        SURFACE.update({k: Counter(v) for k, v in blob["surface"].items()})
        CASED.update(blob["cased"])
        TOTALS.update(blob["totals"])
        print(f"  (reusing cached counts from {cache.name})")
        return Counter(blob["counts"]), blob["docs"]
    counts, docs = build()
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "counts": counts, "docs": docs, "cased": dict(CASED), "totals": dict(TOTALS),
        "surface": {k: dict(v) for k, v in SURFACE.items()},
    }), encoding="utf-8")
    return counts, docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--korpus", action="store_true",
                    help="use MLRS Korpus Malti (gated; accept its terms first)")
    ap.add_argument("--retier", action="store_true",
                    help="also write a proposed retiering of the deck")
    ap.add_argument("--top", type=int, default=5000)
    args = ap.parse_args()

    if args.korpus:
        print("MLRS Korpus Malti — CC BY-NC-SA 4.0")
        counts, docs = _cached_counts("korpus", korpus_counts)
        source = "MLRS Korpus Malti (CC BY-NC-SA 4.0), conversational genres"
    else:
        print("Maltese Wikipedia — CC BY-SA")
        counts, docs = _cached_counts("wiki", wikipedia_counts)
        source = "Maltese Wikipedia (CC BY-SA)"

    print(f"  {docs} documents · {sum(counts.values())/1e6:.2f}M tokens · "
          f"{len(counts)} word types")

    # Blend in conversational data. Wikipedia never says "please"; a learner must.
    tat = tatoeba_counts()
    if tat:
        scale = max(1, sum(counts.values()) // max(1, sum(tat.values())) // 20)
        for w, c in tat.items():
            counts[w] += c * scale
        print(f"  blended Tatoeba ({len(tat)} types, weight ×{scale}) for everyday register")
        source += " + Tatoeba (CC BY 2.0 FR)"

    for w in STOP_NOISE:
        counts.pop(w, None)

    # Drop proper nouns. A word that is almost always capitalised is a name or a
    # place, and a learner's frequency list should rank vocabulary, not people.
    names = [w for w, total in TOTALS.items()
             if total >= 15 and CASED[w] / total >= 0.75 and w in counts]
    for w in names:
        counts.pop(w, None)
    print(f"  dropped {len(names)} proper nouns "
          f"(e.g. {', '.join(sorted(names, key=lambda x: -TOTALS[x])[:6])})")

    ranked = counts.most_common(args.top)

    def display(key: str) -> str:
        forms = SURFACE.get(key)
        return forms.most_common(1)[0][0] if forms else key
    FREQ_OUT.write_text(
        f"# Maltese frequency list derived from {source}.\n"
        f"# Ranks are of folded forms (diacritics normalised), so `ġo` and `go` are one.\n"
        "rank\tword\tcount\n"
        + "".join(f"{i}\t{display(w)}\t{c}\n" for i, (w, c) in enumerate(ranked, 1)),
        encoding="utf-8")
    print(f"\n✓ {FREQ_OUT.relative_to(DATA_DIR.parent)} — top {len(ranked)}")
    print("  top 25: " + " ".join(display(w) for w, _ in ranked[:25]))

    if not args.retier:
        return 0

    rank = {w: i for i, (w, _c) in enumerate(ranked, 1)}
    rows = curriculum._read_tsv(curriculum.VOCAB_TSV)
    proposals = []
    for r in rows:
        head = text.fold(re.split(r"[\s\-']+", r["mt"])[0])
        got = rank.get(head)
        if got is None:
            new = 4
        elif got <= 300:
            new = 1
        elif got <= 1200:
            new = 2
        elif got <= 3000:
            new = 3
        else:
            new = 4
        if new != int(r.get("tier") or 3):
            proposals.append({"id": r["id"], "mt": r["mt"], "en": r["en"],
                              "tier_now": r.get("tier"), "tier_proposed": new,
                              "corpus_rank": got or ""})
    with RETIER_OUT.open("w", encoding="utf-8", newline="") as fh:
        fh.write("# PROPOSAL ONLY — frequency is not the same as usefulness.\n")
        fh.write("# `jekk jogħġbok` is rare in print and essential in a café; keep it tier 1.\n")
        w = csv.DictWriter(fh, delimiter="\t",
                           fieldnames=["id", "mt", "en", "tier_now",
                                       "tier_proposed", "corpus_rank"])
        w.writeheader()
        w.writerows(proposals)
    moved_up = sum(1 for p in proposals if p["tier_proposed"] < int(p["tier_now"] or 3))
    print(f"✓ {RETIER_OUT.relative_to(DATA_DIR.parent)} — {len(proposals)} of "
          f"{len(rows)} words would move ({moved_up} up, {len(proposals)-moved_up} down)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

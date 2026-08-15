#!/usr/bin/env python3
"""Score a candidate tutor model on the Maltese it actually has to produce.

Maltese is genuinely low-resource. Most models — including ones whose model cards
list Maltese — will happily generate confident, wrong Maltese, which is worse than
no tutor at all. This probes a backend with tasks lifted from the real tutor loop and
grades them deterministically, so "can this model do it?" is a measurement.

    # whatever is configured in .env
    python scripts/check_tutor_model.py

    # a specific local model via Ollama / LM Studio
    python scripts/check_tutor_model.py --base-url http://localhost:11434/v1 \
                                        --model hf.co/bartowski/EuroLLM-9B-Instruct-GGUF:Q4_K_M

    python scripts/check_tutor_model.py --model gemma3:12b --base-url http://localhost:11434/v1

Exit code 0 if the model reaches the pass mark, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from backend import text  # noqa: E402

PASS_MARK = 0.70

def _maltese_markers() -> set[str]:
    """Vocabulary that signals "this is actually Maltese", derived from the app's own
    decks rather than a hand-kept list — so it stays in sync as the decks grow.

    Words that are also common English or Italian (`u`, `le`, `ma`, `no`) are dropped,
    since they would fire on a model answering in the wrong language.
    """
    from backend import curriculum

    ambiguous = {"u", "le", "ma", "no", "si", "e", "a", "in", "me", "te", "la", "il"}
    markers: set[str] = set()
    for row in curriculum._read_tsv(curriculum.VOCAB_TSV) + curriculum._read_tsv(curriculum.PHRASES_TSV):
        for word in re.split(r"[\s\-]+", row["mt"].lower()):
            word = word.strip(".,!?;:'’\"")
            if len(word) >= 2 and word not in ambiguous:
                markers.add(word)
    # fused preposition+article forms, which are strong Maltese signals
    markers |= {"mill", "fil", "tal", "bil", "sal", "mal", "għall", "lill", "fl", "bl"}
    return markers


MALTESE_MARKERS = _maltese_markers()


class Task:
    def __init__(self, name, prompt, check, weight=1.0):
        self.name, self.prompt, self.check, self.weight = name, prompt, check, weight


def _has(out: str, *variants: str) -> bool:
    """True if any variant appears, comparing with Maltese folding so a missing
    diacritic doesn't count against the model."""
    folded = text.fold(out)
    return any(text.fold(v) and text.fold(v) in folded for v in variants)


# ħ ġ ż ċ and the digraph għ are near-unique to Maltese orthography, so they are
# strong evidence on their own — important for short answers like "Jien għajjien."
# where a function-word count alone would give a false negative.
MALTESE_CHARS = re.compile(r"[ħĦġĠżŻċĊ]|għ", re.I)


def _is_maltese(s: str) -> bool:
    words = {w.strip(".,!?;:—-\"'").lower() for w in s.split()}
    markers = len(words & MALTESE_MARKERS)
    return markers >= 2 or (markers >= 1 and bool(MALTESE_CHARS.search(s)))


# ── The probes ─────────────────────────────────────────────────────────────

def check_fusion(out: str) -> tuple[float, str]:
    """minn + il- → mill- ; fi + il- → fil-. The single most common learner error,
    so a tutor that misses it is useless."""
    got_mill = _has(out, "mill-Awstralja")
    got_fil = _has(out, "fil-Belt")
    left_broken = _has(out, "minn l-Awstralja") or _has(out, "fi il-Belt")
    score = (0.5 * got_mill) + (0.5 * got_fil)
    if left_broken and score > 0:
        score *= 0.5  # claimed a fix while leaving an error in — actively misleading
    notes = []
    if not got_mill:
        notes.append("missed minn+il- → mill-")
    if not got_fil:
        notes.append("missed fi+il- → fil-")
    if left_broken:
        notes.append("left an uncorrected error in its 'corrected' sentence")
    return score, "; ".join(notes) or "both fusions fixed"


def check_counting(out: str) -> tuple[float, str]:
    """Numbers 2-10 take a short form before a noun: tliet itfal, not *tlieta tfal*."""
    if _has(out, "tliet itfal", "tlett itfal", "tlitt itfal"):
        return 1.0, "correct counting form"
    if _has(out, "tliet tfal", "tlett tfal"):
        return 0.6, "short form right, plural spelling off (tliet itfal)"
    if _has(out, "tlieta tfal"):
        return 0.0, "left *tlieta tfal* uncorrected"
    return 0.2, "no recognisable counting correction"


def check_no_copula(out: str) -> tuple[float, str]:
    """Maltese has no present-tense 'to be'. *Jien inkun għajjien* is a classic
    English-speaker error a tutor must not reproduce."""
    bad = re.search(r"\b(inkun|jien inkun|huwa hu)\b", out, re.I)
    if bad:
        return 0.0, f"inserted a present-tense copula: {bad.group(0)!r}"
    if _is_maltese(out):
        return 1.0, "no spurious copula"
    return 0.3, "answer does not look like Maltese"


def check_language(out: str) -> tuple[float, str]:
    return (1.0, "replied in Maltese") if _is_maltese(out) else (0.0, "did not reply in Maltese")


def check_json(out: str) -> tuple[float, str]:
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return 0.0, "no JSON object in output"
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return 0.2, f"malformed JSON: {exc}"
    missing = [k for k in ("reply_mt", "reply_en", "correction_mt") if not data.get(k)]
    if missing:
        return 0.4, f"JSON parsed but empty/missing: {', '.join(missing)}"
    if not _is_maltese(data["reply_mt"]):
        return 0.5, "valid JSON but reply_mt is not Maltese"
    # The fields must also mean what they say — small models routinely fill the
    # right shape with the wrong content (correction in reply_mt, and so on).
    if not _has(data["correction_mt"], "tliet itfal", "tlett itfal", "tlitt itfal"):
        return 0.7, "valid JSON, Maltese reply, but correction_mt holds the wrong thing"
    if "?" not in data["reply_mt"]:
        return 0.85, "valid JSON, correct fields, but reply_mt is not a question"
    return 1.0, "valid JSON, correct field roles"


TASKS = [
    Task("article fusion",
         "You are a Maltese teacher. A learner wrote: "
         "'Jien minn l-Awstralja u jien noqghod fi il-Belt.' "
         "Rewrite it correctly in Maltese. Output the corrected Maltese sentence first.",
         check_fusion, weight=2.0),
    Task("counting form",
         "You are a Maltese teacher. A learner wrote: 'Jien ghandi tlieta tfal.' "
         "Rewrite it correctly in Maltese. Output the corrected Maltese sentence first.",
         check_counting, weight=1.5),
    Task("no copula",
         "Translate into natural Maltese, output the Maltese only: 'I am tired today.'",
         check_no_copula, weight=1.0),
    Task("stays in Maltese",
         "Reply in Maltese only, one short sentence ending in a question. "
         "The learner just said: 'Jien mill-Awstralja.'",
         check_language, weight=1.0),
    Task("structured output",
         'Return ONLY a JSON object with keys reply_mt (one Maltese sentence ending '
         'in a question), reply_en (its English translation), and correction_mt (the '
         'corrected Maltese for the learner sentence). No prose, no markdown. '
         'Learner said: "Jien ghandi tlieta tfal."',
         check_json, weight=1.5),
]


# ── Backends ───────────────────────────────────────────────────────────────

def _messages(prompt: str, grounding: str) -> list[dict]:
    msgs = []
    if grounding:
        msgs.append({"role": "system", "content": grounding})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def call_openai(base_url: str, model: str, key: str, prompt: str, grounding: str = "") -> str:
    headers = {"content-type": "application/json"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    r = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers, timeout=600,
        json={
            "model": model,
            "messages": _messages(prompt, grounding),
            "temperature": 0.2,
            # Reasoning models spend the budget thinking before they answer, so a
            # tight cap leaves `content` empty and looks like total failure.
            "max_tokens": 1600,
            # Best-effort "stop thinking" across runtimes; not all honour it.
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    if content:
        return content
    # Ollama splits reasoning models into `reasoning` + `content`. If the model
    # never stopped thinking, grade what it did produce rather than scoring a zero
    # that measures our token cap instead of its Maltese.
    return (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()


def call_anthropic(model: str, key: str, prompt: str, grounding: str = "") -> str:
    body = {"model": model, "max_tokens": 600,
            "messages": [{"role": "user", "content": prompt}]}
    if grounding:
        body["system"] = grounding
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        timeout=180, json=body,
    )
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"]).strip()


def main() -> int:
    from backend.config import CFG

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=CFG.openai_base)
    ap.add_argument("--model", default="")
    ap.add_argument("--api-key", default=CFG.openai_key)
    ap.add_argument("--verbose", "-v", action="store_true", help="print raw output")
    ap.add_argument("--trials", type=int, default=3,
                    help="repeat the probe; low-resource output is high-variance")
    ap.add_argument("--bare", action="store_true",
                    help="probe without the grammar reference the real tutor injects")
    args = ap.parse_args()

    # The live tutor puts data/grammar_notes.md in its system prompt on every turn,
    # so a fair test of a model *as deployed* includes it. Weak models lean on it
    # heavily; without it this measures raw Maltese knowledge instead.
    grounding = ""
    if not args.bare:
        from backend import curriculum
        grounding = (
            "You are a Maltese teacher. Follow this reference exactly; it is "
            "authoritative and overrides your own recollection.\n\n"
            + curriculum.grammar_notes()
        )

    if args.base_url:
        model = args.model or CFG.openai_model
        backend = lambda p: call_openai(args.base_url, model, args.api_key, p, grounding)  # noqa: E731
        label = f"{model} @ {args.base_url}"
    elif CFG.anthropic_key:
        model = args.model or CFG.tutor_model
        backend = lambda p: call_anthropic(model, CFG.anthropic_key, p, grounding)  # noqa: E731
        label = f"{model} @ anthropic"
    else:
        print("No backend. Pass --base-url, or set keys in .env.", file=sys.stderr)
        return 2

    mode = "bare (no reference)" if args.bare else "grounded (as deployed)"
    print(f"Probing {label}  —  {mode}, {args.trials} trial(s)\n" + "─" * 66)

    per_task: dict[str, list[float]] = {t.name: [] for t in TASKS}
    run_totals: list[float] = []

    for trial in range(args.trials):
        total = earned = 0.0
        for task in TASKS:
            total += task.weight
            t0 = time.time()
            try:
                out = backend(task.prompt)
            except Exception as exc:  # noqa: BLE001
                print(f"  ✗ {task.name:20} FAILED: {exc}")
                per_task[task.name].append(0.0)
                continue
            score, note = task.check(out)
            earned += score * task.weight
            per_task[task.name].append(score)
            if args.trials == 1 or args.verbose:
                mark = "✓" if score >= 0.75 else "~" if score >= 0.4 else "✗"
                prefix = f"  {mark} {task.name:20}" if args.trials == 1 else \
                         f"  [{trial+1}] {mark} {task.name:20}"
                print(f"{prefix} {score:4.0%}  {note}   [{time.time()-t0:.0f}s]")
            if args.verbose:
                print("      " + out.replace("\n", "\n      ")[:600] + "\n")
        run_totals.append(earned / total if total else 0.0)
        if args.trials > 1 and not args.verbose:
            # Reasoning models can take minutes per trial; don't look hung.
            print(f"  trial {trial+1}/{args.trials}: {run_totals[-1]:.0%}", flush=True)

    print("─" * 66)
    if args.trials > 1:
        # Consistency matters as much as the mean: a tutor that fixes an error only
        # half the time still teaches wrong Maltese half the time.
        for task in TASKS:
            scores = per_task[task.name]
            hit = sum(1 for s in scores if s >= 0.75)
            print(f"  {task.name:20} mean {sum(scores)/len(scores):4.0%}   "
                  f"passed {hit}/{len(scores)} trials")
        print("─" * 66)
    pct = sum(run_totals) / len(run_totals)
    spread = f"  (runs: {', '.join(f'{r:.0%}' for r in run_totals)})" if args.trials > 1 else ""
    verdict = ("USABLE as a tutor" if pct >= PASS_MARK
               else "NOT usable — it will teach wrong Maltese")
    print(f"  Mean score {pct:.0%}{spread}   →  {verdict}")
    return 0 if pct >= PASS_MARK else 1


if __name__ == "__main__":
    raise SystemExit(main())

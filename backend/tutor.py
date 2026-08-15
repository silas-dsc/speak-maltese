"""The conversational tutor.

One model call per learner turn returns a single JSON object holding both the
correction and the reply, so the UI can show a gentle recast *and* keep the
conversation moving without a second round trip.

Corrective-feedback design (this is the bit that matters pedagogically):

* **Recast, not rejection** — the tutor always replies to the *content* first.
* **One focus at a time** — at most two issues surfaced per turn, ranked by whether
  they impede understanding. Everything else is silently absorbed.
* **Prompted repetition** — the learner is asked to say the corrected form aloud,
  which is what actually moves it into production.
* **Errors become cards** — each correction is logged and can be scheduled.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from .config import CFG
from . import curriculum, db, text

RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["correction", "reply_mt", "reply_en"],
    "properties": {
        "correction": {
            "type": "object",
            "required": ["needed"],
            "properties": {
                "needed": {"type": "boolean"},
                "corrected_mt": {"type": "string"},
                "severity": {"type": "string", "enum": ["minor", "moderate", "major"]},
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": ["grammar", "vocab", "spelling", "word-order",
                                         "pronunciation", "register"],
                            },
                            "said": {"type": "string"},
                            "should_be": {"type": "string"},
                            "why": {"type": "string"},
                        },
                    },
                },
                "repeat_prompt_mt": {"type": "string"},
                "repeat_prompt_en": {"type": "string"},
            },
        },
        "praise": {"type": "string"},
        "reply_mt": {"type": "string"},
        "reply_en": {"type": "string"},
        "gloss": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"mt": {"type": "string"}, "en": {"type": "string"}},
            },
        },
        "new_vocab": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mt": {"type": "string"}, "en": {"type": "string"},
                    "pos": {"type": "string"}, "note": {"type": "string"},
                },
            },
        },
        "difficulty_signal": {"type": "string", "enum": ["too_easy", "ok", "too_hard"]},
        "tip": {"type": "string"},
    },
}

SYSTEM = """You are a warm, patient Maltese conversation tutor. The learner is an \
English speaker learning Maltese (Malti) by talking with you.

# Absolute rules
1. `reply_mt` is ALWAYS in Maltese. `reply_en` is its English translation. Never mix.
2. Keep `reply_mt` to 1-2 short sentences and ALWAYS end with a question, so the \
learner has something to answer. Conversation must not stall.
3. Stay inside the learner's known vocabulary plus at most TWO new items per turn \
(comprehensible input, i+1). Their known pool is given below — lean on it heavily.
4. Correct gently. Reply to what they *meant* first; treat the correction as a \
friendly aside, never a rebuke. Never correct more than two issues in one turn, and \
only ones that matter. Ignore typos in accented letters (ġ ħ ż ċ għ) if the word is \
otherwise right — speech recognition drops them constantly.
5. If they wrote in English, or code-switched, that is fine: give them the Maltese \
they were reaching for in `correction.corrected_mt`, and set `needed` to true with a \
`vocab` issue.
6. `correction.repeat_prompt_mt` must be a short, natural, fully correct Maltese \
sentence they can say aloud in one breath, with its English in \
`correction.repeat_prompt_en`. Set both whenever `needed` is true — the pair becomes \
a flashcard, so the English must stand on its own as a prompt.
7. `gloss` breaks `reply_mt` into word-by-word English so they can decode it.
8. Use real, idiomatic Maltese as spoken in Malta today. Common English/Italian \
loanwords (`mela`, `ċaw`, `grazzi`, `orrajt`, `skużi`) are correct Maltese — do not \
"correct" them into archaisms.
9. Set `difficulty_signal` honestly so the app can adapt.

# Output
Return ONE JSON object matching the schema. No markdown, no prose outside the JSON."""


def _reference_block() -> str:
    notes = curriculum.grammar_notes()
    # trim to the parts that most affect correction quality
    return notes


def build_prompt(user_text: str, scenario: dict | None, profile: dict,
                 targets: list[dict], history: list[dict]) -> tuple[str, list[dict]]:
    scenario_block = ""
    if scenario:
        scenario_block = (
            f"\n# Scene\nYou are playing: {scenario['tutor_role']}\n"
            f"Setting: {scenario['name_en']} ({scenario['name']})\n"
            f"Learner's goal: {scenario['goal_en']}\n"
        )

    target_block = ""
    if targets:
        listed = "; ".join(f"{t['mt']} ({t['en']})" for t in targets)
        target_block = (
            "\n# Try to elicit\nSteer the conversation so the learner naturally needs "
            f"these items, without naming them: {listed}\n"
        )

    err_block = ""
    if profile["recent_errors"]:
        lines = "\n".join(
            f"- said \"{e['said']}\" → \"{e['correct']}\" ({e['kind']}: {e['why']})"
            for e in profile["recent_errors"][:6]
        )
        err_block = f"\n# Their recent mistakes (watch for repeats)\n{lines}\n"

    known = ", ".join(profile["known_words"][:150]) or "(nothing yet — start from zero)"
    phrases = "; ".join(profile["known_phrases"][:40]) or "(none yet)"

    system = (
        f"{SYSTEM}\n"
        f"{scenario_block}"
        f"\n# Learner\nCEFR level: {profile['level']} · {profile['learned_count']} items learned\n"
        f"Known words: {known}\n"
        f"Known phrases: {phrases}\n"
        f"{target_block}"
        f"{err_block}"
        f"\n# Maltese reference (authoritative — follow it)\n{_reference_block()}\n"
    )

    messages: list[dict] = []
    for turn in history[-12:]:
        if turn["role"] == "user":
            messages.append({"role": "user", "content": turn["mt"] or turn["en"] or ""})
        else:
            messages.append({"role": "assistant", "content": turn["mt"] or ""})
    messages.append({"role": "user", "content": user_text})
    return system, messages


class TutorUnavailable(RuntimeError):
    pass


async def respond(user_text: str, session_id: str, scenario_id: str | None) -> dict:
    scenarios = {s["id"]: s for s in curriculum.load_scenarios()}
    scenario = scenarios.get(scenario_id or "")
    profile = curriculum.learner_profile()
    targets = curriculum.target_items(scenario_id)
    hist = db.history(session_id, 24)

    system, messages = build_prompt(user_text, scenario, profile, targets, hist)

    if CFG.anthropic_key:
        raw = await _call_anthropic(system, messages)
    elif CFG.openai_key or CFG.openai_base:
        raw = await _call_openai(system, messages)
    else:
        raise TutorUnavailable(
            "No LLM configured. Set ANTHROPIC_API_KEY (or OPENAI_API_KEY / "
            "SM_OPENAI_BASE_URL for a local model) in .env."
        )

    data = _coerce(raw)
    _lint(user_text, data)
    _persist(session_id, scenario_id, user_text, data)
    return data


def _lint(user_text: str, data: dict) -> None:
    """Rule-based safety net over the model's Maltese.

    Preposition + article fusion (`minn il-` → `mill-`) is the most common error an
    English speaker makes, and it is fully mechanical — so it does not need a model.
    Smaller local models miss it often (EuroLLM-9B catches it about one time in
    four), and a "corrected" sentence that still contains the error is worse than no
    correction. So we check the tutor's own output, repair it, and — if the learner
    made the mistake and the model said nothing — raise the correction ourselves.
    """
    corr = data["correction"]

    # 1. Never ship a corrected form or a reply that still contains the error.
    for field in ("corrected_mt", "repeat_prompt_mt"):
        if corr.get(field):
            corr[field] = text.apply_fusion(corr[field])
    for field in ("reply_mt",):
        if data.get(field):
            data[field] = text.apply_fusion(data[field])

    # 2. If the learner made the error and the model let it pass, correct it.
    missed = text.lint_fusion(user_text)
    if not missed:
        return
    already = {i.get("should_be", "").lower() for i in (corr.get("issues") or [])}
    new_issues = [
        {"kind": "grammar", "said": m["found"], "should_be": m["should_be"],
         "why": m["why"]}
        for m in missed if m["should_be"].lower() not in already
    ]
    if not new_issues:
        return
    corr["needed"] = True
    corr["issues"] = (corr.get("issues") or []) + new_issues
    corr["issues"] = corr["issues"][:3]
    fixed = text.apply_fusion(user_text)
    corr.setdefault("severity", "minor")
    if not corr.get("corrected_mt"):
        corr["corrected_mt"] = fixed
    if not corr.get("repeat_prompt_mt"):
        corr["repeat_prompt_mt"] = fixed


async def _call_anthropic(system: str, messages: list[dict]) -> str:
    payload = {
        "model": CFG.tutor_model,
        "max_tokens": 1400,
        "system": system,
        "messages": messages,
        "tools": [{
            "name": "tutor_turn",
            "description": "Return the tutor's correction and reply.",
            "input_schema": RESPONSE_SCHEMA,
        }],
        "tool_choice": {"type": "tool", "name": "tutor_turn"},
    }
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": CFG.anthropic_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
        )
        r.raise_for_status()
        body = r.json()
    for block in body.get("content", []):
        if block.get("type") == "tool_use":
            return json.dumps(block["input"])
    return "".join(b.get("text", "") for b in body.get("content", []))


async def _call_openai(system: str, messages: list[dict]) -> str:
    base = (CFG.openai_base or "https://api.openai.com/v1").rstrip("/")
    payload = {
        "model": CFG.openai_model,
        "messages": [{"role": "system", "content": system}] + messages,
        "response_format": {"type": "json_object"},
        "max_tokens": 1400,
    }
    headers = {"content-type": "application/json"}
    if CFG.openai_key:
        headers["authorization"] = f"Bearer {CFG.openai_key}"
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(f"{base}/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        body = r.json()
    return body["choices"][0]["message"]["content"]


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def _coerce(raw: str) -> dict:
    """Parse the model output into the expected shape, tolerating stray prose."""
    data: Any = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(raw or "")
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict):
        data = {"reply_mt": (raw or "").strip(), "reply_en": "", "correction": {"needed": False}}

    corr = data.get("correction") or {}
    if not isinstance(corr, dict):
        corr = {"needed": False}
    corr.setdefault("needed", False)
    corr.setdefault("issues", [])
    corr.setdefault("severity", "minor")
    data["correction"] = corr
    data.setdefault("reply_mt", "")
    data.setdefault("reply_en", "")
    data.setdefault("gloss", [])
    data.setdefault("new_vocab", [])
    data.setdefault("difficulty_signal", "ok")
    return data


def _persist(session_id: str, scenario_id: str | None, user_text: str, data: dict) -> None:
    db.add_turn(session_id, scenario_id, "user", user_text, None)

    corr = data["correction"]
    if corr.get("needed"):
        for issue in (corr.get("issues") or [])[:3]:
            db.log_error(
                kind=issue.get("kind", "grammar"),
                learner=issue.get("said") or user_text,
                corrected=issue.get("should_be") or corr.get("corrected_mt", ""),
                why=issue.get("why", ""),
            )

    if data.get("new_vocab"):
        data["new_vocab_ids"] = curriculum.register_new_vocab(data["new_vocab"], scenario_id)

    db.add_turn(session_id, scenario_id, "tutor", data.get("reply_mt"),
                data.get("reply_en"), data)

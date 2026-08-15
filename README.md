# Nitkellmu — Speak Maltese

A conversational Maltese tutor that listens, speaks, shows you the writing, corrects
you gently, and schedules everything you meet with spaced repetition.

**There is no LLM.** Conversations are scripted and pre-voiced, and answers are graded
by phonetic match, so a turn comes back in about a hundredth of a second instead of
ten seconds. See [why](#why-no-llm).

```bash
./run.sh
```

Then open <http://127.0.0.1:8137>. First run creates `.venv` and `.env`. **No API keys
are needed for anything** — speech in, speech out and conversation all run locally.

---

## What it does

**Talk (`Taħdita`)** — Pick one of **23 scenes** (introductions, the café, the
market, the doctor, the phone, getting unstuck…) and work through it out loud. Hold the mic or type. Each line comes back as Maltese audio
plus the written Maltese and a translation you can hide.

Every reply is authored and pre-voiced, and what you say is matched against a list of
accepted answers *phonetically* — so `nixtiek kafe jek jogobok` is accepted for
`Nixtieq kafè, jekk jogħġbok.` A right answer moves the conversation on; a near miss
or a wrong one shows you the target and asks for it again, and phrases you get right
are scheduled into the review deck.

Each scene has its own illustration, and finishing one ends with a summary: how many
you got first try, how many took a retry, seconds per turn, and the phrases that went
into your review deck. Completed scenes are ticked in the picker.

Nobody gets stuck: a personal answer (your name, your town) is never graded, and after
two attempts on any line the conversation moves on regardless.

Measured end to end, speech in to spoken Maltese reply out: **0.3s**, of which the
matching is about 10ms.

**Review (`Reviżjoni`)** — FSRS-5 spaced repetition over the whole deck, in three
retrieval directions that rotate per card:

| mode | front | what it trains |
|---|---|---|
| `listen` | Maltese + audio | form–sound mapping, first exposure |
| `recognise` | Maltese | comprehension |
| `produce` | English | **speaking** — the hard, transferable direction |

In `produce` mode you say the answer. It is transcribed, scored against the target
with a Maltese-aware comparison, diffed word by word, and the right grade is
pre-selected for you.

**Progress** — words learned, how many are solid (3 weeks+), day streak, spoken
accuracy, per-topic coverage, your stickiest words, and which *kind* of mistake you
make most.

**Guide (`Gwida`)** — A compact reference: the sounds English speakers get wrong
(`q`, `għ`, `ħ`, `x`, `ż`), article assimilation, the `ma … x` negation frame, verb
prefixes, `għandi`, broken plurals, counting forms.

Keyboard (Review): `space` reveals then grades Good, `1`–`4` grade directly, `r` replays.

---

## The learning design

Each of these is a deliberate choice, not a default:

- **Frequency-first, but curated.** Tier 1 vocabulary is the ~100 items that buy the
  most coverage. See [the note on the source list](#about-the-2000-word-list).
- **Chunks before words.** Phrases interleave with single words in the new-card
  queue, because fluent speech is largely prefabricated sequences. You always leave a
  session with something *sayable*.
- **Comprehensible input at i+1.** Dialogues are levelled, and each scene reuses what
  the earlier ones taught plus a little that is new.
- **Production over recognition.** Recognition and production are tracked separately
  per card, and the queue biases toward whichever is lagging.
- **Interleaving, not blocking.** New cards are spread through the review queue
  rather than front-loaded, and topics are mixed.
- **Prompted repetition.** A wrong answer is not just marked wrong: the correct form
  is shown and spoken, and you say it back before moving on.
- **Errors become cards.** A phrase you had to work for is exactly the item worth
  scheduling, so correct answers go straight into the deck.
- **Spacing to a retention target.** FSRS-5 schedules to 90% recall (configurable)
  instead of a fixed multiplier ladder.

---

## Speech

Maltese is a low-resource language for speech, so this matters more than usual.

**Out (TTS).** The chain is `azure → edge → elevenlabs`, first available wins,
results cached on disk.

- **Azure** has the only genuine `mt-MT` neural voices — `mt-MT-GraceNeural` and
  `mt-MT-JosephNeural`. Set `AZURE_SPEECH_KEY` for the official, supported path.
- **edge** (`edge-tts`) reaches those *same two voices* through Edge's Read-Aloud
  endpoint with **no key**, which is what makes this app work on a fresh clone. It is
  an unofficial endpoint — fine for personal study, not for anything commercial, and
  it can break without notice.
- **mms** — Meta's `facebook/mms-tts-mlt`, fully offline. Robotic but phonetically
  sound. Opt in with `SM_TTS_PROVIDER=mms` after installing `torch` + `transformers`.

Deliberately **not** used: the browser's `speechSynthesis` (no OS ships a Maltese
voice, and an Italian voice reading Maltese teaches wrong pronunciation) and gTTS
(Google Translate has no Maltese audio — it returns *"Unsupported language 'mt'"*).

**In (STT).** The chain is `wav2vec2 → openai_whisper → elevenlabs → faster_whisper → azure`.
Browser `SpeechRecognition` is unusable here for the same reason, so audio is recorded
in the page and posted to the backend. `faster-whisper` runs locally with no key, so
speech input works out of the box; the first run downloads the model.

Because recognisers routinely drop Maltese diacritics and split the fused article,
grading folds `ġ ħ ż ċ`, treats the silent `għ` as absent, makes the article optional,
and splits at hyphens — so `jien mill Awstralja` scores 1.00 against
`Jien mill-Awstralja.` while a real word substitution still costs you.

---

## Why no LLM

An open-ended tutor was built first, and measured, and then removed. The measurements
are why.

Maltese is low-resource enough that most models produce *confident, wrong* Maltese —
which for something that corrects you is worse than nothing. Every model was graded on
five things the tutor actually had to do, grounded (with `grammar_notes.md` in context)
and over repeated trials, because low-resource output is high-variance:

| model | mean | article fusion | counting | stays in Maltese | JSON |
|---|---|---|---|---|---|
| EuroLLM-9B-Instruct Q4_K_M | 82% | 1/4 ⚠ | 4/4 | 4/4 | 4/4 |
| gemma3:12b | 58% | 0/3 | 0/3 | 3/3 | 2/3 |
| qwen3.5:9b | 53% | 0/3 | 0/3 | 3/3 | 0/3 |
| Maltese-EuroLLM-**1.7B** | unusable | — | — | — | — |
| qwen3:4b | unusable | — | — | — | — |

[EuroLLM](https://huggingface.co/utter-project/EuroLLM-9B-Instruct) was the best of
them — EU-funded, trained on all 24 official EU languages including Maltese — and it
still missed `minn` + `il-` → `mill-` three times in four, the commonest error an
English speaker makes. Size is not the constraint: gemma3:12b is larger and 24 points
worse. **A model card listing Maltese is not evidence.**

The smallest option was tried last, since a tiny model would at least have been fast:
`Maltese-EuroLLM-1.7B` is Maltese-specific *and* 1 GB, and it returns a bare newline —
it is a base continuation model, not chat-tuned. There is no small Maltese LLM.

And even the good one was slow: **8–25 seconds a turn** locally, against 0.3s for the
scripted path. For something meant to feel like conversation, that is the whole game.

So the trade was made deliberately: **scripted replies, phonetic grading.** Grading is
imperfect — a near-miss occasionally re-prompts when a person would have let it go —
but the Maltese you *hear* is authored and correct by construction, which is the right
side to be wrong on. What was lost is open-endedness: you cannot say anything you like
and be understood.

The rule-based safety net that was written to cover EuroLLM's fusion blind spot lives
on as `text.lint_fusion`, and now guards the authored dialogue instead: a test asserts
that no line in any deck or dialogue contains an unfused preposition.

## Maltese-specific models on Hugging Face

Filtering the Hub's `mt` tag is misleading — nearly everything under it is a *massively
multilingual* model that merely lists Maltese. The genuinely Maltese-trained models are
few, small, and mostly academic. The ones that matter here:

**Speech recognition** — the most useful category for this app.

| model | notes |
|---|---|
| [`carlosdanielhernandezmena/whisper-large-maltese-8k-steps-64h-ct2`](https://huggingface.co/carlosdanielhernandezmena/whisper-large-maltese-8k-steps-64h-ct2) | Whisper-large + 64h Maltese, already in **CTranslate2** format → drop-in for the `faster_whisper` backend. CC-BY-4.0. |
| [`sam8000/whisper-large-v3-turbo-maltese-malta`](https://huggingface.co/sam8000/whisper-large-v3-turbo-maltese-malta) | Newer base, but transformers format — needs `ct2-transformers-converter` first. MIT. |
| [`oddadmix/MasriSwitch-Gemma3n-Transcriber-v1`](https://huggingface.co/oddadmix/MasriSwitch-Gemma3n-Transcriber-v1) | Handles Maltese↔English code-switching, which is how Maltese is actually spoken. GGUF builds exist. |

**The default is not Whisper at all.** A Maltese **wav2vec2 CTC** model measures
identically and runs ~30× faster:

| model | fWER | app score | **would pass** | **s/clip** |
|---|---|---|---|---|
| **wav2vec2-large-xlsr-53-maltese-64h** (CTC, Metal) | **5.3%** | **0.98** | **96%** | **0.3** |
| whisper-large-maltese-…-ct2 (CPU) | 5.3% | 0.98 | 96% | 8.9 |

Same accuracy to the decimal, a thirtieth of the time. The reason is architectural,
not a matter of size — and it is the same fact that made the turbo Whisper pointless.
Whisper decodes autoregressively and pads every input to a fixed **30-second** window,
so a three-word answer costs what a monologue costs. A CTC model is a **single forward
pass over the audio you actually recorded**: no decoder loop, no padding. On Apple
Silicon it runs on the GPU through PyTorch's Metal backend.

It transcribes lowercase and unpunctuated (`bonġu`, `noqgħod ta' sliema`, `kollox
tajjeb ħafna grazzi`) with the diacritics intact, which is exactly what both the
phonetic matcher and the tutor want.

End to end, speech in to spoken Maltese reply out, in scripted mode: **0.30s**.

Set `SM_W2V_DEVICE=cpu` to force it off the GPU, or `SM_STT_PROVIDER=faster_whisper`
to go back to Whisper.

### Whisper comparison

**Measured**, with `scripts/compare_stt.py` on 25 deck sentences spoken by the app's
own `mt-MT` voice:

| model | size | WER | fWER | CER | app score | **would pass** | s/clip |
|---|---|---|---|---|---|---|---|
| **whisper-large-maltese-…-ct2** | 3.1 GB | 50.3% | **5.3%** | 8.0% | **0.98** | **96%** | 8.8 |
| whisper-large-v3-**turbo**-maltese (converted) | 787 MB | 30.8% | 21.3% | 8.7% | 0.93 | 84% | 8.0 |
| whisper `small` (generic) | 484 MB | 129% | 118% | 47.2% | 0.42 | **4%** | 3.5 |

**The turbo fine-tune was tried and rejected.** `sam8000/whisper-large-v3-turbo-maltese-malta`
ships in transformers format; `scripts/convert_turbo.sh` converts it to CTranslate2
(its repo has no `tokenizer.json`, so one is built from the `vocab.json`/`merges.txt`
pair). It is a quarter the size and **only 9% faster** — 8.0s against 8.8s — while
dropping 12 points of pass rate.

That 9% is the interesting part, and it explains why chasing a smaller model is a dead
end here. Turbo's savings are all in the *decoder* (4 layers instead of 32); the
encoder is unchanged, and Whisper always pads its input to a fixed 30-second window.
For a three-word answer the encoder is essentially the entire cost, so a lighter
decoder buys almost nothing. Beam size showed the same thing (beam 1 vs 5: 10.0s vs
10.6s). **On CPU, ~8s per utterance is the floor for any large-encoder Maltese model**,
and every Maltese fine-tune that exists is large-based.

So the options for speed are a cloud recogniser (`OPENAI_API_KEY`, and the chain
prefers it automatically) — or not waiting on the tutor at all, which is what
[scripted mode](#two-conversation-modes) is for.

Not a marginal gain: **4% → 96%** of utterances the app would mark correct. Generic
`small` does not merely mis-hear Maltese, it hallucinates — `Tinkwetax.` came back as
a different sentence entirely, and WER above 100% means it invented more words than
were said. On a low-resource language the fine-tune is the difference between working
and not.

Read `fWER`, not `WER`, for the fine-tune: it transcribes lowercase and unpunctuated,
which strict WER punishes even when every word is right.

This is now the **default** in `.env.example`. `faster-whisper` resolves the repo id
through the Hub, so the only setup is the ~3 GB first-run download.

The cost is real and worth stating plainly: **~9s to transcribe each thing you say**
on CPU, against ~3s for generic `small`. The model is loaded at startup in a
background thread, so the UI is up in about a second and your *first* utterance is no
slower than your tenth — but the per-utterance wait stays. Beam size does not help
(beam 1 and beam 5 both score 0.98; 10.0 vs 10.6 s/clip), because the cost is the
model, not the search.

If that pause bothers you, the fix is a *cloud* recogniser, not a smaller local one —
set `OPENAI_API_KEY` and the chain prefers Whisper API automatically. Dropping back to
`small` only makes it fast and wrong.

Reproduce, or test your own voice — which is what really matters, since the app has to
understand *you* rather than a synthesiser:

```bash
python scripts/compare_stt.py --record 20 --models small,carlosdanielhernandezmena/whisper-large-maltese-8k-steps-64h-ct2
```

**Language models** — [`MLRS/BERTu`](https://huggingface.co/MLRS/BERTu) is the
reference Maltese encoder (University of Malta), with POS/NER/sentiment heads
alongside it. It is a masked LM, so it *cannot* run the conversation — useful for
grammar tooling, not for chat. There are Maltese SFT/DPO fine-tunes of the very
EuroLLM this app recommends (`jjzha/EuroLLM-9B-Instruct-2512-*maltese*`), but they ship
as unmerged LoRA adapters with no stated licence, so they need merging and quantising
before Ollama can serve them. `st192011/Maltese-EuroLLM-1.7B-*` is Maltese-tuned and
has GGUF builds, but at 1.7B it is well below what the tutor needs.

**Text to speech** — thin. `MohamedGomaa30/spark-tts-normazlied-masri-mega` is a
Maltese SparkTTS, and Meta's `facebook/mms-tts-mlt` (already an option here via
`SM_TTS_PROVIDER=mms`) remains the most practical offline voice. Neither matches the
Azure `mt-MT` neural voices.

The picture overall: **Maltese ASR fine-tunes are worth adopting, Maltese LLMs are not
ready**, which is why the tutor still points at EuroLLM-9B and the app carries its own
rule-based `lint_fusion` safety net.

## About the 2000-word list

The requested source, [commonlyusedwords.com's 2000 most common Maltese
words](https://commonlyusedwords.com/2000-most-common-maltese-words/), is a **machine
translation of an English frequency list**, not a Maltese frequency list, and it has
real errors:

| # | Maltese given | Glossed as | Actually |
|---|---|---|---|
| 7 | `pulzieri` | "in" | *inches* |
| 62 | `sal-bott` | "to can" | the modal verb read as a tin can |
| 75 | `ġust` | "just" | a calque; Maltese uses `biss` / `eżatt` |
| 2 | `li tkun` | "to be" | a subordinate clause, not a headword |

Teaching from it directly would drill wrong Maltese, so the app ships a hand-curated
deck instead — **350 words** (`data/core_vocab.tsv`) and **120 phrases**
(`data/phrases.tsv`), frequency- and utility-ordered across tiers 1–3, with example
sentences, gender/plural notes and verb roots.

The original list is still available for coverage, quarantined:

```bash
python scripts/import_frequency_list.py
```

That fetches it, drops entries failing sanity checks, and writes
`data/frequency_import.tsv` at **tier 4** — so imported items only ever surface after
the curated core, each flagged `unverified machine translation`.

---

## Configuration

Everything is in `.env` (see `.env.example`). **All of it is optional** — the app runs
with no keys at all.

| Variable | Default | Notes |
|---|---|---|
| `SM_W2V_MODEL` | Maltese wav2vec2 | The default recogniser. |
| `SM_W2V_DEVICE` | `auto` | `mps` on Apple Silicon, else `cpu`. |
| `OPENAI_API_KEY` | — | Optional cloud Whisper STT. Not needed; the local default is faster and measures the same. |
| `AZURE_SPEECH_KEY` | — | Official `mt-MT` voices + Azure STT. |
| `SM_TTS_PROVIDER` | `auto` | `azure` · `edge` · `elevenlabs` · `mms` |
| `SM_STT_PROVIDER` | `auto` | `openai_whisper` · `elevenlabs` · `faster_whisper` · `azure` |
| `SM_WHISPER_MODEL` | `small` | Local STT size. |
| `SM_DAILY_NEW` | `12` | New cards per day. |
| `SM_TARGET_RETENTION` | `0.9` | Raise for shorter intervals and more reviews. |

The settings dialog (⚙) shows which providers actually resolved.

---

## Layout

```
backend/
  main.py         FastAPI routes + static serving
  dialogue.py     scripted turns: match an answer, pick the canned reply
  phonetics.py    Maltese phonetic keying, so matching survives ASR spelling
  srs.py          FSRS-5 scheduler
  text.py         Maltese folding, scoring, word diff, article assimilation
  curriculum.py   deck loading, queue building, i+1 learner profile
  tts.py  stt.py  provider chains
  db.py           SQLite: cards, state, reviews, turns, errors
data/
  core_vocab.tsv  350 curated words        ← edit these freely
  phrases.tsv     120 formulaic chunks
  dialogues.json  3 scripted conversations
  grammar_notes.md  learner reference
frontend/         index.html · app.js · style.css (no build step)
```

Progress lives in `data/progress.db` (SQLite); audio is cached in
`data/audio_cache/`. Both are gitignored — delete either to reset.

Adding vocabulary is just editing a TSV and restarting; ids must stay unique and
stable, since they key your review history.

## Tests

```bash
./.venv/bin/python -m pytest tests/ -q
```

77 tests covering the scheduler (interval growth, lapses, difficulty bounds, grade
ordering), the Maltese text comparison (diacritic and hyphen tolerance, article
assimilation, fusion linting, language classification), the phonetic matcher, the
scripted dialogues (every authored line is checked for correct Maltese, and every
node for reachability), and deck integrity.

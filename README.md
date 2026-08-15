# Nitkellmu — Speak Maltese

A conversational Maltese tutor that listens, speaks, shows you the writing, corrects
you gently, and schedules everything you meet with spaced repetition.

```bash
./run.sh
```

Then open <http://127.0.0.1:8137>. First run creates `.venv` and `.env`; add an
`ANTHROPIC_API_KEY` to `.env` to enable conversation. Speech in and out work with no
keys at all.

---

## What it does

**Talk (`Taħdita`)** — Pick a scene (café, market, directions, doctor, the village
festa…) and talk. Hold the mic or type. Every tutor turn comes back as Maltese audio
plus the written Maltese, an English translation you can hide, and a word-by-word
gloss you can unfold when a sentence won't come apart.

**Gentle correction** — When you get something wrong, the tutor answers *what you
meant* first, then shows a small amber card: the corrected sentence, at most two
issues with a one-line reason, and a **repeat prompt**. You say the corrected version
back into the mic; it is scored, and once you land it the sentence is added to your
review deck. Corrections you never repeat still get logged, and the tutor is told
about your recent mistakes on every turn so it watches for repeats.

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
prefixes, `għandi`, broken plurals, counting forms. The same file is fed to the tutor
as grounding on every turn, so the app and the tutor can't drift apart.

Keyboard: `space` reveals / grades Good, `1`–`4` grade, `r` replays audio, and in the
Talk view holding `space` is push-to-talk.

---

## The learning design

Each of these is a deliberate choice, not a default:

- **Frequency-first, but curated.** Tier 1 vocabulary is the ~100 items that buy the
  most coverage. See [the note on the source list](#about-the-2000-word-list).
- **Chunks before words.** Phrases interleave with single words in the new-card
  queue, because fluent speech is largely prefabricated sequences. You always leave a
  session with something *sayable*.
- **Comprehensible input at i+1.** The tutor is handed your known-word pool every
  turn and told to stay inside it plus at most two new items.
- **Production over recognition.** Recognition and production are tracked separately
  per card, and the queue biases toward whichever is lagging.
- **Interleaving, not blocking.** New cards are spread through the review queue
  rather than front-loaded, and topics are mixed.
- **Recast + prompted repetition.** The correction pattern with the strongest
  evidence behind it: reply to the meaning, model the correct form, elicit it back.
- **Errors become cards.** A correction you had to work for is exactly the item worth
  scheduling.
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

**In (STT).** The chain is `openai_whisper → elevenlabs → faster_whisper → azure`.
Browser `SpeechRecognition` is unusable here for the same reason, so audio is recorded
in the page and posted to the backend. `faster-whisper` runs locally with no key, so
speech input works out of the box; the first run downloads the model.

Because recognisers routinely drop Maltese diacritics and split the fused article,
grading folds `ġ ħ ż ċ`, treats the silent `għ` as absent, makes the article optional,
and splits at hyphens — so `jien mill Awstralja` scores 1.00 against
`Jien mill-Awstralja.` while a real word substitution still costs you.

---

## Running the tutor on a free local model

Yes — but check the model first. Maltese is low-resource enough that most models will
generate *confident, wrong* Maltese, which for a correcting tutor is worse than no
tutor. `scripts/check_tutor_model.py` measures a model on the five things the tutor
actually has to do, with deterministic grading:

```bash
ollama pull hf.co/bartowski/EuroLLM-9B-Instruct-GGUF:Q4_K_M
python scripts/check_tutor_model.py --base-url http://localhost:11434/v1 \
       --model hf.co/bartowski/EuroLLM-9B-Instruct-GGUF:Q4_K_M --trials 4
```

Then point the app at it — no API key needed:

```bash
SM_OPENAI_BASE_URL=http://localhost:11434/v1
SM_OPENAI_MODEL=hf.co/bartowski/EuroLLM-9B-Instruct-GGUF:Q4_K_M
```

**Measured on this machine** — grounded, i.e. with `grammar_notes.md` in context as
the live tutor does. Trials shown as *passed / run*, because low-resource output is
high-variance and a single run is misleading:

| model | mean | article fusion | counting form | no copula | stays in Maltese | JSON |
|---|---|---|---|---|---|---|
| **EuroLLM-9B-Instruct** Q4_K_M | **82%** ✅ | 1/4 ⚠ | 4/4 | 4/4 | 4/4 | 4/4 |
| gemma3:12b | 58% | 0/3 | 0/3 | 3/3 | 3/3 | 2/3 |
| qwen3.5:9b | 53% | 0/3 | 0/3 | 3/3 | 3/3 | 0/3 |
| qwen3:4b | unusable | — | — | — | — | — |

[EuroLLM](https://huggingface.co/utter-project/EuroLLM-9B-Instruct) is the one to use.
It is EU-funded and trained on all 24 official EU languages, Maltese included, and it
is the only local model tested that both corrects reliably and returns clean JSON.
~5.6 GB, a couple of seconds per turn on Apple silicon.

Bigger general-purpose models do not rescue this — gemma3:12b is larger than EuroLLM
and scores 24 points lower, because size is not the constraint, Maltese in the
training mix is. qwen3.5:9b never once produced usable structured output (it is a
reasoning model and buried the JSON in its thinking), "corrected" a sentence while
leaving the error in it, and invented the non-word *tfaliet*. qwen3:4b asserted that
the Maltese for "I" is *Naw*. **A model card listing Maltese is not evidence — measure
it.**

**The gap, and the safety net.** EuroLLM's weak spot is preposition + article fusion
(`minn` + `il-` → `mill-`), which it fixes about one time in four — and that is the
single most common error an English speaker makes. But it is also *fully mechanical*,
so it does not need a model at all. `text.lint_fusion` checks it by rule on every
turn, regardless of backend, and will:

1. repair the tutor's own output, so a "corrected" sentence never ships with the
   error still in it, and
2. raise the correction itself when you made the mistake and the model let it pass.

That closes EuroLLM's main gap deterministically. Everything else — idiom, register,
whether a reply is a *good* conversational turn — still degrades on a small model, so
a hosted model remains noticeably better company. Local is genuinely usable; it is not
equal.

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

**Measured**, with `scripts/compare_stt.py` on 25 deck sentences spoken by the app's
own `mt-MT` voice:

| model | WER | fWER | CER | app score | **would pass** | s/clip |
|---|---|---|---|---|---|---|
| **whisper-large-maltese-…-ct2** | 50.3% | **5.3%** | 8.0% | **0.98** | **96%** | 8.7 |
| whisper `small` (generic) | 127% | 120% | 48.4% | 0.40 | **4%** | 3.3 |

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

Everything is in `.env` (see `.env.example`). All of it is optional except the tutor.

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Enables conversation. Without it, review drills still work. |
| `SM_TUTOR_MODEL` | `claude-opus-5` | |
| `OPENAI_API_KEY` / `SM_OPENAI_BASE_URL` | — | Alternative tutor; point at Ollama/LM Studio for fully local. Also enables Whisper STT. |
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
  tutor.py        LLM turn: correction + reply as one structured call
  srs.py          FSRS-5 scheduler
  text.py         Maltese folding, scoring, word diff, article assimilation
  curriculum.py   deck loading, queue building, i+1 learner profile
  tts.py  stt.py  provider chains
  db.py           SQLite: cards, state, reviews, turns, errors
data/
  core_vocab.tsv  350 curated words        ← edit these freely
  phrases.tsv     120 formulaic chunks
  scenarios.json  13 conversation scenes
  grammar_notes.md  learner reference + tutor grounding
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

39 tests covering the scheduler (interval growth, lapses, difficulty bounds,
grade ordering), the Maltese text comparison (diacritic and hyphen tolerance,
article assimilation, diffs) and deck integrity.

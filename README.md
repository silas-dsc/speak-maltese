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

**Talk (`Taħdita`)** — Pick one of **35 scenes** (introductions, the café, the
market, the doctor, the pharmacy, the bus, getting lost, booking a table, getting
unstuck…) and work through it out loud. Hold the mic or type. Each line comes back as Maltese audio
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

### A smaller CTC model, for the browser

The 200MB on-device model is what kills a tab on an iPhone SE, so the same author's
**QuartzNet15x5** — 18.9M parameters against wav2vec2-large's 315M, trained on the same
64h Maltese corpus — was measured on the same 25 clips, through the ONNX export at
[`OpenVoiceOS/…quartznet15x5…_onnx`](https://huggingface.co/OpenVoiceOS/carlosdanielhernandezmena-stt_mt_quartznet15x5_sp_ep255_64h_onnx):

| model | params | weights | fWER | app score | **would pass** | s/clip |
|---|---|---|---|---|---|---|
| **wav2vec2-large-xlsr-53-maltese-64h** | 315M | 201MB (q4f16) | **5.3%** | **0.98** | **96%** | 0.2 (Metal) |
| QuartzNet15x5 (NeMo CTC, ONNX) | **18.9M** | **76MB** (fp32) | 18.5% | 0.93 | 80% | **0.1 (CPU)** |

**Rejected, and not by a close margin.** 80% pass means one correct answer in five is
thrown back at the learner, and being told you are wrong when you are right is the one
failure this app cannot afford. The errors are not noise either, they are confident
wrong words: `mingħajr zokkor` → `min għajd sokkor`, `nqum fis-sitta` → `u fis-sitta`,
`hawn` → `hawl`, and the degemination this project keeps fighting (`irrid` → `irid`).
Both models miss `minn l-Awstralja` and `x'jum hu` identically, so those two are the
reference audio, not the recogniser.

What it does prove is the *shape* of the win, if the accuracy could be recovered: half
the wall-clock on a **CPU** against the large model on the GPU, convolution-only so no
WebGPU is needed, and small enough to load anywhere.

Reproduce with `scripts/compare_stt.py`; the NeMo backend builds its own 64-bin log-mel
features (`onnx-asr` ships only 80 and 128), with every constant taken from the
checkpoint's `model_config.yaml` — `window_size: 0.02` is 320 samples, not the 400 the
Conformer models use, and that one is a silent accuracy loss rather than an error.

### Constrained decoding, and why it does not rescue the small model

The app never has to transcribe. It shows a line, the learner says it, and the only
question is whether what came back was that line — so scoring the *known target* with
the CTC forward algorithm asks a strictly easier question than free decoding, and none
of QuartzNet's confident wrong words above is a plausible alignment of the target.
`scripts/constrained_ctc.py` measures that, per frame, against the greedy path as a
ceiling.

It works, spectacularly, on the wrong test. Scored against the other 24 lines in the
eval set both models pick the true line **100%** of the time, with true confidence ~0.98
against ~0.10 for the alternatives. That number is worthless: `Bonġu` against `In-nanna
tagħmel il-pastizzi` is not a decision anything gets wrong, and a decoder that only ever
scores the target would accept silence.

The test that counts uses **near-misses** — the same line with a word dropped, a
geminate lost (`irrid` → `irid`), an assimilated article swapped (`mal-` → `mill-`) —
which share most of their audio with the target:

| model | beats other lines | **beats every near-miss** | conf ✓ | conf near-miss | **TPR@95 hard** |
|---|---|---|---|---|---|
| **wav2vec2-large-xlsr-53-maltese-64h** | 100% | **88%** | 0.983 | 0.539 | **92%** |
| QuartzNet15x5 | 100% | 72% | 0.988 | 0.594 | **8%** |

The last column is the one that decides it: at the confidence threshold that rejects 95%
of near-misses, the large model still accepts 92% of correct answers and QuartzNet
accepts **8%**. Its acoustic posteriors are not sharp enough for the ratio to mean
anything — a near-miss explains its audio almost as well as the truth does, so no
threshold separates them. Constrained decoding is a real capability, and it needs a
model that is confident frame by frame.

Two of the 25 clips (`Illum x'jum hu?`, `Jien minn l-Awstralja.`) fail for *both* models
in the same way, here and under free decoding. Those are the reference audio, not the
recognisers.

### Matching against the stored audio instead

The app ships a recording of every line it can ask for, so the recogniser could in
principle be replaced by a comparison: encode what was said, encode what should have
been said, warp one onto the other. No transcription, and — with cepstral features — no
model at all. `scripts/dtw_match.py` measures it, with the reference in the app's own
`mt-MT-GraceNeural` and the query in `mt-MT-JosephNeural`, so speaker mismatch is
present rather than assumed. Negatives are the same near-misses as above, spoken in the
query voice.

| encoder | rank-1 | closer than every near-miss | **TPR@95 hard** |
|---|---|---|---|
| wav2vec2 posteriors (201MB) | 100% | 72% | **44%** |
| QuartzNet posteriors (76MB) | 96% | 60% | 32% |
| MFCC + CMVN (**no model**) | 92% | 68% | 28% |

**All three fail, including the large model.** That is the useful part: at 44% against
the 92% the same model reaches by scoring the target sequence directly, the weakness is
the *method*, not the size of the encoder. Template matching asks whether the audio
resembles one particular rendering of the line, and warping is glad to stretch across a
missing word — the deletion costs a little extra path, not a contradiction. Constrained
CTC asks whether the audio contains that exact sequence of tokens, where a missing word
is an obligatory emission that never happened. Structural beats similar.

Both voices here are synthetic, so a human learner is a larger mismatch than this and
the numbers are optimistic — most of all for MFCC. The step pattern is also the
slope-constrained symmetric one, chosen so the search vectorises, and it absorbs
deletions more cheaply than a stricter pattern would; a duration penalty would recover
some of the near-miss gap. Neither caveat is worth chasing while the ceiling sits at
44%.

### Distilling the teacher into 10MB

Every attempt above took a model that already existed. The one thing left was to train
one *for this app*, which is a different problem: the answer space is closed, the audio
distribution is `edge-tts` output, and there is a 315M-parameter model on hand to
supervise it frame by frame. `scripts/distill_stt.py` does that.

* **Teacher signal.** The large model's log-posteriors over its 44 characters, at 50fps,
  for every clip — a soft distribution per frame rather than a hard label per utterance.
  That is the part that matters: near-miss rejection is a likelihood ratio, so what has
  to be transferred is the teacher's *confidence*, not its argmax.
* **Data.** Every line the app can ask for, in both `mt-MT` voices at two rates each —
  1,494 lines, 5,976 clips, rendered by `prebuild_audio.py`. The 25 evaluation sentences
  are excluded by name, and the eval clips are rendered at a rate that is not in the
  training set either.
* **Student.** A QuartzNet-shaped depthwise-separable convolution stack. No attention
  anywhere, deliberately: WASM has no fast attention kernel, and WebGPU is exactly what
  an iPhone could not afford. 2.56M parameters, 10.2MB of fp32 ONNX.

Augmentation is SpecAugment in the feature domain only, because the teacher's posteriors
were computed on the clean audio and anything that shifted the waveform in time would
leave them describing the wrong frames.

Four sizes were trained on the same data with the same schedule, to find where it breaks:

| | params | size | fWER | app score | free pass | closer than every near-miss | **near-miss TPR@95** | s/clip |
|---|---|---|---|---|---|---|---|---|
| wav2vec2-large (teacher) | 315M | 201MB | **5.3%** | **0.98** | **96%** | 88% | **92%** | 0.26 (Metal) |
| student | 2.56M | 10.2MB | 13.1% | 0.94 | 88% | 92% | 84% | 0.11 (CPU) |
| student | 1.01M | 4.0MB | 18.3% | 0.95 | 88% | **96%** | 80% | 0.08 (CPU) |
| **student** | **0.53M** | **2.1MB** | 21.5% | **0.95** | **88%** | 92% | **84%** | **0.08 (CPU)** |
| student | 0.24M | 1.0MB | 25.4% | 0.92 | 80% | 92% | 80% | 0.08 (CPU) |
| QuartzNet15x5 | 18.9M | 76MB | 18.5% | 0.93 | 80% | 72% | 8% | 0.15 (CPU) |

**These are the first small models that work at all.** 84% against the teacher's 92% on
the test that decides — at a hundredth of the size, on the CPU, in a third of the
wall-clock the teacher needs on the GPU. Every student beats the 76MB QuartzNet's 8% by
an order of magnitude, and on "closer than every near-miss" all of them beat the teacher
outright. Where they lose is the *tail*: near-miss confidences average 0.69-0.77 against
the teacher's 0.539, so a threshold strict enough to reject 95% of them costs more right
answers.

**The curve is flat, and that is the finding.** fWER degrades exactly as capacity falls —
13% → 18% → 22% → 25% — and the app-level numbers do not move with it: 84%, 80%, 84%,
80% on 25 clips, where one clip is four points. Free transcription gets steadily worse
while *deciding whether the learner said the line* does not, because deciding was never
the hard part. Under those two metrics **2.1MB is indistinguishable from 10MB**, and
capacity is not what is binding — the data is. All four overfit 1,494 lines: train KD
falls to 0.05 while dev sits at 0.18 from about epoch 50.

Twenty-five clips is a small set and four points is one of them, so read this as "flat
between 1MB and 10MB", not as a ranking within it.

**The caveat that matters more than any of the numbers.** The student has never heard a
human being. Its entire training distribution is two synthetic voices, while the teacher
inherits XLSR-53's pretraining on ~56,000 hours of real multilingual speech — which is
precisely what makes a recogniser survive an unfamiliar voice. These numbers are measured
on synthetic speech and should be read as an upper bound on real learners.
`scripts/compare_stt.py --record 20` is how that gets settled, and it needs somebody's
actual voice.

Which also says where the next gains are, and they are cheap: more lines and more voices.
`edge-tts` has other `mt-MT` speakers, `--rate` is a free axis, and
`scripts/import_corpus.py` can widen the text far past the 1,494 lines the deck holds.
Nothing here suggests a bigger student would help.

### Accented TTS as a proxy for learner speech — tried, and it did not work

There is no L2 Maltese corpus; I looked. But `edge-tts` has 322 voices of which two are
Maltese, and a foreign voice reading Maltese orthography mispronounces it roughly the way a
speaker of that language would. The label is the deck line regardless, so it is supervised
data in unlimited quantity, aimed squarely at the gap. `scripts/render_accents.py` renders
it — 6,008 clips across English, Italian, Arabic, French, German and Spanish voices.

It genuinely does mangle Maltese. Against `Bonġu! Jien Marija. X'jismek?` the teacher hears:

| voice | heard |
|---|---|
| it-IT | `bonu gejne maria eks jesmeck` |
| ar-EG | `bondgo l-agin maraggi lixe gismak` |
| es-ES | `bonù kien marika inkis hismajk` |
| en-AU | `bongu ġian marija eks ġiżmek` |

`ġ` confused with g and j, no gemination, and `x` read as the English letter name — the
learner error classes, exactly.

**It made no difference, and slightly hurt.** 12,016 accented passes added to the 74,571
already there, trained the same way, measured on the same held-out sets:

| | | v2 (shipped) | v3 (with accents) |
|---|---|---|---|
| learner's voice | fWER | **93.7%** | 94.3% |
| | rank-1 | **80%** | **80%** |
| synthetic | fWER | **16.3%** | 19.0% |
| native (FLEURS) | fWER | **74.6%** | 78.1% |

Identical on the number the app grades with, worse on everything else — consistently, not
noisily. The training itself worked (dev KD 0.49 → 0.29); it learned the wrong thing. A
grapheme-to-phoneme engine produces one deterministic mispronunciation per voice, where a
learner produces a variable, partly-correct approximation that drifts word to word. The
model got better at English-TTS-reading-Maltese, which is a different point in acoustic
space from a human attempting it.

Not deployed. The renderer and the pipeline support stay, because the negative result is
worth keeping and the next person will otherwise have the same idea.

**The two levers that did work**, for contrast: real native speech (102.5% → 74.6% fWER)
and constrained ranking (0% → 76% of a learner's correct answers accepted). What is left
with evidence behind it is real learner speech from many speakers, which is a
data-collection problem rather than a modelling one.

### Grading a learner, measured on a learner

25 recordings in a non-native voice settled several things at once, and most of them were
not what the model was doing.

**The ceiling is not the small model.** On 16 clean clips of a learner's Maltese, the 201MB
teacher — 315M parameters, fine-tuned on 64 hours of native Maltese — marks **25%** of
correct answers correct. The 2.1MB student manages 0%. Every recogniser available is
trained on native speech, and a learner's `għ`, geminates and `q` are not what any of them
expect. No amount of distillation closes that.

**An absolute confidence cannot be calibrated across speakers.** The first threshold,
0.9867, came from synthetic speech; on a real voice a *correct* answer scored 0.766, so it
never fired once. Worse, near-misses scored 0.784 — above the target — so no cut exists.

**Ranking does work, because it is scale-free.** Score the line we asked for against a field
of other answers the script accepts, on the same audio with the same denominator, and take
the ordering rather than the value:

| on the learner's clean clips | accepted |
|---|---|
| threshold at 0.92 | 0% |
| **target beats a field of 24** | **75%** |

Ranking alone is not the whole rule, and negative controls are what showed it. Against an
all-blank posterior a *shorter* sequence is the likelier reading, so silence and room noise
both won against a field of longer alternatives. And one wrong answer beat its field by
0.006, which is noise rather than evidence.

So acceptance also needs a floor and a margin, and both were fitted by sweeping them
against 25 clean recordings, 75 wrong-line pairings, and 36 clips of silence and noise at
several levels:

| floor | ranking | accepted | wrong-line | silence/noise |
|---|---|---|---|---|
| 0.65 | yes | 19/25 | 3/175 (2%) | 0/64 (0%) |
| **0.55** | **yes** | **22/25** | **4/175 (2%)** | **6/64 (9%)** |
| 0.24 | no | 25/25 | 131/175 (**75%**) | 20/64 (**31%**) |

0.55 is the setting: three more of a learner's correct answers for no measurable change in
wrong-line acceptance. It does let some room noise through, which `capture.js` is the first
line against.

Accepting all 25 was asked for and costed. It requires dropping the ranking test altogether,
because three of those clips *lose* their rank — the model prefers a different deck line to
the one that was said. At that point 75% of wrong answers and a third of all silence are
marked correct, which is not a lenient grader but no grader, and it would feed false correct
answers into the FSRS scheduler and rot the review deck. Declined on those numbers. An energy gate was tried, to
let the floor come down — real speech peaked at 0.26 and above, noise mostly below — and it
bought exactly one clip (19→20) while loud noise still cleared it. One extra knob fitted on
25 samples for a four-point gain is over-fitting, so it was dropped.

Three of the six remaining rejections rank *first* and are refused by the floor alone
(`Dur lejn ix-xellug` at 0.621, `Magħluq il-Ħadd` at 0.570, `Ma jogħġobnix` at 0.573). That
is the cost of the setting, taken knowingly: a tutor that congratulates room noise is worse
than one that occasionally asks again.

**The trade is deliberate.** Ranking accepts near-misses: say `irid` for `irrid` and it
passes, because the posteriors do not resolve consonant length in a learner's speech. Saying
a *different* line is still rejected. For somebody learning, being able to progress beats a
phonetic precision that no available Maltese model can actually judge — and that was a
product decision, taken with these numbers in hand, not a technical accident.

The recordings also exposed the recorder: 9 of the 25 were faulty and nothing said so at the
time. One clip sat 30dB below the rest and was the worst-scoring in the set; eight more were
digitally clipped, all late in the run. Both are now reported per clip as they land.

### The model had never heard a human

The first student was trained on two synthetic voices and nothing else, and the report
from a real phone was that it was "very inaccurate". Measuring it properly, on 150
held-out FLEURS clips of real Maltese speakers, showed that was generous:

| on real human speech | fWER | CER | app score | pass |
|---|---|---|---|---|
| wav2vec2-large (teacher) | **19.5%** | **6.8%** | **0.93** | **93%** |
| first student, 2.1MB | 102.5% | 68.0% | 0.17 | **0%** |

Not a degradation — a total failure. The same student scores 21.5% fWER on synthetic
speech, and the teacher handling the identical clips rules out the eval being unfair.
Every number this project had reported until then was measured on the app's own TTS
voices, which is exactly the kind of blind spot that produces a confident wrong answer.

So the training distribution was rebuilt around what actually reaches the model:

* **Real speakers.** FLEURS `mt_mt` — 3,149 clips, ~15,000 utterance-length chunks.
  Distillation needs no transcripts (the teacher labels whatever it is given), so any
  Maltese audio counts. Long Wikipedia sentences are cut to ~3s because the app asks for
  phrases, not paragraphs.
* **The codec.** Training audio now goes through the same Opus round-trip
  `MediaRecorder` produces. Nothing else reproduces what 24kbit Opus does to the
  fricatives `ħ`, `x` and `għ` live in.
* **Vocal tracts, rooms, microphones.** Resampling shifts pitch and formants together —
  crude VTLP, but it turns two voices into many. Plus early reflections, a noise floor,
  level variation and clipping.

All of that is time-domain, so the teacher is re-run per variant: its posteriors describe
the audio it was given, frame for frame. That is the step the first round skipped — it
masked features only, to keep precomputed posteriors aligned, which is precisely why it
could not fix a domain gap.

**The first attempt at this collapsed, and the failure is worth keeping.** With 60% of the
data untranscribed, real speech trained on frame-level KD alone — and blank frames
outnumber character frames by an order of magnitude in any CTC posterior, so matching the
teacher per frame is satisfied most cheaply by predicting blank everywhere. The student
went silent on real audio (100% blank across 535 frames) while still reading the synthetic
clips it had a CTC target for. The fix needed no extra teacher time: the posteriors were
already on disk, so decoding them gives the sequence the teacher would have transcribed,
and that becomes a CTC target like any other (`distill_stt.py pseudo`).

| | fWER | CER | app score | pass |
|---|---|---|---|---|
| **real speech** | | | | |
| teacher, 201MB | 19.5% | 6.8% | 0.93 | 93% |
| **v2, 2.1MB** | **74.6%** | **26.3%** | **0.64** | **19%** |
| v1, 2.1MB | 102.5% | 68.0% | 0.17 | 0% |
| **synthetic** | | | | |
| **v2, 2.1MB** | **16.3%** | 10.2% | 0.94 | **92%** |
| v1, 2.1MB | 21.5% | 11.6% | 0.95 | 88% |

Better on both, at the same 2.1MB. Under constrained scoring the real-speech gain is
larger than the transcription numbers suggest: the true line is ranked first 87% of the
time against v1's 33%, and its confidence rises from 0.622 to 0.973.

**It is still far short of the teacher, and 19% free-decode pass on real speech is not a
good number.** What it is, is the first version that works on a human being at all. The
levers left are more real Maltese speech and actual recordings from the people using it —
not a bigger model, which the flat size curve already ruled out.

### What the app grades on now

The app used to free-decode and then string-match the transcript against the line it had
asked for — grading on the model's weakest output while holding its strongest in reserve.
It now scores the target sequence directly (`frontend/nanostt.js`, parity-tested against
`scripts/constrained_ctc.py`) and accepts at a confidence of **0.9867**, the cut that
rejects 95% of near-misses while keeping 84% of correct answers.

It is a **floor, never a penalty**: below the threshold, grading falls back to the
transcript diff exactly as before. So a threshold miscalibrated for a real voice can fail
to help, but cannot mark a good answer wrong.

Measured in the browser, target fixed and the speech varied — the direction the app
actually faces:

| spoken | transcript | confidence | verdict |
|---|---|---|---|
| `Hemm spiżerija hawn qrib?` | `hemm spiżerija aw rib` | 1.0085 | **accept** |
| `Irrid nara tabib.` | `irrid nara tabib` | 1.0317 | accept |
| `Nara tabib.` (word dropped) | `nara tabib` | 0.3461 | reject |
| `Irrid nara.` (trailed off) | `irrid nara` | 0.3651 | reject |
| `Irid nara tabib.` (degeminated) | `irid nara tabib` | 1.0021 | **accept — wrong** |

The first row is the point: a transcript that string-matching would have failed, correctly
accepted. Word-level errors are turned away with a wide margin.

**Geminates are not caught, and this does not fix them.** `kolox` for `kollox` scores 1.02
— the model transcribed the degeminated audio as `kollox`, so its posteriors do not resolve
consonant length at all. That is not a regression (string-matching accepted it too, for the
same reason) but it is the clearest thing the next round of training has to buy, and the
teacher can already do it: 5.3% fWER against the student's 21.5%.

### Is 201MB the floor?

For anything `onnxruntime-web` can execute on a GPU, close to it. The shipped
`model_q4f16.onnx` breaks down as:

| | size | why it is not smaller |
|---|---|---|
| 4-bit `MatMulNBits` weights | 156.0 MB | already the smallest ORT web executes |
| fp16 block scales | ~19.5 MB | one per 32 weights; `block_size=128` would save ~15MB |
| fp16 `Conv` weights | 25.2 MB | no sub-8-bit convolution kernel exists in ORT web |
| norms, biases, constants | ~0.6 MB | |

So ~185MB is the realistic floor, which does not change the outcome on a phone — a
WebKit page gets 250-350MB. **Below that means fewer parameters, not fewer bits.** int8
is not the answer either and was rejected on measurement, not principle: onnxruntime-web
has no int8 GPU kernel, so an int8 model falls back to WASM at 0.22× realtime.

One incidental find in that accounting: a single 16.8MB fp16 tensor feeds a `ReduceL2`
node — the weight-norm reparameterisation of `pos_conv_embed`, which the export left to
be renormalised on every forward pass. `remove_weight_norm()` before export folds it.
That is compute, not bytes.

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

## A real Maltese frequency list

`scripts/build_frequency.py` derives one from **MLRS Korpus Malti** (University of
Malta) — 45M tokens across its conversational genres, deliberately excluding the legal
and parliamentary ones which would otherwise rank `regolament` above `ħobż`. The
result is `data/frequency_mt.tsv`, the top 5,000 words. It replaces the
machine-translated list this project started from, which was never a Maltese frequency
list at all.

```bash
python scripts/build_frequency.py --korpus --retier
```

The corpus is **gated and CC BY-NC-SA 4.0**: you accept its terms yourself, and
anything derived inherits non-commercial and share-alike — so that file is carved out
of this repo's MIT licence. Without access it falls back to Maltese Wikipedia (CC
BY-SA), blended with Tatoeba to offset the encyclopedic register.

Two things the build has to get right:

* **Names are not vocabulary.** Tatoeba's Maltese sentences use `Ziri` as their
  stand-in person the way the English ones use "Tom" — 328 occurrences in 646
  sentences, which put it at *rank 10* of the first list. Tokens that are almost
  always capitalised are dropped. The threshold matters: at 75% it also swallowed
  `gvern`, `ministru` and `partit`, which are ordinary nouns that happen to be
  capitalised in news copy.
* **Display is not the matching key.** Counting folds diacritics, so `tiegħu` and
  `għall` group correctly — but printing the folded key gives `tieu` and `all`. The
  commonest real spelling is kept for display.

And the reason retiering is a *proposal* rather than an edit, in one line:

| word | corpus rank | when a learner needs it |
|---|---|---|
| `li` | 1 | eventually |
| `jekk jogħġbok` | 549 | first lesson |
| `bonġu` | 2,588 | first minute |

Frequency measures print. It does not measure what you need to say to order a coffee,
so `--retier` writes `data/retier_proposal.tsv` for you to read rather than rewriting
the deck.

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

## Scene coverage

A word you only ever meet as a flashcard is a word you cannot say. So the decks and
the scenes are checked against each other:

```bash
python scripts/coverage.py
python scripts/coverage.py --missing --tier 3   # what to write next
```

```
35 scenes · 113 turns · 855 spoken lines
712 distinct Maltese words appear in the dialogues

Deck coverage: 430/430 words  (100%)
  tier 1  ████████████████████  130/130 (100%)
  tier 2  ████████████████████  184/184 (100%)
  tier 3  ████████████████████  116/116 (100%)
```

Comparison is on the folded form, so a missing diacritic never reads as a miss. The
scenes were written *from* the `--missing` list rather than invented and measured
afterwards, which is why the last twelve are the ones they are — a pharmacy, a bus,
getting lost, booking a table.

A test asserts this stays at 100%: adding a word to a deck now means writing it into
a conversation too, which is the constraint that keeps the two halves honest.

---

## Where your progress lives

In your browser, in IndexedDB — not on the server.

That is a deliberate change. The schedule used to live in a SQLite file next to
the app, which is fine for one person on one laptop and wrong for anything else:
every visitor to a deployment shared one review history, and free hosting throws
the disk away on each restart, so the one thing in this app that cannot be
regenerated was also the least durable. Decks, audio and images all rebuild from
this repo; nobody can recover which words you were about to forget.

So the FSRS scheduler was ported to JavaScript and the server made stateless:

| | |
|---|---|
| `frontend/srs.js` | FSRS-5, a direct translation of `backend/srs.py` |
| `frontend/store.js` | IndexedDB — cards, schedules, review log |
| `frontend/schedule.js` | queue building, counts, streaks, progress |

`backend/srs.py` stays as the reference implementation. `tests/test_srs_parity.py`
runs both over the same review sequences and compares state, stability, difficulty,
due dates and the interval labels on the grade buttons — because two implementations
of one algorithm drift silently, and the symptom is reviews arriving at subtly wrong
times with no way to notice.

Settings → **Export** writes the whole database to a JSON file, for moving between
devices or keeping a copy. **Import** replaces what is on the device.

---

## Recognition on the device

Speech recognition is the last thing that needs a server. It can run in the page
instead, which makes the whole app static: no cold starts, no round trip per
utterance, and it keeps working offline.

```bash
pip install optimum onnx onnxruntime onnxconverter-common onnx_ir
python scripts/export_onnx_web.py --only q4f16     # ~2 min, writes web/models/
./run.sh
```

Then **Settings → Recognise speech on this device**. It is off by default because
turning it on downloads about 200MB, once.

To try the recogniser on its own, without the app around it:

```bash
python -m http.server 8000
```

and open <http://localhost:8000/web/stt-test.html>. Hold the button and speak, or
press **Score against the eval clips** to run it over `data/eval_clips`. Add
`?dtype=fp16` or `?device=wasm` to compare builds. transformers.js comes from a
CDN so there is no build step; the model and the audio never leave the machine.

### What the numbers say

| build | size | speed | notes |
|---|---|---|---|
| wasm int8 | 355 MB | 0.22× realtime | unusable — 2s of speech takes 9s |
| webgpu int8 | 355 MB | 0.34× | no int8 GPU kernel; falls back to CPU |
| webgpu fp32 | 1262 MB | 11.6× | |
| webgpu fp16 | 631 MB | 23.5× | |
| **webgpu q4f16** | **201 MB** | **30×** | ships |

**WebGPU is required.** On WASM the same model is slower than any network, so the
toggle refuses to enable without it and the server path stays. Chrome and Edge
everywhere, Safari 26+, Firefox on Windows only.

Accuracy holds: all 334 accepted answers in the scenes were rendered to speech and
run through the 4-bit build, and 333 grade correct — see
[`tests/test_q4_recogniser.py`](tests/test_q4_recogniser.py), which replays those
transcripts against the real grader.

---

## Deploying

`Dockerfile` builds a self-contained image; `README.hf.md` is the Space card for
Hugging Face (rename it to `README.md` in the Space).

**A Docker Space is not free any more.** `POST /api/repos/create` answers `402
Payment Required` — "Static Spaces are free for everyone, but hosting Gradio and
Docker Spaces on free cpu-basic requires a PRO subscription" — so the recogniser
needs either a PRO account or any other container host with ~2GB of memory. The
image is not tied to Spaces; it wants a port and 2GB.

Wherever it ends up, that URL is what `STT_BASE` names in
[`.github/workflows/pages.yml`](.github/workflows/pages.yml). Set the repository
variable and the next Pages deploy stops shipping on-device recognition entirely,
which is the only way speaking works on a phone: a WebKit page on an iPhone SE has
250-350MB to live in, and the model is 200MB before a single tensor. Leave it unset
and the static build stays on-device, where iOS is excluded by
`localstt.affordable()` and speaking is only offered to devices that can hold it.

Two things make the image behave on a small host:

* **The model is baked in at build time.** Downloading 1.2GB of weights on first
  request would put the whole wait on whoever opens the app after a restart, and
  a host that sleeps when idle restarts often.
* **Startup is shown, not hidden.** A cold container needs up to a minute to load
  the recogniser. `/api/health` reports whether it is actually loaded — the
  *loaded object*, not merely that a preload thread started — and the client holds
  a progress screen until it says ready. The first two steps report real progress;
  the model wait creeps asymptotically and snaps to full when health flips, because
  the server cannot know how far through it is and a fake percentage is worse than
  an honest "still working".

  The static build does the same thing across origins: given `STT_BASE` it polls
  `/api/health` there during startup, which both wakes a sleeping container and
  shows the wait. That poll is the pre-warm — paying the cold start behind the
  progress bar rather than at the moment somebody holds the mic. It is bounded at
  30 seconds, after which the app opens anyway and says speaking is still coming;
  reading, listening and reviewing never needed the recogniser.

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

144 tests covering the scheduler (interval growth, lapses, difficulty bounds, grade
ordering), the Maltese text comparison (diacritic and hyphen tolerance, article
assimilation, fusion linting, language classification), the phonetic matcher, the
HTTP API, and the build scripts.

The largest group is on the content itself, because a model is no longer in the loop
and so nothing at runtime can notice a badly written scene:

* every authored line is checked for correct Maltese and for sun-letter assimilation;
* every scene is played to the end over the real API, answering correctly, and must
  neither loop nor skip a turn;
* every accepted answer must be graded correct by the matcher, and must be told apart
  from its siblings — two answers that sound alike would credit the wrong one;
* every `Għid: …` hint must itself be accepted, so the app never asks for a sentence
  it then refuses;
* and every word in the decks must be spoken in some scene, which is what keeps
  coverage at 100% rather than letting it drift.

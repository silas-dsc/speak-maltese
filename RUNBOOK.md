# Runbook: turning the staged switches on

Eight accuracy levers are implemented and switched off, because each one needs a
measurement that cannot be taken in CI — the grader ones need a learner's recordings, the
training ones need the teacher and a GPU. This is the order to take those measurements
in, and what each unblocks.

Nothing here is a suggestion about *what* the answer will be. Every step ends in a number,
and the number decides.

---

## 0. Local setup (macOS)

Recording is macOS-only: `compare_stt.py --record` captures through ffmpeg's
`avfoundation`, and plays the reference line back with `afplay`. So this part has to
happen on the laptop.

```bash
git clone https://github.com/silas-dsc/speak-maltese
cd speak-maltese
brew install ffmpeg                     # the recorder and every decoder go through it
./run.sh                                # creates .venv from requirements.txt, serves the app
```

`run.sh` installs torch, transformers and faster-whisper, which is a few gigabytes and is
what steps 3–6 need. Stop it once the venv exists; nothing below needs the server.

Two extras, only for the steps that name them:

```bash
.venv/bin/pip install onnxruntime       # step 3, the sweep — runs the 2.1MB model
.venv/bin/pip install onnx onnxscript   # step 6, exporting a retrained student
```

Check it works before recording anything:

```bash
.venv/bin/python -m pytest tests/ -q    # ~380 pass; the torch ones stop skipping
.venv/bin/python scripts/compare_stt.py --list-inputs
```

`--list-inputs` matters more than it looks. `:default` is whatever macOS currently calls
the default input, and on a machine with BlackHole or Teams installed that can be a
virtual device — which records twenty-five files of silence and tells you nothing until
scoring. Note the index of the real microphone and pass it explicitly.

On the machine this was written for, the audio devices come back as:

```
[0] BlackHole 2ch          [1] External Microphone       [2] MacBook Pro Microphone
[3] Silas's iPhone SE 2020 Microphone                    [4] Microsoft Teams Audio
```

`[0]` is exactly the trap above, and it is what `:default` may well resolve to.

**Run this yourself, in your own terminal.** `--list-inputs` needs no permission because
enumerating devices is not opening one, so it will happily succeed anywhere — but the
first *capture* triggers the macOS microphone consent dialog, and macOS shows that dialog
only to a foreground application. Started from anything that cannot put a window in front
of you (a tool-spawned shell, an agent, a CI runner), ffmpeg prints its banner and then
blocks forever with no error, no timeout and no clue. That failure is silent by
construction, so it is worth knowing before it eats twenty minutes.

Grant it once under System Settings → Privacy & Security → Microphone, for the terminal
you are using. After that, `--record` checks the device itself before it prompts for
anything, so there is no separate command to run: it opens the input once, reports how
long the device took to start and what its noise floor is, and refuses outright on a
device that produces pure silence. `[0]` above is refused by that check.

**Do not test a device with a two-second capture.** It is the obvious thing to try and it
reports the opposite of the truth. avfoundation hands ffmpeg the device before the device
is delivering, so the front of every recording is missing — about 0.3s on the built-in
microphone, and roughly two seconds on the *first* open of an iPhone over Continuity. A
`-t 2` capture of a perfectly good iPhone microphone therefore comes back as 0.01 seconds,
one packet, which reads as a dead input. It is not dead; it had not started. Measure with
three seconds or more, and read the captured *duration* alongside the peak.

The iPhone microphone is the most representative choice — the app runs on that phone, so
its microphone is the closest thing to what the recogniser meets in production, and every
number in the README carries the caveat that the existing clips are clean desk takes. It
needs the phone awake, unlocked and nearby; the phone shows "Connected to <your Mac>" when
Continuity has it. Measured on this machine it starts in 0.28–0.32s once warm and sits at
a noise floor of 0.006–0.009, quieter than the built-in microphone, so there is plenty of
headroom before the clipping that cost eight of the first 25 takes.

---

## 1. Record the clips — the gate

**If the clips already exist, skip this step and restore them instead.** All 75 are on the
Hugging Face Hub as the private dataset `silasdsc/speak-maltese-learner-clips`, alongside
the untrimmed originals and the synthetic contrast set; the restore command is in
[Where the recordings live](README.md#where-the-recordings-live). Recording is 40 minutes
of someone's time and the takes are not reproducible, so re-record only to *add* to the
set, never to recover it.


```bash
.venv/bin/python scripts/compare_stt.py --record 25 --input :1
```

It prompts each sentence, plays the app's own rendering of it, then arms the microphone,
waits out the device's measured start-up delay, and prints `▶ speak now`. **Wait for that
line.** Speaking into the gap before it loses the first word, which is where the hardest
sounds are — għ- and x- and ħ-. Recording stops when you press Enter. Every take is
level-checked as it lands: it reports quiet and clipped clips
immediately, which is worth acting on — of the original 25, **nine were faulty and nothing
said so at the time**, and the worst-scoring recording in the set was one sitting 30 dB
below the rest. A take that comes back shorter than half a second is reported as a stalled
input rather than a quiet one, because the fix for that is not the gain knob.

Redo anything it flags — `bad` discards exactly the takes that failed a level check,
and a comma-separated list names them by hand:

```bash
.venv/bin/python scripts/compare_stt.py --redo bad --record 25 --input :1
.venv/bin/python scripts/compare_stt.py --redo 4,11 --record 25 --input :1
```

Speak the way a learner would, not the way a careful reader would. The point of this set
is that it is *not* synthetic speech, and a self-conscious over-articulated take is closer
to TTS than to how the app will actually be used.

**This step gates steps 2 and 3 entirely.** Everything else can be done in any order.

---

## 2. Build the negatives

```bash
.venv/bin/python scripts/make_negatives.py
```

Writes 90 clips to `data/eval_clips/negatives/` — 20 digital silence, 20 white noise
across five levels, and each recording both attenuated by 30 dB and time-reversed — plus a
`manifest.tsv` recording exactly what was built.

Read the caveat it prints. The historical set's composition was never committed, so this
is a reconstruction of a set of the same size and kinds, not the same set. **Percentages
from it are not comparable to the ones in the README.** That is why step 3 always measures
the current rule alongside any candidate.

---

## 3. Sweep the grader — unblocks `DUR_FRAMES`, `DUR_SD_SLOPE`, `MARGIN_SIGMAS`, `FIELD_LOCAL`

One expensive pass, then any number of cheap ones:

```bash
.venv/bin/python scripts/sweep_grader.py --models frontend/stt      # scores, caches
.venv/bin/python scripts/sweep_grader.py --grid all --seeds 5       # free from here
```

The first command runs a CTC forward pass per hypothesis per clip and caches the
likelihoods to `data/eval_clips/sweep_cache.json`. Afterwards every parameter combination
is arithmetic over that cache, so a grid of two hundred settings costs what one does.

Read the table like this:

- **`learner` wants to go up. Everything else wants to stay at zero.** A change that
  raises `learner` while moving `wrong-line` or any negative column is a trade, and the
  README's history is that such trades were declined.
- **A one-clip difference is not a result.** With 75 clips one clip is 1.3 points, and the
  field is sampled — the README measures a four-point spread across seeds with the prior
  on. Use `--seeds 5` and believe a change only if it wins on every seed. Three of the
  refused clips flip with the draw rather than with any setting, so a change that moves
  only those has moved nothing.
- **Check *why* a clip was refused, not just that it was.** `decide` reports `clear` and
  `passes_floor` separately, because a floor that is too high and a field that is too hard
  want opposite fixes.

`--grid margin` and `--probe-loss` are already answered: loosening the margin buys no
accepts at any value down to zero, and forgiving a ranked loss large enough to matter
admits half of backwards speech. Both are recorded in the README; do not re-derive them.

The one to be most careful with is `--grid frames`. Switching `DUR_FRAMES` to `speech`
without refitting the constants is measured to be catastrophic — it charges a five-token
rival correctly on 37% of clips against the deployed 91% — so if that row looks reasonable
on this sample, be suspicious of the sample rather than reassured. It needs the refit from
step 4 alongside it.

To turn a winner on: edit the constant in **both** `scripts/constrained_ctc.py` and
`frontend/nanostt.js` (or `frontend/app.js` for `MARGIN_SIGMAS` / `FIELD_LOCAL`), and
update the pins in `tests/test_scripts.py` that currently hold them at zero. The pins exist
so this is a deliberate act.

---

## 4. The frame-unit question — answered, do not re-run

**The unit hypothesis is dead and the refit is not an improvement.** Refitting on all
63,114 distillation passes gives `93.38 + 1.6238 × tokens` at sd 27.11. The slope matches
the deployed 1.8794; only the intercept and sd differ, and a unit error would have moved
the slope. Swept against the clips and negatives the refit loses at every λ from 0.1 to
1.2, because it expects 116 frames for a line that occupies 76 — so the true line already
looks too short and a shorter rival looks *less* wrong. The sign of the term flips.

The corpus is not the population: it is FLEURS sentences and full TTS renderings, and the
app asks for `Bonġu!`. The deployed constants describe short prompted phrases. Their
disagreement with the corpus is not an error to fix, and `--grid calib` re-measures this
in seconds if anyone doubts it.

Consequence: the `frames` row in step 3 no longer needs a refit before being trusted.

---

## 5. More real speech — the largest measured effect

Real speech took the student from 102.5% to 74.6% fWER, which no amount of capacity did.
A transcript is not needed, because the teacher labels whatever it is handed:

```bash
mkdir -p data/corpora/voxpopuli        # drop any Maltese audio in, any format
.venv/bin/python scripts/distill_stt.py teacher --sources corpus --corpus-name voxpopuli \
    --real-limit 2000 --shard vox
```

Start with `--real-limit` small. VoxPopuli holds about 9,100 hours of unlabelled Maltese
and the teacher pass is the expensive part of this whole project; a first run should tell
you the throughput before it tells you the accuracy. MASRI-HEADSET and MASRI-TUBE add 25
real speakers close-mic and at two metres — read the research/academic licence before
shipping anything trained on them.

---

## 6. The training runs, in order

Each ends in an exported `model.onnx` scored by `constrained_ctc.py`, so each is a number
rather than an opinion. Run them in this order, because they compound and you want to know
which one paid.

**6a. Variance first.** The README's own finding is that two checkpoints of identical
architecture score 29% and 83% on the learner's voice, so between-run variance dwarfs
everything else. Reduce that before measuring anything.

```bash
.venv/bin/python scripts/distill_stt.py train --ema-decay 0.999 --select rank --tag ema
.venv/bin/python scripts/distill_stt.py export --tag ema --ema
```

Watch the printed `rank-1` against `dev kd`. If they disagree — dev loss improving while
rank-1 does not — that gap is the finding, and it means selecting on loss was picking the
wrong checkpoint all along.

**6b. Deep supervision.** Free at inference; the head is dropped at export.

```bash
.venv/bin/python scripts/distill_stt.py train --aux-at 5 --aux-weight 0.3 \
    --ema-decay 0.999 --select rank --tag aux
```

**6c. Clean the labels.** Removes the teacher's own 5.3% error from the half where the
line is known. Destructive to the shard, so copy it first.

```bash
cp -r data/distill data/distill.raw
.venv/bin/python scripts/distill_stt.py constrain --shard tts --constrain-alpha 0.5
.venv/bin/python scripts/distill_stt.py train --ema-decay 0.999 --select rank --tag kd
```

**6d. Gemination.** The named blocker: `kolox` scores 1.02 against `kollox` and is
accepted. The second command is not optional — excising frames leaves the teacher's
remaining posteriors describing audio it never saw.

```bash
.venv/bin/python scripts/distill_stt.py degeminate --shard tts
.venv/bin/python scripts/distill_stt.py constrain  --shard tts_degem
.venv/bin/python scripts/distill_stt.py train --ema-decay 0.999 --select rank --tag gem
```

Then check the thing it was for, specifically — whether `kolox` still scores 1.02 on audio
of `kolox`. The aggregate app score will barely move either way; that one pair is the
result.

---

## The ceiling none of this reaches

Worth holding in view before spending GPU time. On 16 clean clips of a learner's Maltese,
the **201 MB teacher** marks 25% of correct answers correct. The 2.1 MB student manages 0%.
Every Maltese recogniser available is trained on native speech, and a learner's `għ`,
geminates and `q` are not what any of them expect.

Steps 5 and 6 push toward that ceiling. They do not move it. The only thing that moves it
is real L2 Maltese — recordings from the people actually using the app — which is a consent
and product question before it is a modelling one, and probably worth deciding before
committing to the expensive half of this list.

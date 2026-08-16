---
title: Speak Maltese
emoji: 🇲🇹
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Learn Maltese by talking — scripted scenes, speech, spaced repetition
---

# Nitkellmu — Speak Maltese

Learn Maltese by having conversations. The app speaks, listens, shows the written
Maltese, corrects you gently and schedules what you got wrong.

**Your progress stays in your browser.** Reviews, schedules and streaks live in
IndexedDB on your own device — this Space stores nothing about you, and everyone
who opens it starts their own deck. Use the export button in Settings to move
progress between devices or keep a copy.

## What it does here

* **35 scripted scenes** covering every word in the deck — a café, a pharmacy, a
  bus, getting lost, booking a table. Every line is authored and pre-voiced, so
  the Maltese you hear is correct by construction.
* **Speech recognition** by a Maltese wav2vec2 CTC model, running on this Space's
  CPU. It is baked into the image, so the only wait is the first load after a
  restart — the app shows you that wait rather than pretending to be ready.
* **FSRS-5 spaced repetition**, running in your browser.

## The first visit is the slow one

Free Spaces sleep when idle and reload the model on waking. That takes up to a
minute, and the startup screen tells you where it has got to. Reading, listening
and typing work throughout; only speaking needs the recogniser.

## Licence

MIT, except: `data/frequency_mt.tsv` is derived from
[MLRS Korpus Malti](https://mlrs.research.um.edu.mt/) and carries **CC BY-NC-SA 4.0**;
imported example sentences come from [Tatoeba](https://tatoeba.org) under
**CC BY 2.0 FR**. See `LICENSE`.

Source: <https://github.com/silas-dsc/speak-maltese>

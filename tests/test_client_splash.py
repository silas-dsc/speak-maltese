"""The startup screen, against a recogniser that lives on another host.

A phone cannot hold the 200MB model — an iPhone SE gives a page 250-350MB and the
tab is reloaded rather than slowed — so the static build sends utterances to a
deployment of this same app instead. That host is free and sleeps when idle, which
moves the wait rather than removing it: the first utterance would pay tens of
seconds for a container waking up and a model loading, arriving as a mic button
that does nothing.

So the startup screen wakes it, behind the progress bar it already shows. The three
things that must hold are all about not trading one broken state for another:

  * a host that is merely asleep is waited for, through the seconds where it
    answers nothing at all;
  * a host that never answers must not keep the app shut — reading, listening and
    typing never needed it;
  * and where there is a host, the device must never start the download. That is
    the whole point, and a percentage bar creeping over a model fetch is what the
    crash looked like.

Tested here rather than in a browser because all three are about clocks and
failures, and both are easier to hold still with a stub than to provoke for real.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node is not installed")

DRIVER = r"""
import * as splash from '../frontend/splash.js';

/* A clock that costs nothing. The waits under test are tens of seconds long, and
   `setTimeout` is the only thing that consumes them, so time is whatever the
   scheduled delays add up to. */
let now = 0;
Date.now = () => now;
globalThis.setTimeout = (fn, ms = 0) => { now += ms; queueMicrotask(fn); return 0; };
globalThis.clearTimeout = () => {};

/* Enough DOM for the bar to be painted into. The screen's shape is a browser's
   business; what is tested here is which waits happen and in which order. */
function stubDocument() {
  const painted = [];
  const leaf = () => ({ style: {}, textContent: '', hidden: false });
  globalThis.document = {
    createElement: () => {
      const bar = leaf();
      const step = leaf();
      const note = leaf();
      return {
        className: '',
        innerHTML: '',
        classList: { add: () => {} },
        remove: () => {},
        querySelector: (sel) => {
          if (sel === '.splash-step') return step;
          if (sel === '.splash-note') return note;
          return bar;
        },
        // Read after every paint: the label is the promise being made to the
        // learner, so a wait with the wrong one is a wait that lies.
        _read: () => ({ width: bar.style.width, label: step.textContent }),
      };
    },
    body: { append: (el) => painted.push(el) },
  };
  return painted;
}

/** A static build's files, plus a host that answers `wakesAfter` polls late.
    `wakesAfter: null` never answers at all — asleep, or gone. */
function stubFetch({ base, wakesAfter }) {
  const seen = [];
  let polls = 0;
  const json = (body) => ({ ok: true, json: async () => body });
  globalThis.fetch = async (url) => {
    seen.push(url);
    if (url === 'api/bootstrap.json') {
      return json({
        static: true,
        stt_base: base,
        models_base: '/models/',
        capabilities: { stt: base ? ['remote'] : [] },
        defaults: {},
      });
    }
    if (url === 'api/deck.json') return json({ cards: [] });
    if (url === 'api/dialogues.json') return json([]);
    if (url === 'audio/index.json') return json({});
    if (base && url === `${base}/api/health`) {
      polls += 1;
      // A booting Space answers nothing, and the holding page it serves meanwhile
      // has no CORS headers — so this is what one looks like from the page.
      if (wakesAfter === null || polls <= wakesAfter) throw new TypeError('Failed to fetch');
      return json({ ready: true, warming: false, stt: ['wav2vec2'], tts: [] });
    }
    throw new Error(`unexpected fetch: ${url}`);
  };
  return { seen, polls: () => polls };
}

async function scenario({ base, wakesAfter = 0 }) {
  now = 0;
  const painted = stubDocument();
  const net = stubFetch({ base, wakesAfter });
  let modelAsked = false;
  let notice = '';
  const boot = await splash.run({
    onDeck: () => {},
    onStatic: () => {},
    onModel: async () => { modelAsked = true; return false; },
    onNotice: (msg) => { notice = msg; },
  });
  const labels = painted.map((el) => el._read().label);
  return {
    opened: !!boot?.static,
    modelAsked,
    notice,
    polls: net.polls(),
    healthUrl: net.seen.find((u) => u.endsWith('/api/health')) || '',
    elapsed: now,
    lastLabel: labels[labels.length - 1] || '',
  };
}

const out = {};
out.warm = await scenario({ base: 'https://space.example', wakesAfter: 0 });
out.asleep = await scenario({ base: 'https://space.example', wakesAfter: 6 });
out.gone = await scenario({ base: 'https://space.example', wakesAfter: null });
out.onDevice = await scenario({ base: '' });

console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def result():
    driver = ROOT / "tests" / "_splash_driver.mjs"
    driver.write_text(DRIVER, encoding="utf-8")
    try:
        proc = subprocess.run([node, str(driver)], cwd=ROOT,
                              capture_output=True, text=True, timeout=30)
    finally:
        driver.unlink(missing_ok=True)
    if proc.returncode != 0:
        pytest.fail(f"node driver failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


def test_a_warm_host_is_asked_once_and_costs_nothing(result):
    """The poll is also the pre-warm, so it happens on every load. On the second
    visit it is one request against a running container and must not be felt."""
    warm = result["warm"]
    assert warm["opened"], "the app did not start"
    assert warm["polls"] == 1, f"{warm['polls']} polls against a host already up"
    assert warm["healthUrl"] == "https://space.example/api/health", warm["healthUrl"]
    # Only the closing flourish and the fade, both fixed: no waiting happened.
    assert warm["elapsed"] < 2000, f"{warm['elapsed']}ms spent on a warm host"


def test_a_sleeping_host_is_waited_for(result):
    """The container is asleep and answering nothing. That is the wait worth having
    — it is the one the first utterance would otherwise pay — so the door stays
    shut, and nothing is said to the learner afterwards because nothing went wrong."""
    asleep = result["asleep"]
    assert asleep["opened"]
    assert asleep["polls"] == 7, f"gave up after {asleep['polls']} polls"
    assert asleep["notice"] == "", f"warned about a wait that worked: {asleep['notice']}"


def test_a_host_that_never_answers_still_lets_the_learner_in(result):
    """The failure that matters: an app that will not open. Reading, listening,
    reviewing and typing never needed the recogniser, so the wait is bounded and
    the learner is told why speaking is not ready yet."""
    gone = result["gone"]
    assert gone["opened"], "a sleeping host kept the app shut"
    assert 30000 <= gone["elapsed"] < 45000, f"waited {gone['elapsed']}ms"
    assert "still waking" in gone["notice"], gone["notice"]


def test_a_named_host_means_the_model_is_never_fetched(result):
    """Not a fallback. The device asking for the model is exactly what killed the
    tab, so where there is a host to send audio to, that path is not offered at
    all — and where there is not, it still is."""
    assert not result["warm"]["modelAsked"]
    assert not result["asleep"]["modelAsked"]
    assert not result["gone"]["modelAsked"], "fell back to the 200MB download"
    assert result["onDevice"]["modelAsked"], "with no host, the device must try"
    assert result["onDevice"]["polls"] == 0, "polled a host that was never named"

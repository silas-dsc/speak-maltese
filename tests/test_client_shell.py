"""The shell on a phone: what is on screen, and how you get anywhere from it.

Two things, both of them about an iPhone SE. On a narrow or a short screen the tab
row is gone and the sheet is the *only* way to the other four views, so the sheet
has to be complete — a view added to the tabs and not to the sheet is a view a phone
cannot reach. And the conversation shows one exchange at a time, which is the rest
of this file.

The conversation used to be a scrolling log, and on an iPhone SE it had about 280px
to live in. A turn is 60-90px of that, so by the third one the question being
answered had gone off the top — reading the prompt meant scrolling up, reading the
marking meant scrolling back down, and the app was unusable in exactly the moment
it was being used. So the drill shows one exchange at a time, with the rest a tap
away.

Which turns count as "one exchange" is the whole of it, and it is not obvious. The
shape is prompt → answer → reply, where the reply is a tutor turn carrying a
verdict and a prompt is a tutor turn without one — so the rule is to take turns
from the end until a tutor turn with no verdict, and take that one too.

The tempting rule is "keep the last three". It is right until a scene answers one
prompt twice — which is the normal case, because a wrong answer does not advance —
and then it silently hides the question the learner is being marked against. That
is a bug you would find by using the app on a phone and not by reading the code,
which is why the rule is a named pure function and why this test exists.

`currentExchange` is lifted out of app.js and run, the same way test_api.py lifts
`routeFor` out of the service worker: the rest of that file needs a DOM, and the
rule does not.
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
import { readFileSync } from 'node:fs';

const src = readFileSync('frontend/app.js', 'utf8');
const start = src.indexOf('function currentExchange');
if (start < 0) throw new Error('app.js has no currentExchange to test');
const end = src.indexOf('\n}', start) + 2;
const currentExchange = new Function(
  `${src.slice(start, end)}; return currentExchange;`)();

const cases = JSON.parse(process.argv[2]);
console.log(JSON.stringify(cases.map((kinds) => currentExchange(kinds))));
"""


def exchanges(cases: list[list[str]]) -> list[list[int]]:
    driver = ROOT / "tests" / "_focus_driver.mjs"
    driver.write_text(DRIVER, encoding="utf-8")
    try:
        proc = subprocess.run([node, str(driver), json.dumps(cases)], cwd=ROOT,
                              capture_output=True, text=True, timeout=30)
    finally:
        driver.unlink(missing_ok=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_one_exchange_is_the_prompt_the_answer_and_the_marking():
    """Four conversations, at the four points a learner is ever looking at one."""
    got = exchanges([
        # A scene just started: the prompt is all there is.
        ["prompt"],
        # Mid-answer, before the reply lands.
        ["prompt", "answer"],
        # The whole exchange — and this is the one that matters, because the marking
        # is only useful next to the thing it marks.
        ["prompt", "answer", "reply"],
        # A second scene's first prompt: everything before it is history.
        ["prompt", "answer", "reply", "prompt"],
    ])
    assert got == [[0], [0, 1], [0, 1, 2], [3]]


def test_a_second_try_at_the_same_prompt_keeps_the_prompt_on_screen():
    """A wrong answer does not advance the scene, so one prompt collects two answers
    and two replies. "Keep the last three" would show the retry and its marking with
    the question gone — the learner would be told they were wrong about something
    they could no longer read."""
    got = exchanges([["prompt", "answer", "reply", "answer", "reply"]])
    assert got == [[0, 1, 2, 3, 4]]

    # And a third try, which is where MAX_ATTEMPTS waves them on.
    deep = ["prompt"] + ["answer", "reply"] * 3
    assert exchanges([deep]) == [list(range(len(deep)))]


def test_the_end_of_a_scene_is_a_screen_of_its_own():
    """The run summary is tagged `prompt`, so it starts an exchange rather than
    arriving underneath the last answer. Reaching it should leave the tally on
    screen and nothing else."""
    got = exchanges([["prompt", "answer", "reply", "prompt"]])
    assert got == [[3]]


def test_an_empty_conversation_hides_nothing():
    """`applyFocus` runs on a view change and at boot, when there may be no turns at
    all. An empty list must come back empty rather than reaching for index -1."""
    assert exchanges([[]]) == [[]]


def test_the_drill_starts_focused_on_a_phone_and_not_on_a_laptop():
    """The default is a media query, and it is the reason the SE gets one exchange
    while a desktop gets the transcript. A learner who has said which they want
    keeps it — `focusTouched` — so only the untouched default may move."""
    app = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "const ROOM_FOR_TRANSCRIPT = window.matchMedia(" in app
    assert "(max-width: 640px), (max-height: 560px)" in app
    assert "if (!focusTouched) setFocus(e.matches)" in app

    css = (ROOT / "frontend" / "style.css").read_text(encoding="utf-8")
    # The hiding itself is CSS: the JS only marks which turns are the exchange.
    assert ".chat.is-focus .turn:not(.is-current) { display: none; }" in css


def test_every_view_is_reachable_from_the_phone_menu():
    """On a phone `.tabs` is `display: none`, so the sheet is the only navigation
    there is. A fifth view added to the tab row and forgotten in the sheet is not a
    cosmetic omission — it is a screen no phone can open, and nothing else in the
    suite would notice."""
    import re

    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    tabs = set(re.findall(r'class="tab[^"]*" data-view="([a-z]+)"', html))
    sheet = set(re.findall(r'class="sheet-item[^"]*" data-view="([a-z]+)"', html))

    assert tabs, "no tabs found — did the markup change shape?"
    assert tabs == sheet, f"only in the tab row: {tabs - sheet}; only in the sheet: {sheet - tabs}"

    # Every one of them is a section that exists.
    for view in sorted(tabs):
        assert f'id="view-{view}"' in html, f"{view} has a button and no screen"

    css = (ROOT / "frontend" / "style.css").read_text(encoding="utf-8")
    # The phone block, up to the `}` in the first column that closes it. Narrow *or*
    # short: a phone in landscape is 667x375, wide enough for the tab row and with no
    # height to spare for it — the same query focus mode defaults on.
    phone = css.split("@media (max-width: 640px), (max-height: 560px) {", 1)[1].split("\n}", 1)[0]
    assert ".tabs { display: none; }" in phone, \
        "the tab row is what the sheet replaces; if it is still shown, the sheet is clutter"
    assert ".nav-btn { display: flex; }" in phone, "…and something has to open the sheet"


def test_the_conversation_pays_for_almost_no_chrome():
    """The complaint was structural: a header, a wrapped tab row, a scene picker, two
    buttons, a line of progress and a picture — about 265px of an SE's ~560px of
    usable height, before a single word of Maltese.

    All of it is gone or moved: the picker and its buttons to the scenes screen and
    the sheet, the tab row to the sheet, the picture behind the conversation. What is
    left above the chat is one 34px row. This asserts the *absence*, which is the
    only way to keep it — furniture comes back one well-meaning element at a time."""
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    drill = html.split('id="view-drill"')[1].split("</section>")[0]

    for gone in ("<select", 'class="scenario-bar"', 'id="drillProgress"'):
        assert gone not in drill, f"{gone} is back above the conversation"

    # The scene name is the way to the path, and the actions are behind the ⋯.
    for wanted in ('id="drillScene"', 'id="drillMore"', 'id="drillStep"'):
        assert wanted in drill, f"missing: {wanted}"

    # And the actions that used to be buttons on that bar still exist somewhere.
    for action in ('id="sheetRestart"', 'id="sheetNext"', 'id="sheetTranscript"'):
        assert action in html, f"{action} was removed rather than moved"

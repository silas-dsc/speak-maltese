"""The shell on a phone: what is on screen, and how you get anywhere from it.

Below 640px — and on anything shorter than 560px, which is a phone in landscape —
the tab row is gone and the bottom sheet is the *only* way to the other four views.
So the sheet has to be complete: a view added to the tabs and forgotten in the sheet
is not a cosmetic omission, it is a screen no phone can open, and nothing else in the
suite would notice.

The rest of this file is about what the conversation does *not* pay for. The
complaint that started it was an iPhone SE: a header, a wrapped tab row, a scene
picker, two buttons, a line of progress and an illustration — about 265px of a screen
with ~560px to give once Safari has taken its share, before a word of Maltese.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_every_view_is_reachable_from_the_phone_menu():
    """On a phone `.tabs` is `display: none`, so the sheet is the only navigation
    there is."""
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
    # height to spare for it.
    phone = css.split("@media (max-width: 640px), (max-height: 560px) {", 1)[1].split("\n}", 1)[0]
    assert ".tabs { display: none; }" in phone, \
        "the tab row is what the sheet replaces; if it is still shown, the sheet is clutter"
    assert ".nav-btn { display: flex; }" in phone, "…and something has to open the sheet"


def test_the_conversation_pays_for_almost_no_chrome():
    """All of the furniture is gone or moved: the picker and its buttons to the
    scenes screen and the sheet, the tab row to the sheet, the scene illustration to
    the turns themselves. What is left above the chat is one 34px row.

    This asserts the *absence*, which is the only way to keep it — furniture comes
    back one well-meaning element at a time."""
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    drill = html.split('id="view-drill"')[1].split("</section>")[0]

    for gone in ("<select", 'class="scenario-bar"', 'id="drillProgress"',
                 # The 2:1 banner: every turn carries its own picture now, so a scene
                 # image above them was the same scene said twice.
                 'class="scene-hero"', 'id="sceneImg"',
                 # …and the toggle that chose between showing one exchange and all of
                 # them, which is a control for two views of a screen needing one.
                 'id="transcriptToggle"', 'class="chat is-focus"'):
        assert gone not in drill, f"{gone} is back above the conversation"

    # The scene name is the way to the path, and the actions are behind the ⋯.
    for wanted in ('id="drillScene"', 'id="drillMore"', 'id="drillStep"',
                   'class="chat" id="drillChat"'):
        assert wanted in drill, f"missing: {wanted}"

    # And the actions that used to be buttons on that bar still exist somewhere.
    for action in ('id="sheetRestart"', 'id="sheetNext"'):
        assert action in html, f"{action} was removed rather than moved"


def test_nothing_is_left_of_focus_mode():
    """The conversation showed one exchange at a time for a while, with the rest
    behind a toggle. It is every turn now, and half a removal is worse than none:
    `applyFocus` was called from four places, the turns carried a `data-kind` for it
    to slice on, and the CSS hid anything without `.is-current` — so a survivor here
    is a conversation with turns that never appear."""
    js = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    for gone in ("applyFocus", "focusMode", "currentExchange", "setFocus",
                 "dataset.kind", "transcriptToggle", "sheetTranscript",
                 "showSceneImage"):
        assert gone not in js, f"app.js still refers to {gone}"

    css = (ROOT / "frontend" / "style.css").read_text(encoding="utf-8")
    for gone in (".is-focus", ".transcript-toggle", ".scene-hero"):
        assert gone not in css, f"style.css still styles {gone}"


def test_a_turn_shows_its_own_picture_beside_the_question():
    """One picture per turn rather than one per scene, square, and the same height as
    the bubble it belongs to. Sized from the row's width: driving it the other way
    round — stretch to the bubble, square from that — fed back on itself, because a
    wide picture leaves a narrow bubble and a narrow bubble is a taller one."""
    js = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    assert 'src="img/turn-${escapeHtml(art)}.webp"' in js
    # Named after the turn and stored on it, so a restored conversation comes back
    # with the same pictures beside the same questions.
    assert "art: `${drill.dialogue}-${node.node}`," in js
    # A missing picture is a question, not a broken-image icon.
    assert "img.onerror = () => { img.remove(); el.classList.remove('has-art'); };" in js

    css = (ROOT / "frontend" / "style.css").read_text(encoding="utf-8")
    art = css.split(".turn-art {", 1)[1].split("}", 1)[0]
    assert "aspect-ratio: 1 / 1;" in art, "the picture is no longer square"
    assert "width: clamp(" in art, "sized from the bubble again — see the docstring"


def test_the_answer_can_be_seen_and_heard_before_it_is_given():
    """**Show me**, and the honesty that has to come with it: a turn whose answer was
    on screen a moment earlier is not evidence of recall, so it is not filed into the
    review deck as a phrase the learner produced. FSRS would schedule it as known."""
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    for wanted in ('id="drillReveal"', 'id="drillAnswerMt"', 'id="drillAnswerPlay"',
                   'id="drillAnswerSlow"'):
        assert wanted in html, f"missing: {wanted}"

    js = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "drill.peeked = true;" in js, "a peek is not recorded"
    assert "!r.moved_on && !drill.peeked" in js, \
        "a peeked answer is being filed into the review deck"
    # Cleared on every new node, or one peek would taint the rest of the scene.
    assert "function hideAnswer() {\n  drill.peeked = false;" in js
    assert "  hideAnswer();\n}" in js, "installDrillNode no longer resets the cue"


def test_an_open_question_shows_the_patterns_not_a_gendered_looking_example():
    """The examples carry a name, and a learner who sees `Jisimni Silas` for one pattern
    and `Jien Sally` for another can reasonably conclude that `Jien` is the women's
    form. Maltese has plenty of masculine and feminine pairs, so it is a sensible
    inference — it is simply wrong here: `Jien …` and `Jisimni …` are interchangeable
    and the only thing that varies with the speaker is the name in the gap.

    So on an open question the patterns go first, together, equally weighted, and the
    example below is marked as an example."""
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert 'id="drillAnswerFrames"' in html

    js = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    # Only on a question whose answer is the learner's own.
    assert "const frames = drill.present.free ? (drill.present.frames || []) : [];" in js
    assert "'Either pattern — your own answer in the gap'" in js
    assert "'This pattern — your own answer in the gap'" in js
    # The example is labelled as one, so it is not read as the answer to repeat.
    assert "`e.g. ${answer.mt}`" in js

    css = (ROOT / "frontend" / "style.css").read_text(encoding="utf-8")
    frames = css.split(".drill-answer .frames {")[1].split("}")[0]
    assert "flex-basis: 100%" in frames, "the patterns share a line with the example"


def test_the_line_to_say_can_be_heard_on_its_own():
    """A near miss shows the line to say back. The bubble's own Play speaks the *reply*
    — `Kważi. Għid: …` — which buries the target inside a sentence and behind a word of
    Maltese the learner has just been told they got wrong. It was the one set response
    with no way to hear it by itself, which is what "some audio seems to be missing"
    was pointing at."""
    js = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "data-target-play" in js and "data-target-slow" in js
    assert "el.querySelector('[data-target-play]').onclick = () => speak(target.mt);" in js
    assert "speak(target.mt, { rate: 0.7 })" in js


def test_a_spoken_line_is_awaited_until_it_has_finished():
    """`speak()` resolved on `play()`, which returns as soon as playback *begins* — 10ms
    for a 2.59-second line, measured. So `await speak(reply)` in a drill turn waited for
    nothing, and 450ms later the next prompt called `speak()` again, which pauses
    whatever is playing. Every tutor reply longer than half a second was cut off
    mid-sentence, every turn, and that was most of "audio seems cut off".

    It has to settle on failure too, or a line with no file would hang the turn — and it
    has to settle when *superseded*, or the promise of the audio just paused is never
    resolved at all."""
    js = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    speak = js.split("function speak(text, { rate } = {}) {")[1].split("\n}")[0]
    assert "addEventListener('ended', settle, { once: true })" in speak
    assert "addEventListener('error', settle, { once: true })" in speak
    assert "if (currentDone) { currentDone(); currentDone = null; }" in speak, \
        "superseding audio leaves its promise hanging"
    assert ".catch((err) => {" in speak and "settle();" in speak

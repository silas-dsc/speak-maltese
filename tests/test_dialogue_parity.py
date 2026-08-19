"""The in-browser dialogue engine must grade exactly as the server one does.

Same reason as the FSRS and text parity tests: `frontend/dialogue.js` exists so the
static build can hold a conversation without a backend, and a translation that
drifts produces a learner stuck on a line the server would have accepted.

The inputs are all 334 recorded 4-bit transcripts, each replayed at the node it
belongs to, plus the awkward cases — silence, a free node, the quoted-foreign-word
frame, and the two-attempt backstop.
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

from backend import dialogue  # noqa: E402

node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node is not installed")

FIXTURE = ROOT / "tests" / "fixtures" / "q4_transcripts.tsv"

EDGE = [
    ("cafe", "c1", "", 0),
    ("cafe", "c1", "xi xi xi", 0),
    ("cafe", "c1", "xi xi xi", 2),           # backstop waves it through
    ("greet", "g1", "silas", 0),             # free node
    ("greet", "g1", " ", 0),                 # silence on a free node
    ("greet", "g1", "jien pietru", 0),       # open question, frame landed
    ("greet", "g1", "mela, jien pietru", 0),  # …behind a hesitation
    ("greet", "g1", "jien", 0),              # frame with nothing in it
    ("greet", "g2", "mill-awstralja", 0),    # the fused article, split or not
    ("greet", "g2", "mill awstralja", 0),
    ("family", "f4", "għandi ħamsa u tletin sena", 0),   # anchored at both ends
    ("family", "f4", "ħamsa u tletin sena", 0),
    ("family", "f4", "dak sigriet!", 0),     # a listed answer outside the frame
    ("home", "h3", "tliet kmamar", 0),       # anchored at the end only
    ("home", "h3", "tlieta", 0),
    ("phone", "x1", "bonġu, jien Pietru", 0),  # a greeting in front of the frame
    ("feelings", "z1", "ninsab imdejjaq", 0),  # near a listed answer off the frame
    ("greet", "g1", "a", 0),                 # too short: the target has to be shown
    ("greet", "g3", "Noqhod il-Belt", 0),    # how the recogniser spells għ
    ("routine", "y2", "Tibda fid-disgħa", 0),  # the question echoed, not answered
    ("family", "f4", "Għandi sigriet", 0),   # near an answer that is not the frame
    ("likes", "l1", "Nħobb il-ħut", 0),      # a first letter the recogniser dropped
    ("greet", "g2", "Mir-Russja", 0),        # minn fused with an assimilating article
    ("people", "o2", "Huma għalliema", 0),   # they, not he
    ("people", "o2", "X'jaħdem hu?", 0),     # the question with its x' still on it
    ("family", "f2", "ma għandix aħwa", 0),  # the negation, not the frame
    ("stuck", "s3", "kif tgħidx xi s bil-malti", 0),
    ("stuck", "s3", "xi xi xi", 0),
    ("cafe", "c1", "nixtieq kafè, jekk jogħġbok.", 0),
    # Accepted on a lead rather than on a score: under 0.86, and nearer to the line the
    # node wanted than to any of the 377 the script accepts anywhere. The weighted sound
    # distance and the rival scan are both new, both in two languages, and both land on
    # a boundary — so they are replayed here rather than trusted.
    ("keys", "r2", "yien bilats", 0),
    ("keys", "r1", "ninsa olilo illum", 0),
    ("doctor", "h1", "għandi uġigiħ tia' toniku", 0),
    ("outing", "m1", "ix-xemix qeigħa tiddii", 0),
    ("restaurant", "r3", "nieħu l-iħuit jiekk joigħġoik", 0),
    # …and one that clears the floor and is still turned away, because `Ma nafx` and
    # `Ma nifhimx` are 0.03 apart on it and that is not daylight.
    ("stuck", "s2", "ma nfix", 0),
]

DRIVER = r"""
import { readFileSync } from 'node:fs';
import * as d from '../frontend/dialogue.js';
d.load(JSON.parse(readFileSync('data/dialogues.json', 'utf8')));
const cases = JSON.parse(process.argv[2]);
console.log(JSON.stringify(cases.map(([did, nid, said, attempts]) => {
  const r = d.evaluate(did, nid, said, attempts);
  return { verdict: r.verdict, score: r.score, matched: r.matched_mt,
           advance: r.advance, moved_on: r.moved_on, frame_scored: r.frame_scored,
           on_lead: r.on_lead,
           say_this: r.say_this_mt || null,
           next: r.next ? r.next.node : null, finished: !!r.finished };
})));
"""


def cases() -> list[tuple[str, str, str, int]]:
    out = list(EDGE)
    for line in FIXTURE.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        did, nid, _expected, *heard = line.split("\t")
        out.append((did, nid, heard[0] if heard else "", 0))
    return out


@pytest.fixture(scope="module")
def both():
    data = cases()
    driver = ROOT / "tests" / "_dialogue_driver.mjs"
    driver.write_text(DRIVER, encoding="utf-8")
    try:
        proc = subprocess.run([node, str(driver), json.dumps(data, ensure_ascii=False)],
                              cwd=ROOT, capture_output=True, text=True, timeout=120)
    finally:
        driver.unlink(missing_ok=True)
    if proc.returncode != 0:
        pytest.fail(f"node driver failed:\n{proc.stderr}")
    js = json.loads(proc.stdout)

    py = []
    for did, nid, said, attempts in data:
        r = dialogue.evaluate(did, nid, said, attempts)
        py.append({"verdict": r["verdict"], "score": r["score"],
                   "matched": r["matched_mt"], "advance": r["advance"],
                   "moved_on": r["moved_on"], "frame_scored": r["frame_scored"],
                   "on_lead": r["on_lead"],
                   "say_this": r.get("say_this_mt"),
                   "next": (r.get("next") or {}).get("node") if r.get("next") else None,
                   "finished": bool(r.get("finished"))})
    return data, js, py


def test_every_turn_grades_identically(both):
    data, js, py = both
    bad = [(data[i], js[i], py[i]) for i in range(len(data)) if js[i] != py[i]]
    detail = "\n".join(f"  {c}\n    js {a}\n    py {b}" for c, a, b in bad[:5])
    assert not bad, f"{len(bad)} of {len(data)} turns differ:\n{detail}"


def test_the_lead_path_is_actually_reached_on_both_sides(both):
    """A field compared for equality is only compared where it varies. Both engines must
    have accepted something on a lead here, or `on_lead` is being checked as False
    against False and the new branch is untested in the browser."""
    _data, js, py = both
    assert sum(1 for r in js if r["on_lead"]) >= 4, "the browser engine never led"
    assert sum(1 for r in py if r["on_lead"]) >= 4, "the server engine never led"


def test_the_fixture_still_passes_in_the_browser_engine(both):
    """Not just 'the same as Python' — still actually correct. The 4-bit
    recogniser clears the server grader at 99.7%; the ported one must match."""
    data, js, _py = both
    replayed = [(c, r) for c, r in zip(data, js) if c not in EDGE]
    correct = sum(1 for _c, r in replayed if r["verdict"] == "correct")
    assert correct / len(replayed) >= 0.99, f"only {correct}/{len(replayed)}"


# ── The prompt side ────────────────────────────────────────────────────────────

PRESENT_DRIVER = r"""
import { readFileSync } from 'node:fs';
import * as d from '../frontend/dialogue.js';
d.load(JSON.parse(readFileSync('data/dialogues.json', 'utf8')));
const out = {};
for (const dia of d.all()) {
  for (const nid of Object.keys(dia.nodes || {})) {
    out[`${dia.id}/${nid}`] = d.present(dia.id, nid);
  }
}
console.log(JSON.stringify(out));
"""


def test_the_prompt_side_is_identical_too():
    """`evaluate` is the half that grades, and it is what the rest of this file
    replays. `present` is the half that decides what is on the screen, and it drifted
    the moment something was added to one copy and not the other — which is what
    happened when the drill learned to show the answer before asking for it:
    `answer` and `free` went into the Python and the browser's copy would have gone
    on rendering a "Show me" button that never had a line to show.

    Every node, both engines, whole object compared."""
    driver = ROOT / "tests" / "_present_driver.mjs"
    driver.write_text(PRESENT_DRIVER, encoding="utf-8")
    try:
        proc = subprocess.run([node, str(driver)], cwd=ROOT,
                              capture_output=True, text=True, timeout=60)
    finally:
        driver.unlink(missing_ok=True)
    assert proc.returncode == 0, proc.stderr
    js = json.loads(proc.stdout)

    py = {f"{d['id']}/{nid}": dialogue.present(d["id"], nid)
          for d in dialogue.all_dialogues() for nid in (d.get("nodes") or {})}

    assert set(js) == set(py), "the two engines do not agree on which nodes exist"
    assert py, "no nodes to compare"
    mismatched = {k: (py[k], js[k]) for k in py if py[k] != js[k]}
    assert not mismatched, (
        f"{len(mismatched)} nodes differ, first: "
        + repr(next(iter(mismatched.items()))))

    # And the field the reveal button depends on is actually populated.
    with_answer = [k for k, v in py.items() if (v.get("answer") or {}).get("mt")]
    assert len(with_answer) > len(py) * 0.9, (
        f"only {len(with_answer)} of {len(py)} nodes offer a model answer")

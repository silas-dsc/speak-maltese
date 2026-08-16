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
    ("stuck", "s3", "kif tgħidx xi s bil-malti", 0),
    ("stuck", "s3", "xi xi xi", 0),
    ("cafe", "c1", "nixtieq kafè, jekk jogħġbok.", 0),
]

DRIVER = r"""
import { readFileSync } from 'node:fs';
import * as d from '../frontend/dialogue.js';
d.load(JSON.parse(readFileSync('data/dialogues.json', 'utf8')));
const cases = JSON.parse(process.argv[2]);
console.log(JSON.stringify(cases.map(([did, nid, said, attempts]) => {
  const r = d.evaluate(did, nid, said, attempts);
  return { verdict: r.verdict, score: r.score, matched: r.matched_mt,
           advance: r.advance, moved_on: r.moved_on,
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
                   "moved_on": r["moved_on"],
                   "next": (r.get("next") or {}).get("node") if r.get("next") else None,
                   "finished": bool(r.get("finished"))})
    return data, js, py


def test_every_turn_grades_identically(both):
    data, js, py = both
    bad = [(data[i], js[i], py[i]) for i in range(len(data)) if js[i] != py[i]]
    detail = "\n".join(f"  {c}\n    js {a}\n    py {b}" for c, a, b in bad[:5])
    assert not bad, f"{len(bad)} of {len(data)} turns differ:\n{detail}"


def test_the_fixture_still_passes_in_the_browser_engine(both):
    """Not just 'the same as Python' — still actually correct. The 4-bit
    recogniser clears the server grader at 99.7%; the ported one must match."""
    data, js, _py = both
    replayed = [(c, r) for c, r in zip(data, js) if c not in EDGE]
    correct = sum(1 for _c, r in replayed if r["verdict"] == "correct")
    assert correct / len(replayed) >= 0.99, f"only {correct}/{len(replayed)}"

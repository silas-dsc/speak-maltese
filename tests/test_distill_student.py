"""The student's shape, and the promise that training extras never reach the device.

Two of the things `distill_stt.py` can now do are only worth doing if they cost the
shipped model nothing: an intermediate CTC head supervises the middle of the block stack,
and an EMA of the weights averages one training trajectory into one file. Both are
training-time only, and both would be quietly worthless — or worse — if they leaked into
`model.onnx`. "The head is dropped at export" is a claim about a traced graph, which is
exactly the kind of claim that stops being true when someone edits `forward`.

So this builds a shard, trains on it for two epochs, exports, and reads the ONNX back.
Small enough to run in a second; the point is the wiring, not the learning.

Nothing here needs the teacher, the corpus or a GPU — the shard is synthesised, so the
numbers are meaningless and the structure is not.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

WORDS = ("bonġu", "grazzi", "kollox", "jien", "malta", "irrid", "tabib", "hawn")


def load_distill():
    spec = importlib.util.spec_from_file_location(
        "distill_under_test", ROOT / "scripts" / "distill_stt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_shard(work: Path) -> int:
    """A `teacher` shard in miniature: mel, teacher posteriors, and an index."""
    work.mkdir(parents=True, exist_ok=True)
    vocab = {"<pad>": 0, "|": 1}
    for i, ch in enumerate("abcdefgħijklmnopqrstuvwxzżċġ"):
        vocab[ch] = i + 2
    (work / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")

    rng = np.random.default_rng(0)
    items, mels, posts = [], [], []
    for _ in range(48):
        text = " ".join(rng.choice(WORDS, size=int(rng.integers(2, 4))))
        ids = [vocab["|" if c == " " else c] for c in text
               if ("|" if c == " " else c) in vocab]
        frames = int(len(ids) * 2 + rng.integers(6, 14))
        mels.append(rng.normal(0, 1, (frames * 2, 64)).astype(np.float16))
        # Blank-heavy, the way a real CTC posterior is.
        logits = rng.normal(0, 1, (frames, len(vocab)))
        logits[:, 0] += 2.0
        logits -= np.log(np.exp(logits).sum(-1, keepdims=True))
        posts.append(logits.astype(np.float16))
        items.append({"text": text, "source": "tts", "augment": "identity",
                      "frames": frames})

    np.save(work / "mel_tts.npy", np.concatenate(mels))
    np.save(work / "post_tts.npy", np.concatenate(posts))
    (work / "index_tts.json").write_text(
        json.dumps({"vocab_size": len(vocab), "n_mels": 64, "items": items}),
        encoding="utf-8")
    return len(vocab)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    work = tmp_path_factory.mktemp("distill") / "work"
    module = load_distill()
    module.WORK = work
    write_shard(work)
    rc = module.stage_train(width=32, blocks=4, kernel=5, epochs=2, batch=4, lr=1e-3,
                            kd_weight=0.9, tag="t", aux_at=2, aux_weight=0.3,
                            ema_decay=0.9, select="rank", select_n=8, select_field=6)
    assert rc == 0
    return module, work


def test_the_auxiliary_head_exists_only_when_asked(trained):
    module, _work = trained
    assert module.build_student(30, 32, 4, 5).aux_head is None
    assert module.build_student(30, 32, 4, 5, aux_at=0).aux_head is None
    assert module.build_student(30, 32, 4, 5, aux_at=2).aux_head is not None
    # Past the end of the stack is a no-op rather than a crash.
    assert module.build_student(30, 32, 4, 5, aux_at=99).aux_head is None


def test_the_auxiliary_head_costs_the_shipped_model_nothing(trained):
    """It adds parameters while training and none of them are exported, so the two
    models have to agree on the count that matters."""
    module, _work = trained
    plain = module.build_student(30, 32, 4, 5)
    withaux = module.build_student(30, 32, 4, 5, aux_at=2)
    extra = sum(p.numel() for p in withaux.aux_head.parameters())
    assert module.param_count(withaux) == module.param_count(plain) + extra
    # `forward` must be the plain path in both, or the trace picks up the extra head.
    x = torch.zeros(1, 64, 60)
    assert plain(x).shape == withaux(x).shape
    main, aux = withaux.forward_aux(x)
    assert main.shape == aux.shape


def test_the_checkpoint_records_how_it_was_chosen(trained):
    module, work = trained
    ckpt = torch.load(work / "t" / "student.pt", map_location="cpu",
                      weights_only=False)
    assert ckpt["aux_at"] == 2
    assert "rank1" in ckpt and 0.0 <= ckpt["rank1"] <= 1.0
    assert "ema" in ckpt, "EMA weights were requested and not saved"
    assert any(k.startswith("aux_head.") for k in ckpt["state"])


def test_the_averaged_weights_are_actually_different(trained):
    """An EMA that tracked the raw weights exactly would be a no-op wearing a name."""
    module, work = trained
    ckpt = torch.load(work / "t" / "student.pt", map_location="cpu",
                      weights_only=False)
    moved = [k for k, v in ckpt["ema"].items()
             if v.dtype.is_floating_point
             and not torch.allclose(v, ckpt["state"][k].float(), atol=1e-7)]
    assert moved, "the EMA never diverged from the live weights"


def test_the_export_drops_the_auxiliary_head(trained):
    """The claim this whole file exists for: it is not in the graph."""
    onnx = pytest.importorskip("onnx")
    module, work = trained
    assert module.stage_export("t") == 0
    graph = onnx.load(str(work / "t" / "onnx" / "model.onnx")).graph

    assert not [i.name for i in graph.initializer if "aux" in i.name.lower()]
    ops = {}
    for node in graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    # 4 blocks: one stem conv, two convs a block, one output head. One ReLU a block
    # plus the stem's, one residual add a block. BatchNorm folds into the convs.
    assert ops["Conv"] == 1 + 2 * 4 + 1
    assert ops["Relu"] == 1 + 4
    assert ops["Add"] == 4
    assert ops["LogSoftmax"] == 1

    exported = sum(int(np.prod(i.dims)) for i in graph.initializer)
    plain = module.param_count(module.build_student(
        torch.load(work / "t" / "student.pt", map_location="cpu",
                   weights_only=False)["vocab_size"], 32, 4, 5))
    # Folding BatchNorm is the only reason these differ, and it differs by an amount
    # that can be predicted exactly: each of the five norms (the stem's and one a block)
    # gives up a weight and a bias of `width` and leaves behind a single conv bias of
    # `width`. Pinned rather than bounded, because an auxiliary head would be 990
    # parameters and any loose bound wide enough to allow folding would hide it.
    assert plain - exported == (4 + 1) * 32


def test_the_averaged_weights_can_be_exported(trained):
    module, work = trained
    assert module.stage_export("t", use_ema=True) == 0
    assert (work / "t" / "onnx" / "model.onnx").exists()


def test_exporting_averaged_weights_that_do_not_exist_fails_cleanly(trained,
                                                                    tmp_path):
    """Asking for an average a run never kept should say so, not export the raw weights
    while claiming otherwise."""
    module, work = trained
    ckpt = torch.load(work / "t" / "student.pt", map_location="cpu",
                      weights_only=False)
    ckpt.pop("ema")
    (work / "noema").mkdir(exist_ok=True)
    torch.save(ckpt, work / "noema" / "student.pt")
    assert module.stage_export("noema", use_ema=True) == 2


# ── Constraining the teacher to the text we already know ────────────────────

def test_the_ctc_posteriors_match_torch(trained):
    """The forward-backward has an exact independent reference and should be held to it.

    `ctc_loss`'s gradient with respect to the logits is `softmax - posterior`, so torch
    hands over the same occupancies by a completely different route. Anything that drifts
    here — a wrong skip condition, a backward recursion off by one frame — produces a
    distribution that still sums to 1 and still looks plausible, which is why the check
    is against torch rather than against invariants alone."""
    module, _work = trained
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(8):
        frames = int(rng.integers(10, 45))
        v_size, blank = 9, 0
        ids = [int(rng.integers(1, v_size)) for _ in range(int(rng.integers(2, 8)))]
        logits = torch.tensor(rng.normal(0, 1.5, (frames, v_size)),
                              requires_grad=True, dtype=torch.float64)
        lsm = torch.log_softmax(logits, dim=-1)
        loss = torch.nn.functional.ctc_loss(
            lsm[:, None, :], torch.tensor([ids]), torch.tensor([frames]),
            torch.tensor([len(ids)]), blank=blank, reduction="sum")
        loss.backward()
        reference = (torch.softmax(logits, dim=-1) - logits.grad).detach().numpy()
        mine = module.ctc_posteriors(lsm.detach().numpy(), ids, blank)
        worst = max(worst, float(np.abs(reference - mine).max()))
    # The returned array is float32, so its own epsilon is the floor here.
    assert worst < 1e-6, f"forward-backward disagrees with torch by {worst:.2e}"


def test_the_posteriors_spell_only_the_target(trained):
    module, _work = trained
    rng = np.random.default_rng(9)
    v_size, blank = 10, 0
    ids = [3, 5, 5, 2]
    logprobs = np.log(rng.dirichlet(np.ones(v_size), size=30)).astype(np.float32)
    gamma = module.ctc_posteriors(logprobs, ids, blank)
    assert np.allclose(gamma.sum(-1), 1.0, atol=1e-5)
    forbidden = [v for v in range(v_size) if v not in set(ids) | {blank}]
    assert gamma[:, forbidden].max() == 0.0, "mass landed on characters not in the line"
    # A geminate needs a blank between its halves, so `5` cannot own every frame it
    # touches — if it did, the extended-target skip rule would be wrong.
    assert gamma[:, blank].sum() > 0


def test_the_posteriors_decline_impossible_targets(trained):
    """More characters than frames has no alignment, and must say so rather than
    returning a normalised nonsense."""
    module, _work = trained
    logprobs = np.log(np.full((3, 6), 1 / 6)).astype(np.float32)
    assert module.ctc_posteriors(logprobs, [1, 2, 3, 4, 5], 0) is None
    assert module.ctc_posteriors(logprobs, [], 0) is None


def test_constraining_a_shard_moves_only_the_labelled_passes(tmp_path_factory):
    """The TTS half has a text label independent of the teacher and gets constrained;
    FLEURS is pseudo-labelled from the teacher's own argmax, so constraining it would
    sharpen its mistakes instead of removing them."""
    module = load_distill()
    work = tmp_path_factory.mktemp("constrain") / "work"
    module.WORK = work
    write_shard(work)

    meta_path = work / "index_tts.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["items"][0]["source"] = "fleurs"
    meta["items"][1]["text"] = ""
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    before = np.load(work / "post_tts.npy").copy()
    assert module.stage_constrain("tts", 0.5) == 0
    after = np.load(work / "post_tts.npy")

    frames = [it["frames"] for it in meta["items"]]
    starts = np.cumsum([0] + frames)
    # The two exempt passes are untouched; a labelled one has moved.
    for i in (0, 1):
        lo, hi = starts[i], starts[i + 1]
        assert np.array_equal(before[lo:hi], after[lo:hi]), f"pass {i} should be raw"
    lo, hi = starts[2], starts[3]
    assert not np.array_equal(before[lo:hi], after[lo:hi])
    # Still a distribution over every row it rewrote.
    rows = np.exp(after[lo:hi].astype(np.float32)).sum(-1)
    assert np.allclose(rows, 1.0, atol=2e-2)


def test_constraining_at_zero_changes_nothing_meaningful(tmp_path_factory):
    """alpha 0 is the identity, up to the float16 the shard stores."""
    module = load_distill()
    work = tmp_path_factory.mktemp("constrain0") / "work"
    module.WORK = work
    write_shard(work)
    before = np.load(work / "post_tts.npy").copy()
    assert module.stage_constrain("tts", 0.0) == 0
    after = np.load(work / "post_tts.npy")
    assert np.allclose(before.astype(np.float32), after.astype(np.float32), atol=2e-2)


# ── Cutting a geminate out of the audio ─────────────────────────────────────

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def aligned_shard(work: Path, texts=("kollox", "grazzi", "irrid", "malta"),
                  per: int = 3) -> dict:
    """A shard whose posteriors spell their labels on a known frame layout.

    The random shard the other tests use cannot exercise this: an alignment over noise
    never commits, so nothing would be cut and the test would pass by doing nothing.
    Here each character owns `per` frames with a blank between every pair — which is
    what CTC obliges between two halves of a doubled letter — so the frames that ought
    to be excised are known in advance.

    Every mel row is stamped with its own frame index, so which frames survived can be
    read straight back out of the derived shard."""
    work.mkdir(parents=True, exist_ok=True)
    vocab = {"<pad>": 0, "|": 1}
    for i, ch in enumerate(ALPHABET):
        vocab[ch] = i + 2
    (work / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")

    items, mels, posts, layouts = [], [], [], {}
    for text in texts:
        seq = []
        for j, ch in enumerate(text):
            if j:
                seq.append(0)
            seq += [vocab["|" if ch == " " else ch]] * per
        seq = [0, 0] + seq + [0, 0]
        frames = len(seq)
        logprobs = np.full((frames, len(vocab)), -12.0, dtype=np.float32)
        for t, sym in enumerate(seq):
            logprobs[t, sym] = 0.0
        logprobs -= np.log(np.exp(logprobs).sum(-1, keepdims=True))

        stamped = np.zeros((frames * 2, 64), dtype=np.float16)
        for t in range(frames):
            stamped[2 * t] = t
            stamped[2 * t + 1] = t
        mels.append(stamped)
        posts.append(logprobs.astype(np.float16))
        items.append({"text": text, "source": "tts", "augment": "identity",
                      "frames": frames})
        layouts[text] = seq

    np.save(work / "mel_a.npy", np.concatenate(mels))
    np.save(work / "post_a.npy", np.concatenate(posts))
    (work / "index_a.json").write_text(
        json.dumps({"vocab_size": len(vocab), "n_mels": 64, "items": items}),
        encoding="utf-8")
    return layouts


def test_the_geminate_finder_ignores_spaces_and_boundaries():
    module = load_distill()
    # space id 1 doubled is not a geminate; 3,3 and 7,7 are.
    assert module._geminate_positions([3, 3, 5, 1, 1, 7, 7], 1) == [1, 6]
    assert module._geminate_positions([1, 2, 3], 1) == []
    assert module._geminate_positions([], 1) == []


def test_degeminating_cuts_the_second_half_and_its_blank(tmp_path_factory):
    """The claim is specific, so the test is too: for `kollox` the frames removed must be
    the mandatory blank between the two `l`s plus the three the second `l` owns."""
    module = load_distill()
    work = tmp_path_factory.mktemp("degem") / "work"
    module.WORK = work
    layouts = aligned_shard(work)

    assert module.stage_degeminate("a") == 0
    meta = json.loads((work / "index_a_degem.json").read_text(encoding="utf-8"))
    mel = np.load(work / "mel_a_degem.npy")

    by_text = {}
    off = 0
    for item in meta["items"]:
        n = item["frames"]
        kept = [int(mel[(off + t) * 2, 0]) for t in range(n)]
        by_text[item["text"]] = kept
        off += n

    # `malta` has no geminate and must not appear at all.
    assert set(by_text) == {"kolox", "grazi", "irid"}

    seq = layouts["kollox"]
    dropped = sorted(set(range(len(seq))) - set(by_text["kolox"]))
    # Two `l`s at three frames each with one blank between: the cut is that blank and
    # the whole second run, and nothing else.
    assert len(dropped) == 4
    assert dropped == list(range(dropped[0], dropped[0] + 4))
    assert seq[dropped[0]] == 0, "the separating blank should lead the cut"
    assert len({seq[i] for i in dropped[1:]}) == 1, "the cut should be one letter's run"


def test_degeminating_keeps_mel_and_posteriors_in_step(tmp_path_factory):
    """Cutting one and not the other would leave every later frame supervised by the
    teacher's answer for a different moment — the exact failure time-domain augmentation
    was avoided for in the first place."""
    module = load_distill()
    work = tmp_path_factory.mktemp("degem2") / "work"
    module.WORK = work
    aligned_shard(work)
    assert module.stage_degeminate("a") == 0

    meta = json.loads((work / "index_a_degem.json").read_text(encoding="utf-8"))
    mel = np.load(work / "mel_a_degem.npy")
    post = np.load(work / "post_a_degem.npy")
    total = sum(i["frames"] for i in meta["items"])
    assert mel.shape[0] == total * 2, "two mel rows a frame"
    assert post.shape[0] == total

    vocab = json.loads((work / "vocab.json").read_text(encoding="utf-8"))
    off = 0
    for item in meta["items"]:
        n = item["frames"]
        window = np.asarray(post[off:off + n], dtype=np.float32)
        ids = [vocab["|" if c == " " else c] for c in item["text"]]
        # The degeminated label still has to be alignable against what is left.
        assert module.ctc_posteriors(window, ids, 0) is not None
        # And the frame stamps have to still be ascending — a scrambled gather here
        # would pass the length checks above and be silently wrong.
        stamps = [int(mel[(off + t) * 2, 0]) for t in range(n)]
        assert stamps == sorted(stamps)
        off += n


def test_degeminating_a_shard_with_nothing_to_cut_says_so(tmp_path_factory):
    module = load_distill()
    work = tmp_path_factory.mktemp("degem3") / "work"
    module.WORK = work
    aligned_shard(work, texts=("malta", "hawn"))
    assert module.stage_degeminate("a") == 1
    assert not (work / "index_a_degem.json").exists()

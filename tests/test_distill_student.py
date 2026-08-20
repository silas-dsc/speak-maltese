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

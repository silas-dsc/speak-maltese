"""Encode a line to model tokens, keeping the token strings alongside the ids.

`constrained_ctc.encode` returns ids only, which is all the ranking needs. Per-token
scoring has to say *which* sound scored badly, so the readable token must survive the
encoding rather than be recovered by guessing at the vocabulary afterwards. The mapping
itself is deliberately identical to that function's — including `space` being a vocabulary
key rather than an id — because a second, subtly different encoder would put the two
scores on different token sequences and nothing would say so."""

from __future__ import annotations


def encode_tokens(flat: str, vocab: dict[str, int],
                  space: str) -> tuple[list[int], list[str]]:
    ids: list[int] = []
    toks: list[str] = []
    for ch in flat:
        tok = space if ch == " " else ch
        if tok in vocab:
            ids.append(vocab[tok])
            toks.append("␣" if ch == " " else ch)
    return ids, toks

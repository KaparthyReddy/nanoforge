from __future__ import annotations

import numpy as np


def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, rng=None):
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")
    rng = rng or np.random.default_rng()
    idx = np.atleast_2d(np.asarray(idx, dtype=np.int64))
    block_size = model.config.block_size
    vocab = model.config.vocab_size
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block_size:])
        last = logits.data[:, -1, :]
        if temperature == 0:
            nxt = last.argmax(axis=-1)
        else:
            last = last / temperature
            if top_k is not None:
                kth = np.sort(last, axis=-1)[:, -min(top_k, vocab)][:, None]
                last = np.where(last < kth, -np.inf, last)
            z = last - last.max(axis=-1, keepdims=True)
            probs = np.exp(z)
            probs /= probs.sum(axis=-1, keepdims=True)
            nxt = np.array([rng.choice(vocab, p=p) for p in probs])
        idx = np.concatenate([idx, nxt[:, None]], axis=1)
    model.train(was_training)
    return idx

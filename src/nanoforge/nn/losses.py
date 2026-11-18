from __future__ import annotations

import numpy as np

from ..tensor import log_softmax


def mse_loss(pred, target):
    d = pred - target
    return (d * d).mean()


def cross_entropy(logits, targets):
    vocab = logits.shape[-1]
    flat = logits.reshape(-1, vocab)
    t = np.asarray(targets).reshape(-1).astype(np.int64)
    logp = log_softmax(flat, axis=-1)
    picked = logp[np.arange(t.shape[0]), t]
    return -picked.mean()

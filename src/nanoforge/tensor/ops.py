from __future__ import annotations

import math

import numpy as np

from .tensor import Tensor, _acc, _make, unbroadcast


def softmax(x, axis=-1):
    z = x.data - x.data.max(axis=axis, keepdims=True)
    e = np.exp(z)
    y = e / e.sum(axis=axis, keepdims=True)
    out = _make(y, (x,), "softmax")

    def _backward():
        g = out.grad
        _acc(x, y * (g - (g * y).sum(axis=axis, keepdims=True)))

    out._backward = _backward
    return out


def log_softmax(x, axis=-1):
    z = x.data - x.data.max(axis=axis, keepdims=True)
    y = z - np.log(np.exp(z).sum(axis=axis, keepdims=True))
    out = _make(y, (x,), "log_softmax")

    def _backward():
        g = out.grad
        _acc(x, g - np.exp(y) * g.sum(axis=axis, keepdims=True))

    out._backward = _backward
    return out


def gelu(x):
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + (c * (x + 0.044715 * (x**3))).tanh())


def masked_fill(x, mask, value):
    mask = np.asarray(mask, dtype=bool)
    out = _make(np.where(mask, value, x.data), (x,), "masked_fill")

    def _backward():
        _acc(x, unbroadcast(out.grad * (~mask), x.shape))

    out._backward = _backward
    return out


__all__ = ["Tensor", "gelu", "log_softmax", "masked_fill", "softmax"]

from __future__ import annotations

import numpy as np

from ..tensor import gelu
from .module import Module, Parameter


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, init_std=None, rng=None):
        rng = rng or np.random.default_rng()
        if init_std is None:
            bound = 1.0 / np.sqrt(in_features)
            w = rng.uniform(-bound, bound, (in_features, out_features))
        else:
            w = rng.normal(0.0, init_std, (in_features, out_features))
        self.weight = Parameter(w)
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x):
        y = x @ self.weight
        return y + self.bias if self.bias is not None else y


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, init_std=0.02, rng=None):
        rng = rng or np.random.default_rng()
        self.weight = Parameter(rng.normal(0.0, init_std, (num_embeddings, embedding_dim)))

    def forward(self, idx):
        return self.weight[np.asarray(idx, dtype=np.int64)]


class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.weight = Parameter(np.ones(dim))
        self.bias = Parameter(np.zeros(dim))

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        xc = x - mean
        var = (xc * xc).mean(axis=-1, keepdims=True)
        return xc / (var + self.eps).sqrt() * self.weight + self.bias


class Dropout(Module):
    def __init__(self, p=0.1, rng=None):
        self.p = p
        self.rng = rng or np.random.default_rng()

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * mask


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class GELU(Module):
    def forward(self, x):
        return gelu(x)


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

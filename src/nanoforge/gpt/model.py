from __future__ import annotations

import math

import numpy as np

from ..nn import GELU, Dropout, Embedding, LayerNorm, Linear, Module, cross_entropy
from ..tensor import masked_fill, softmax


class CausalSelfAttention(Module):
    def __init__(self, config, rng):
        n = config.n_embd
        proj_std = 0.02 / math.sqrt(2 * config.n_layer)
        self.n_head = config.n_head
        self.n_embd = n
        self.c_attn = Linear(n, 3 * n, bias=config.bias, init_std=0.02, rng=rng)
        self.c_proj = Linear(n, n, bias=config.bias, init_std=proj_std, rng=rng)
        self.attn_drop = Dropout(config.dropout, rng)
        self.resid_drop = Dropout(config.dropout, rng)

    def forward(self, x):
        B, T, C = x.shape
        H = self.n_head
        hd = C // H
        qkv = self.c_attn(x)
        q = qkv[:, :, 0:C].reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        k = qkv[:, :, C : 2 * C].reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        v = qkv[:, :, 2 * C : 3 * C].reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        att = (q @ k.swapaxes(-1, -2)) * (1.0 / math.sqrt(hd))
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        att = softmax(masked_fill(att, mask, -1e9), axis=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(Module):
    def __init__(self, config, rng):
        n = config.n_embd
        proj_std = 0.02 / math.sqrt(2 * config.n_layer)
        self.fc = Linear(n, 4 * n, bias=config.bias, init_std=0.02, rng=rng)
        self.act = GELU()
        self.proj = Linear(4 * n, n, bias=config.bias, init_std=proj_std, rng=rng)
        self.drop = Dropout(config.dropout, rng)

    def forward(self, x):
        return self.drop(self.proj(self.act(self.fc(x))))


class Block(Module):
    def __init__(self, config, rng):
        self.ln1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, rng)
        self.ln2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config, rng)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(Module):
    def __init__(self, config, seed=0):
        rng = np.random.default_rng(seed)
        self.config = config
        self.wte = Embedding(config.vocab_size, config.n_embd, init_std=0.02, rng=rng)
        self.wpe = Embedding(config.block_size, config.n_embd, init_std=0.02, rng=rng)
        self.drop = Dropout(config.dropout, rng)
        self.blocks = [Block(config, rng) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd)

    def forward(self, idx, targets=None):
        idx = np.asarray(idx, dtype=np.int64)
        _, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.config.block_size}")
        x = self.drop(self.wte(idx) + self.wpe(np.arange(T)))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = x @ self.wte.weight.T
        loss = cross_entropy(logits, targets) if targets is not None else None
        return logits, loss

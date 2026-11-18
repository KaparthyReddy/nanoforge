from __future__ import annotations

import numpy as np


class Optimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.buffers = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, buf in zip(self.params, self.buffers):
            if p.grad is None:
                continue
            buf *= self.momentum
            buf += p.grad
            p.data -= self.lr * buf


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.beta1**self.t
        bc2 = 1.0 - self.beta2**self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            if self.weight_decay:
                p.data *= 1.0 - self.lr * self.weight_decay
            p.data -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)


def clip_grad_norm(params, max_norm):
    params = [p for p in params if p.grad is not None]
    total = float(np.sqrt(sum((p.grad**2).sum() for p in params)))
    coef = max_norm / (total + 1e-6)
    if coef < 1.0:
        for p in params:
            p.grad = p.grad * coef
    return total

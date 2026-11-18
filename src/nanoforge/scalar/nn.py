from __future__ import annotations

import random

from .value import Value


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    def __init__(self, nin, activation="relu", rng=None):
        rng = rng or random.Random()
        self.w = [Value(rng.uniform(-1.0, 1.0)) for _ in range(nin)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == "relu":
            return act.relu()
        if self.activation == "tanh":
            return act.tanh()
        if self.activation == "linear":
            return act
        raise ValueError(f"unknown activation: {self.activation}")

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout, activation="relu", rng=None):
        self.neurons = [Neuron(nin, activation, rng) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts, activation="relu", seed=None):
        rng = random.Random(seed)
        sizes = [nin] + list(nouts)
        self.layers = [
            Layer(
                sizes[i],
                sizes[i + 1],
                activation if i < len(nouts) - 1 else "linear",
                rng,
            )
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            if not isinstance(x, list):
                x = [x]
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

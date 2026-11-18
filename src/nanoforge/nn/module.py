from __future__ import annotations

import numpy as np

from ..tensor import Tensor


class Parameter(Tensor):
    __slots__ = ()

    def __init__(self, data):
        super().__init__(np.array(data, dtype=np.float64), requires_grad=True)


def _walk_params(name, value):
    if isinstance(value, Parameter):
        yield name, value
    elif isinstance(value, Module):
        yield from value.named_parameters(name + ".")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _walk_params(f"{name}.{i}", v)
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _walk_params(f"{name}.{k}", v)


def _walk_modules(value):
    if isinstance(value, Module):
        yield from value.modules()
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk_modules(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _walk_modules(v)


class Module:
    training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def named_parameters(self, prefix=""):
        for name, value in vars(self).items():
            yield from _walk_params(f"{prefix}{name}", value)

    def parameters(self):
        seen = set()
        params = []
        for _, p in self.named_parameters():
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)
        return params

    def modules(self):
        yield self
        for value in vars(self).values():
            yield from _walk_modules(value)

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def train(self, mode=True):
        for m in self.modules():
            m.training = mode
        return self

    def eval(self):
        return self.train(False)

    def num_parameters(self):
        return sum(p.size for p in self.parameters())

    def state_dict(self):
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state, strict=True):
        names = set()
        for name, p in self.named_parameters():
            names.add(name)
            if name not in state:
                if strict:
                    raise KeyError(f"missing key: {name}")
                continue
            value = np.asarray(state[name], dtype=np.float64)
            if value.shape != p.shape:
                raise ValueError(f"shape mismatch for {name}: {value.shape} vs {p.shape}")
            p.data[...] = value
        if strict:
            extra = set(state) - names
            if extra:
                raise KeyError(f"unexpected keys: {sorted(extra)}")

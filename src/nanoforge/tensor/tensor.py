from __future__ import annotations

import numpy as np


def unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def _acc(t, g):
    if t.requires_grad:
        t.grad = np.array(g, dtype=np.float64) if t.grad is None else t.grad + g


def _make(data, children, op):
    requires_grad = any(c.requires_grad for c in children)
    return Tensor(
        data,
        requires_grad=requires_grad,
        _children=tuple(children) if requires_grad else (),
        _op=op,
    )


def _norm_index(idx):
    if isinstance(idx, Tensor):
        return idx.data.astype(np.int64)
    if isinstance(idx, tuple):
        return tuple(_norm_index(i) for i in idx)
    if isinstance(idx, list):
        return np.asarray(idx, dtype=np.int64)
    return idx


class Tensor:
    __slots__ = ("_backward", "_op", "_prev", "data", "grad", "requires_grad")
    __array_priority__ = 1000

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self.requires_grad = requires_grad
        self._prev = _children
        self._backward = lambda: None
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def item(self):
        return self.data.item()

    def numpy(self):
        return self.data

    def detach(self):
        return Tensor(self.data)

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    @staticmethod
    def _wrap(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    def __add__(self, other):
        other = self._wrap(other)
        out = _make(self.data + other.data, (self, other), "+")

        def _backward():
            _acc(self, unbroadcast(out.grad, self.shape))
            _acc(other, unbroadcast(out.grad, other.shape))

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = _make(self.data * other.data, (self, other), "*")

        def _backward():
            _acc(self, unbroadcast(out.grad * other.data, self.shape))
            _acc(other, unbroadcast(out.grad * self.data, other.shape))

        out._backward = _backward
        return out

    def __pow__(self, p):
        if not isinstance(p, (int, float)):
            raise TypeError("exponent must be int or float")
        out = _make(self.data**p, (self,), f"**{p}")

        def _backward():
            _acc(self, p * (self.data ** (p - 1)) * out.grad)

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __rsub__(self, other):
        return self._wrap(other) + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (self._wrap(other) ** -1)

    def __rtruediv__(self, other):
        return self._wrap(other) * (self**-1)

    def __matmul__(self, other):
        other = self._wrap(other)
        if self.ndim < 2 or other.ndim < 2:
            raise ValueError("matmul requires tensors with ndim >= 2")
        out = _make(np.matmul(self.data, other.data), (self, other), "@")

        def _backward():
            g = out.grad
            _acc(self, unbroadcast(np.matmul(g, np.swapaxes(other.data, -1, -2)), self.shape))
            _acc(other, unbroadcast(np.matmul(np.swapaxes(self.data, -1, -2), g), other.shape))

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = _make(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            _acc(self, np.broadcast_to(g, self.shape))

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        s = self.sum(axis=axis, keepdims=keepdims)
        count = self.data.size // s.data.size
        return s * (1.0 / count)

    def exp(self):
        e = np.exp(self.data)
        out = _make(e, (self,), "exp")

        def _backward():
            _acc(self, e * out.grad)

        out._backward = _backward
        return out

    def log(self):
        out = _make(np.log(self.data), (self,), "log")

        def _backward():
            _acc(self, out.grad / self.data)

        out._backward = _backward
        return out

    def sqrt(self):
        return self**0.5

    def tanh(self):
        t = np.tanh(self.data)
        out = _make(t, (self,), "tanh")

        def _backward():
            _acc(self, (1.0 - t * t) * out.grad)

        out._backward = _backward
        return out

    def relu(self):
        out = _make(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward():
            _acc(self, (self.data > 0) * out.grad)

        out._backward = _backward
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = _make(self.data.reshape(shape), (self,), "reshape")

        def _backward():
            _acc(self, out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if not axes:
            axes = tuple(range(self.ndim))[::-1]
        inverse = np.argsort(axes)
        out = _make(self.data.transpose(axes), (self,), "transpose")

        def _backward():
            _acc(self, out.grad.transpose(inverse))

        out._backward = _backward
        return out

    def swapaxes(self, a, b):
        perm = list(range(self.ndim))
        perm[a], perm[b] = perm[b], perm[a]
        return self.transpose(*perm)

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, idx):
        idx = _norm_index(idx)
        out = _make(self.data[idx], (self,), "getitem")

        def _backward():
            g = np.zeros_like(self.data)
            np.add.at(g, idx, out.grad)
            _acc(self, g)

        out._backward = _backward
        return out

    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("tensor does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be provided for non-scalar tensors")
            grad = np.ones_like(self.data)
        grad = np.array(grad, dtype=np.float64)
        if grad.shape != self.shape:
            raise ValueError("grad shape mismatch")
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for child in node._prev:
                if child.requires_grad and id(child) not in visited:
                    stack.append((child, False))
        self.grad = grad
        for node in reversed(topo):
            if node.grad is not None:
                node._backward()

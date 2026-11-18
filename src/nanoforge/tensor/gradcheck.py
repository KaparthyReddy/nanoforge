from __future__ import annotations

import numpy as np

from .tensor import Tensor


def _evaluate(fn, arrays):
    return fn(*[Tensor(a.copy()) for a in arrays]).data.item()


def gradcheck(fn, *arrays, eps=1e-6, atol=1e-6, rtol=1e-4):
    arrays = [np.array(a, dtype=np.float64) for a in arrays]
    tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    out = fn(*tensors)
    if out.data.size != 1:
        raise ValueError("fn must return a scalar tensor")
    out.backward()
    for i, base in enumerate(arrays):
        analytic = tensors[i].grad
        if analytic is None:
            analytic = np.zeros_like(base)
        numeric = np.zeros_like(base)
        for idx in np.ndindex(*base.shape):
            original = base[idx]
            base[idx] = original + eps
            plus = _evaluate(fn, arrays)
            base[idx] = original - eps
            minus = _evaluate(fn, arrays)
            base[idx] = original
            numeric[idx] = (plus - minus) / (2 * eps)
        if not np.allclose(analytic, numeric, atol=atol, rtol=rtol):
            err = np.abs(analytic - numeric).max()
            raise AssertionError(f"gradcheck failed for input {i}: max abs error {err:.3e}")
    return True

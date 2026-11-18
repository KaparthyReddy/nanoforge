import numpy as np
import pytest

from nanoforge.tensor import Tensor, gelu, gradcheck, log_softmax, masked_fill, softmax

rng = np.random.default_rng(0)


def r(*shape):
    return rng.standard_normal(shape)


W35 = r(3, 5)
C46 = r(4, 6)
C423 = r(4, 2, 3)
C243 = r(2, 4, 3)


def test_add_broadcast():
    assert gradcheck(lambda a, b: ((a + b) * (a + b)).sum(), r(3, 4), r(4))


def test_mul_div_broadcast():
    assert gradcheck(lambda a, b: (a * b / (b * b + 1.0)).sum(), r(2, 3), r(1, 3))


def test_sub_neg_pow():
    assert gradcheck(lambda a: ((-a - 1.0) ** 2).sum(), r(3, 3))
    assert gradcheck(lambda a: ((a * a + 1.0) ** 1.5).sum(), r(3, 3))


def test_rtruediv():
    assert gradcheck(lambda a: (1.0 / (a * a + 1.0)).sum(), r(2, 3))


def test_constant_ops_do_not_break_backward():
    assert gradcheck(lambda a: (a / 2.0 - 3.0).sum(), r(2, 3))
    assert gradcheck(lambda a: (a - np.ones((2, 3))).sum(), r(2, 3))


def test_matmul_2d():
    assert gradcheck(lambda a, b: (a @ b).tanh().sum(), r(3, 4), r(4, 2))


def test_matmul_batched_broadcast():
    assert gradcheck(lambda a, b: (a @ b).tanh().sum(), r(2, 3, 4), r(4, 5))
    assert gradcheck(lambda a, b: (a @ b).tanh().sum(), r(2, 3, 4), r(2, 4, 5))


def test_sum_mean():
    assert gradcheck(lambda a: (a.sum(axis=1) ** 2).sum(), r(3, 4))
    assert gradcheck(lambda a: (a.mean(axis=(0, 2), keepdims=True) ** 2).sum(), r(2, 3, 4))
    assert gradcheck(lambda a: a.mean() * a.sum(), r(3, 4))


def test_unary():
    assert gradcheck(lambda a: a.exp().sum(), r(3, 3))
    assert gradcheck(lambda a: (a * a + 1.0).log().sum(), r(3, 3))
    assert gradcheck(lambda a: a.tanh().sum(), r(3, 3))
    assert gradcheck(lambda a: (a.relu() * a).sum(), r(4, 4))
    assert gradcheck(lambda a: (a * a + 1.0).sqrt().sum(), r(3, 3))


def test_reshape_transpose():
    assert gradcheck(lambda a: (a.reshape(6, 4).T * Tensor(C46)).tanh().sum(), r(2, 3, 4))
    assert gradcheck(lambda a: (a.transpose(2, 0, 1) * Tensor(C423)).sum(), r(2, 3, 4))
    assert gradcheck(lambda a: (a.swapaxes(1, 2) ** 2 * Tensor(C243)).sum(), r(2, 3, 4))


def test_getitem():
    assert gradcheck(lambda a: (a[[0, 2, 2, 1]] ** 2).sum(), r(4, 3))
    assert gradcheck(lambda a: (a[:, 1:3] ** 2).sum(), r(3, 4))
    assert gradcheck(lambda a: (a[np.arange(3), np.array([1, 0, 2])] ** 2).sum(), r(3, 4))


def test_softmax_log_softmax():
    assert gradcheck(lambda a: (softmax(a) * W35).sum(), r(3, 5))
    assert gradcheck(lambda a: (log_softmax(a) * W35).sum(), r(3, 5))
    assert gradcheck(lambda a: (softmax(a, axis=0) * W35).sum(), r(3, 5))


def test_softmax_rows_sum_to_one():
    y = softmax(Tensor(r(4, 7) * 50)).data
    assert np.allclose(y.sum(axis=-1), 1.0)


def test_gelu():
    assert gradcheck(lambda a: gelu(a).sum(), r(3, 4))


def test_masked_fill_causal_softmax():
    mask = ~np.tril(np.ones((4, 4), dtype=bool))
    w = r(4, 4)
    assert gradcheck(lambda a: (softmax(masked_fill(a, mask, -1e9)) * w).sum(), r(4, 4))
    y = softmax(masked_fill(Tensor(r(4, 4)), mask, -1e9)).data
    assert np.allclose(np.triu(y, k=1), 0.0)


def test_grad_accumulates_over_reuse():
    x = Tensor(np.array([3.0]), requires_grad=True)
    y = (x + x + x * x).sum()
    y.backward()
    assert x.grad[0] == pytest.approx(2.0 + 6.0)


def test_backward_non_scalar_raises():
    x = Tensor(np.ones((2, 2)), requires_grad=True)
    with pytest.raises(RuntimeError):
        (x * 2.0).backward()


def test_no_grad_without_requires_grad():
    x = Tensor(np.ones(3))
    y = (x * 2.0).sum()
    assert not y.requires_grad
    with pytest.raises(RuntimeError):
        y.backward()

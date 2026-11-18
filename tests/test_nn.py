import numpy as np
import pytest

from nanoforge.nn import (
    SGD,
    AdamW,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    Module,
    Sequential,
    Tanh,
    clip_grad_norm,
    cross_entropy,
    mse_loss,
)
from nanoforge.tensor import Tensor, gradcheck


def test_linear_shapes():
    layer = Linear(4, 3, rng=np.random.default_rng(0))
    assert layer(Tensor(np.ones((5, 4)))).shape == (5, 3)
    assert layer(Tensor(np.ones((2, 5, 4)))).shape == (2, 5, 3)


def test_parameter_discovery_and_tying():
    class Tied(Module):
        def __init__(self):
            self.a = Linear(2, 2, rng=np.random.default_rng(0))
            self.b = Linear(2, 2, rng=np.random.default_rng(1))
            self.b.weight = self.a.weight

    assert len(Tied().parameters()) == 3


def test_state_dict_roundtrip():
    m1 = Sequential(Linear(3, 4, rng=np.random.default_rng(0)), Tanh(), Linear(4, 2))
    m2 = Sequential(Linear(3, 4, rng=np.random.default_rng(5)), Tanh(), Linear(4, 2))
    m2.load_state_dict(m1.state_dict())
    x = Tensor(np.random.default_rng(2).standard_normal((6, 3)))
    assert np.allclose(m1(x).data, m2(x).data)


def test_load_state_dict_strict():
    m = Linear(2, 2)
    with pytest.raises(KeyError):
        m.load_state_dict({})


def test_layernorm_stats_and_grad():
    x = np.random.default_rng(0).standard_normal((4, 8)) * 3 + 2
    ln = LayerNorm(8)
    y = ln(Tensor(x)).data
    assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(y.var(axis=-1), 1.0, atol=1e-3)
    w = np.random.default_rng(1).standard_normal((4, 8))
    assert gradcheck(lambda a: (ln(a) * w).sum(), x)


def test_dropout_modes():
    drop = Dropout(0.5, rng=np.random.default_rng(0))
    x = Tensor(np.ones(1000))
    drop.eval()
    assert np.array_equal(drop(x).data, x.data)
    drop.train()
    y = drop(x).data
    assert (y == 0).sum() > 300
    assert y.mean() == pytest.approx(1.0, abs=0.15)


def test_embedding_grad_accumulates():
    emb = Embedding(10, 4, rng=np.random.default_rng(0))
    out = emb(np.array([[1, 1, 3]]))
    assert out.shape == (1, 3, 4)
    out.sum().backward()
    assert np.allclose(emb.weight.grad[1], 2.0)
    assert np.allclose(emb.weight.grad[3], 1.0)
    assert np.allclose(emb.weight.grad[0], 0.0)


def test_cross_entropy():
    assert cross_entropy(Tensor(np.zeros((3, 4))), np.array([0, 1, 2])).item() == pytest.approx(
        np.log(4)
    )
    t = np.array([0, 3, 1, 1, 2])
    assert gradcheck(lambda a: cross_entropy(a, t), np.random.default_rng(0).standard_normal((5, 4)))
    t3 = np.array([[0, 1], [2, 3]])
    assert gradcheck(
        lambda a: cross_entropy(a, t3), np.random.default_rng(1).standard_normal((2, 2, 4))
    )


def test_sgd_linear_regression():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 3))
    y = x @ np.array([[1.0], [-2.0], [0.5]]) + 0.3
    model = Linear(3, 1, rng=rng)
    opt = SGD(model.parameters(), lr=0.1)
    loss = None
    for _ in range(200):
        loss = mse_loss(model(Tensor(x)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 1e-6


def test_adamw_mlp_classification():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(-2, 0.5, (50, 2)), rng.normal(2, 0.5, (50, 2))])
    t = np.array([0] * 50 + [1] * 50)
    model = Sequential(Linear(2, 8, rng=rng), Tanh(), Linear(8, 2, rng=rng))
    opt = AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
    loss = None
    for _ in range(100):
        logits = model(Tensor(x))
        loss = cross_entropy(logits, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc = (model(Tensor(x)).data.argmax(axis=1) == t).mean()
    assert loss.item() < 0.05
    assert acc == 1.0


def test_clip_grad_norm():
    model = Linear(4, 1, rng=np.random.default_rng(0))
    for p in model.parameters():
        p.grad = np.ones_like(p.data) * 3.0
    before = clip_grad_norm(model.parameters(), 1.0)
    after = np.sqrt(sum((p.grad**2).sum() for p in model.parameters()))
    assert before > 1.0
    assert after == pytest.approx(1.0, abs=1e-3)

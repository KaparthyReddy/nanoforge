import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from nanoforge.nn import LayerNorm, cross_entropy  # noqa: E402
from nanoforge.tensor import Tensor, gelu, masked_fill, softmax  # noqa: E402


def _t(a):
    return torch.tensor(a, dtype=torch.float64, requires_grad=True)


def _close(a, b):
    np.testing.assert_allclose(a, b.detach().numpy(), rtol=1e-6, atol=1e-8)


def test_layernorm_gelu_cross_entropy_parity():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 6, 8))
    w = rng.standard_normal((8, 5))
    g = rng.standard_normal(8)
    b = rng.standard_normal(8)
    tgt = rng.integers(0, 5, (4, 6))

    ln = LayerNorm(8)
    ln.weight.data[...] = g
    ln.bias.data[...] = b
    nx, nw = Tensor(x, requires_grad=True), Tensor(w, requires_grad=True)
    loss = cross_entropy(gelu(ln(nx)) @ nw, tgt)
    loss.backward()

    tx, tw, tg, tb = _t(x), _t(w), _t(g), _t(b)
    h = F.gelu(F.layer_norm(tx, (8,), tg, tb, eps=1e-5), approximate="tanh") @ tw
    tloss = F.cross_entropy(h.reshape(-1, 5), torch.tensor(tgt.reshape(-1)))
    tloss.backward()

    _close(loss.data, tloss)
    _close(nx.grad, tx.grad)
    _close(nw.grad, tw.grad)
    _close(ln.weight.grad, tg.grad)
    _close(ln.bias.grad, tb.grad)


def test_causal_attention_parity():
    rng = np.random.default_rng(1)
    shape = (2, 3, 5, 4)
    q, k, v = (rng.standard_normal(shape) for _ in range(3))
    wt = rng.standard_normal(shape)

    nq, nk, nv = (Tensor(a, requires_grad=True) for a in (q, k, v))
    mask = np.triu(np.ones((5, 5), dtype=bool), k=1)
    scores = (nq @ nk.swapaxes(-1, -2)) * (1.0 / np.sqrt(4))
    out = softmax(masked_fill(scores, mask, -1e9), axis=-1) @ nv
    (out * wt).sum().backward()

    tq, tk, tv = _t(q), _t(k), _t(v)
    tout = F.scaled_dot_product_attention(tq, tk, tv, is_causal=True)
    (tout * torch.tensor(wt)).sum().backward()

    _close(out.data, tout)
    _close(nq.grad, tq.grad)
    _close(nk.grad, tk.grad)
    _close(nv.grad, tv.grad)

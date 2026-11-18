import pytest

from nanoforge.scalar import MLP, Value


def test_reference_graph():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    assert g.data == pytest.approx(24.7041, abs=1e-3)
    assert a.grad == pytest.approx(138.8338, abs=1e-3)
    assert b.grad == pytest.approx(645.5773, abs=1e-3)


@pytest.mark.parametrize(
    "fn,x0",
    [
        (lambda x: x.tanh(), 0.3),
        (lambda x: x.exp(), 0.5),
        (lambda x: x.log(), 1.7),
        (lambda x: x.sigmoid(), -0.4),
        (lambda x: x.relu(), 0.8),
        (lambda x: x**3, 1.2),
        (lambda x: x * x + x, 0.9),
        (lambda x: 1.0 / x, 2.0),
        (lambda x: (x * 2.0 - 1.0).tanh() * x.exp(), 0.35),
    ],
)
def test_numeric_gradient(fn, x0):
    eps = 1e-6
    x = Value(x0)
    y = fn(x)
    y.backward()
    numeric = (fn(Value(x0 + eps)).data - fn(Value(x0 - eps)).data) / (2 * eps)
    assert x.grad == pytest.approx(numeric, rel=1e-5, abs=1e-7)


def test_reused_node_accumulates():
    x = Value(3.0)
    y = x + x + x * x
    y.backward()
    assert x.grad == pytest.approx(2.0 + 2 * 3.0)


def test_deep_graph_no_recursion_error():
    x = Value(1.0)
    y = x
    for _ in range(5000):
        y = y + 1.0
    y.backward()
    assert x.grad == 1.0


def test_mlp_fits_line():
    model = MLP(1, [1], seed=0)
    xs = [-1.0, 0.0, 1.0, 2.0]
    ys = [-1.0, 1.0, 3.0, 5.0]
    loss = None
    for _ in range(300):
        preds = [model([x]) for x in xs]
        loss = sum(((p - y) ** 2 for p, y in zip(preds, ys)), Value(0.0)) / len(xs)
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= 0.05 * p.grad
    assert loss.data < 1e-4

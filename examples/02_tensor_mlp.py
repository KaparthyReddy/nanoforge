import numpy as np

from nanoforge.nn import AdamW, Linear, Sequential, Tanh, cross_entropy
from nanoforge.tensor import Tensor

rng = np.random.default_rng(0)
x = rng.uniform(-1.5, 1.5, (400, 2))
y = (np.sum(x**2, axis=1) < 1.0).astype(np.int64)

model = Sequential(
    Linear(2, 16, rng=rng),
    Tanh(),
    Linear(16, 16, rng=rng),
    Tanh(),
    Linear(16, 2, rng=rng),
)
opt = AdamW(model.parameters(), lr=0.02, weight_decay=0.0)

for step in range(301):
    logits = model(Tensor(x))
    loss = cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 50 == 0:
        acc = (logits.data.argmax(axis=1) == y).mean()
        print(f"step {step:3d} loss {loss.item():.4f} acc {acc:.3f}")

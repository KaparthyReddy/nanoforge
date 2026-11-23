from nanoforge.scalar import MLP, Value

xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
ys = [-1.0, 1.0, 1.0, -1.0]

model = MLP(2, [8, 1], activation="tanh", seed=0)

for step in range(301):
    preds = [model(x) for x in xs]
    loss = sum(((p - y) ** 2 for p, y in zip(preds, ys)), Value(0.0)) / len(xs)
    model.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.data -= 0.1 * p.grad
    if step % 50 == 0:
        print(f"step {step:3d} loss {loss.data:.5f}")

for x, y in zip(xs, ys):
    print(x, "->", round(model(x).data, 3), "target", y)

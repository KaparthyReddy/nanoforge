from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ..nn import AdamW, clip_grad_norm
from .config import GPTConfig
from .model import GPT


def get_lr(step, config):
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    if step >= config.max_steps:
        return config.min_lr
    span = max(1, config.max_steps - config.warmup_steps)
    ratio = (step - config.warmup_steps) / span
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return config.min_lr + coeff * (config.lr - config.min_lr)


def estimate_loss(model, dataset, batch_size, eval_batches, rng):
    was_training = model.training
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(eval_batches):
            x, y = dataset.get_batch(split, batch_size, rng)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train(was_training)
    return out


def save_checkpoint(path, model):
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    config = np.array(json.dumps(model.config.to_dict()))
    np.savez(path, __config__=config, **model.state_dict())
    return path


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        config = GPTConfig.from_dict(json.loads(str(data["__config__"])))
        state = {k: data[k] for k in data.files if k != "__config__"}
    model = GPT(config)
    model.load_state_dict(state)
    return model


def train(model, dataset, config, log=print):
    rng = np.random.default_rng(config.seed)
    opt = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history = []
    best_val = float("inf")
    start = time.time()
    model.train()
    for step in range(config.max_steps + 1):
        lr = get_lr(step, config)
        opt.lr = lr
        if step % config.eval_interval == 0 or step == config.max_steps:
            losses = estimate_loss(model, dataset, config.batch_size, config.eval_batches, rng)
            history.append(
                {"step": step, "train_loss": losses["train"], "val_loss": losses["val"], "lr": lr}
            )
            log(
                f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| lr {lr:.2e} | {time.time() - start:.1f}s"
            )
            if config.checkpoint_dir and losses["val"] < best_val:
                save_checkpoint(Path(config.checkpoint_dir) / "best.npz", model)
            best_val = min(best_val, losses["val"])
        if step == config.max_steps:
            break
        x, y = dataset.get_batch("train", config.batch_size, rng)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        if config.grad_clip:
            clip_grad_norm(model.parameters(), config.grad_clip)
        opt.step()
    return history

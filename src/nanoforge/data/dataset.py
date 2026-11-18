from __future__ import annotations

import numpy as np


class TokenDataset:
    def __init__(self, tokens, block_size, val_fraction=0.1):
        tokens = np.asarray(tokens, dtype=np.int64)
        n_val = int(len(tokens) * val_fraction)
        split = len(tokens) - n_val
        self.block_size = block_size
        self.train = tokens[:split]
        self.val = tokens[split:]
        if len(self.train) <= block_size:
            raise ValueError("not enough training tokens for this block_size")
        if val_fraction > 0 and len(self.val) <= block_size:
            raise ValueError("not enough validation tokens for this block_size")

    def get_batch(self, split, batch_size, rng):
        data = self.train if split == "train" else self.val
        if split not in ("train", "val") or len(data) <= self.block_size:
            raise ValueError(f"split unavailable: {split}")
        starts = rng.integers(0, len(data) - self.block_size, batch_size)
        x = np.stack([data[i : i + self.block_size] for i in starts])
        y = np.stack([data[i + 1 : i + self.block_size + 1] for i in starts])
        return x, y

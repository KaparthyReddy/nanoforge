from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class TrainConfig:
    batch_size: int = 16
    max_steps: int = 500
    lr: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 50
    eval_batches: int = 10
    seed: int = 0
    checkpoint_dir: str | None = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

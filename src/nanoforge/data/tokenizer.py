from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

_PATTERN = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = dict(enumerate(self.chars))

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path):
        Path(path).write_text(json.dumps({"type": "char", "chars": self.chars}), encoding="utf-8")


def _merge(ids, pair, new_id):
    out = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges=None):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._cache = {}
        for a, b in merges or []:
            self._add_merge(a, b)

    def _add_merge(self, a, b):
        if a not in self.vocab or b not in self.vocab:
            raise ValueError(f"unknown token in merge: {(a, b)}")
        new_id = 256 + len(self.merges)
        self.merges[(a, b)] = new_id
        self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        return new_id

    @property
    def vocab_size(self):
        return len(self.vocab)

    @classmethod
    def train(cls, text, vocab_size):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        counts = Counter(_PATTERN.findall(text))
        words = [(list(w.encode("utf-8")), c) for w, c in counts.items()]
        tok = cls()
        while tok.vocab_size < vocab_size:
            pair_counts = Counter()
            for ids, c in words:
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += c
            if not pair_counts:
                break
            best = max(
                pair_counts.items(),
                key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]),
            )[0]
            new_id = tok._add_merge(*best)
            words = [(_merge(ids, best, new_id), c) for ids, c in words]
        return tok

    def _encode_chunk(self, chunk):
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            best = None
            best_rank = None
            for pair in zip(ids, ids[1:]):
                rank = self.merges.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best = pair
                    best_rank = rank
            if best is None:
                break
            ids = _merge(ids, best, best_rank)
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        out = []
        for chunk in _PATTERN.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids):
        data = b"".join(self.vocab[int(i)] for i in ids)
        return data.decode("utf-8", errors="replace")

    def save(self, path):
        payload = {"type": "bpe", "merges": [list(p) for p in self.merges]}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")


def load_tokenizer(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    kind = payload.get("type")
    if kind == "char":
        return CharTokenizer(payload["chars"])
    if kind == "bpe":
        return BPETokenizer([tuple(p) for p in payload["merges"]])
    raise ValueError(f"unknown tokenizer type: {kind}")

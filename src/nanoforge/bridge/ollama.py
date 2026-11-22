from __future__ import annotations

import json
import time
from dataclasses import dataclass
from urllib.request import Request, urlopen

import numpy as np

from ..gpt import generate

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(RuntimeError):
    pass


@dataclass
class GenerationResult:
    text: str
    tokens: int
    seconds: float

    @property
    def tokens_per_sec(self):
        return self.tokens / self.seconds if self.seconds > 0 else 0.0


class OllamaClient:
    def __init__(self, host=DEFAULT_HOST, timeout=300.0):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = Request(
            self.host + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (OSError, ValueError) as exc:
            raise OllamaError(f"ollama request to {self.host}{path} failed: {exc}") from exc

    def models(self):
        return [m["name"] for m in self._request("/api/tags").get("models", [])]

    def is_available(self):
        try:
            self.models()
        except OllamaError:
            return False
        return True

    def generate(self, model, prompt, max_tokens=128, temperature=0.8, seed=None):
        options = {"num_predict": max_tokens, "temperature": temperature}
        if seed is not None:
            options["seed"] = seed
        payload = {"model": model, "prompt": prompt, "stream": False, "options": options}
        start = time.perf_counter()
        resp = self._request("/api/generate", payload)
        wall = time.perf_counter() - start
        eval_ns = resp.get("eval_duration") or 0
        seconds = eval_ns / 1e9 if eval_ns else wall
        return GenerationResult(
            text=resp.get("response", ""),
            tokens=int(resp.get("eval_count", 0)),
            seconds=seconds,
        )


def run_local(model, tokenizer, prompt, max_new_tokens=128, temperature=0.8, top_k=None, seed=None):
    ids = tokenizer.encode(prompt) if prompt else [0]
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    out = generate(model, ids, max_new_tokens, temperature, top_k, rng)
    seconds = time.perf_counter() - start
    new_ids = out[0, len(ids) :]
    return GenerationResult(
        text=tokenizer.decode(new_ids),
        tokens=len(new_ids),
        seconds=seconds,
    )

# nanoforge

![ci](https://github.com/KaparthyReddy/nanoforge/actions/workflows/ci.yml/badge.svg)

A GPT built from first principles in pure NumPy: scalar autograd, tensor autograd, transformer, training, sampling, and a bridge for benchmarking against local Ollama models. Nothing here calls a deep learning framework.

## Why

Reading micrograd and nanoGPT teaches the ideas. Rebuilding them teaches the details: broadcasting gradients, a stable softmax backward, a causal mask, weight tying, learning-rate schedules. Every piece here is verified against numeric gradients, so each step is trustworthy.

## What is inside

| Layer | Module | Description |
| --- | --- | --- |
| Scalar autograd | `nanoforge.scalar` | micrograd-style `Value` engine and a small MLP |
| Tensor autograd | `nanoforge.tensor` | Reverse-mode autodiff on NumPy arrays: broadcasting, batched matmul, softmax, GELU, masking, `gradcheck` |
| Neural nets | `nanoforge.nn` | Module system, Linear, Embedding, LayerNorm, Dropout, losses, SGD, AdamW |
| Data | `nanoforge.data` | Tiny Shakespeare downloader, char and byte-level BPE tokenizers, dataset |
| GPT | `nanoforge.gpt` | Decoder-only transformer, trainer with cosine schedule, sampler, checkpoints |
| Bridge | `nanoforge.bridge` | Compare generations and speed against a local Ollama model |

## Quickstart

```bash
git clone https://github.com/KaparthyReddy/nanoforge.git
cd nanoforge
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
```

Train and sample:

```bash
nanoforge download
nanoforge train --config configs/tiny.json --out checkpoints/tiny
nanoforge generate --checkpoint checkpoints/tiny/best.npz --prompt "ROMEO:" --max-new-tokens 300 --top-k 40
```

Compare against a local Ollama model (needs `ollama serve` running and a pulled model):

```bash
ollama list
nanoforge compare --checkpoint checkpoints/tiny/best.npz --ollama-model llama3.2:3b
```

Examples live in `examples/`: scalar XOR, a tensor MLP classifier, and a minimal GPT training script.

## Results

Tiny Shakespeare, character-level tokenizer (vocab 65), `configs/tiny.json` (2 layers, 2 heads, 64-dim, context 64, 108,352 parameters), trained on CPU with NumPy.

| Run | Steps | Time | Best val loss | Val bits per char |
| --- | --- | --- | --- | --- |
| tiny | 1,000 | 113 s | 2.03 | 2.93 |
| tiny | 5,000 | 401 s | 1.73 | 2.50 |

A uniform guess over 65 characters scores a loss of 4.17, which is where training starts.

Sample from the 5,000-step run (`--top-k 40`, temperature 0.8):

```
ROMEO:
I am too to be subject to darken to the beens
That when is sincessis though thee, the scir.

CUCENTES:
Name in bows were bess is come me that's recuse
```

Speed against Ollama on the same prompt, one run each: nanoforge generated 548.7 tokens/s from its 108k-parameter model, and `llama3.2:3b` generated 20.5 tokens/s. The numbers are indicative only: the tokenizations differ (characters versus subwords), the models differ by about four orders of magnitude, and Ollama's figure excludes model load time.

## Correctness

- Every tensor op is checked against central-difference numeric gradients, including broadcasting and batched matmul.
- Every parameter of the full GPT is checked against sampled numeric gradients.
- A causality test confirms that changing a future token never changes earlier logits.
- An overfit test confirms the model can memorise a repeating pattern and reproduce it greedily.
- Optional PyTorch parity tests (`pip install -e ".[ref]"`) cover layer norm, GELU, cross-entropy and causal attention to about 1e-6.

## Design decisions

The reasoning behind each choice (iterative topological sort, `requires_grad` pruning, mask constant, weight tying, initialisation, schedules, checkpoint format, BPE details, bridge caveats) is in [docs/design-decisions.md](docs/design-decisions.md).

## Limitations

- CPU only and float64, so it is slow compared with a real framework.
- No KV cache; generation recomputes the full context at every step.
- Weight decay is applied to all parameters, including norms and biases.
- BPE is implemented and tested for correctness but not yet benchmarked end to end.

## Project layout

```
src/nanoforge/
├── scalar/    micrograd-style engine
├── tensor/    NumPy autograd and gradcheck
├── nn/        modules, layers, losses, optimisers
├── data/      downloader, tokenizers, dataset
├── gpt/       model, trainer, sampler
├── bridge/    Ollama client and comparison helpers
└── cli.py     download, train, generate, compare
configs/       tiny and small training configs
examples/      runnable walkthroughs
docs/          design decisions
tests/         unit, integration, optional torch parity
```

## License

MIT

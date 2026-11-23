# Design decisions

## Autograd

- **Define-by-run tape.** Each op records a backward closure while the forward pass runs. This is the simplest model to read and mirrors PyTorch eager mode.
- **Iterative topological sort.** `backward()` uses an explicit stack, not recursion, so deep graphs do not hit Python's recursion limit. A test builds a 5000-node chain to enforce this.
- **`requires_grad` pruning.** Constants wrapped from Python scalars never enter the backward pass. An early version ran their backward closures with `grad=None` and crashed; the fix is to skip children that do not require grad and nodes that received no gradient.
- **Accumulate with `+=`.** A node used twice receives the sum of both gradient paths.
- **`unbroadcast`.** Every binary op sums gradients back over the dimensions NumPy broadcast, so shapes always match their inputs.
- **float64 everywhere.** Finite-difference gradient checks need the precision. The cost is speed, which is acceptable at this scale.
- **Fused softmax and log-softmax backward.** The max is subtracted for numerical stability, and the closed-form gradient avoids building a large Jacobian.
- **`-1e9` instead of `-inf` for the causal mask.** It avoids inf/NaN edge cases, and every causal row has at least one unmasked entry, so the result matches PyTorch's causal attention to about 1e-6.

## Model

- **Pre-LayerNorm blocks (GPT-2 style)** with GELU (tanh approximation), so activations stay well scaled in deeper stacks.
- **Weight tying.** The token embedding doubles as the output projection, which cuts parameters and matches nanoGPT. Side effect: tests that check the initial loss against `ln(vocab)` must use targets independent of the inputs, because tying leaks the input into the logits.
- **Initialisation.** Normal with std 0.02. Residual-path projections are scaled by `1/sqrt(2 * n_layer)` so the residual stream's variance does not grow with depth.
- **Dropout defaults to 0.** Tiny models on small data underfit before they overfit.

## Training

- **AdamW with decoupled weight decay.** Known simplification: decay is applied to all parameters, including LayerNorm and biases. nanoGPT excludes them.
- **Linear warmup then cosine decay** to a `min_lr` floor.
- **Global-norm gradient clipping.**
- **Best checkpoint by validation loss**, plus a final checkpoint at the end of the run.
- **Checkpoints are `.npz` with the config stored as a JSON string.** They load with `allow_pickle=False`, so loading an untrusted checkpoint cannot execute code.

## Data

- **Character tokenizer as the baseline**, byte-level BPE as the upgrade.
- **BPE operates on bytes**, so any UTF-8 text round-trips. Pre-tokenisation uses a regex that keeps leading spaces attached to words.
- **BPE training counts unique pre-tokenised words** instead of scanning the full text on every merge, which makes training fast. Ties are broken deterministically.
- **BPE encoding applies the lowest-rank merge first** (the GPT-2 rule) and caches per-chunk results.
- **Contiguous train/validation split** (last 10% held out), with no shuffling across the boundary.

## Ollama bridge

- **Standard library `urllib` only.** No extra dependencies, and non-streaming requests keep the code small.
- **Tokens per second from Ollama's own `eval_duration`**, which excludes model load time. The local number is wall-clock generation time. The comparison is indicative, not a rigorous benchmark.
- **Tests use a mock HTTP server**, so CI never needs Ollama installed.

## Testing

- Central-difference `gradcheck` on every tensor op, including broadcasting and batched matmul.
- Sampled numeric gradients for every parameter of the full GPT.
- A causality test: changing a future token must not change earlier logits.
- An overfit test: the model must memorise a repeating pattern and reproduce it greedily.
- Optional PyTorch parity tests (`pip install -e ".[ref]"`) for layer norm, GELU, cross-entropy and causal attention.

## Known limitations

- CPU only and float64, so it is slow compared with a framework.
- No KV cache; generation recomputes the full context each step.
- Only the character tokenizer has been benchmarked; BPE is implemented and tested for correctness.
- Weight decay is not excluded from norms and biases.

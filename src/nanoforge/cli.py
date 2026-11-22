from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_run(checkpoint):
    from .data import load_tokenizer
    from .gpt import load_checkpoint

    checkpoint = Path(checkpoint)
    model = load_checkpoint(checkpoint)
    tokenizer = load_tokenizer(checkpoint.parent / "tokenizer.json")
    return model, tokenizer


def cmd_download(args):
    from .data import download_tiny_shakespeare

    print(download_tiny_shakespeare(args.dest, force=args.force))
    return 0


def cmd_train(args):
    from .data import BPETokenizer, CharTokenizer, TokenDataset, download_tiny_shakespeare
    from .gpt import GPT, GPTConfig, TrainConfig, save_checkpoint, train

    cfg = _load_json(args.config)
    data_path = Path(args.data)
    if not data_path.exists():
        download_tiny_shakespeare(data_path)
    text = data_path.read_text(encoding="utf-8")

    if args.tokenizer == "bpe":
        tokenizer = BPETokenizer.train(text, args.bpe_vocab)
    else:
        tokenizer = CharTokenizer.from_text(text)
    tokens = tokenizer.encode(text)
    print(f"tokenizer={args.tokenizer} vocab={tokenizer.vocab_size} tokens={len(tokens)}")

    model_cfg = GPTConfig(vocab_size=tokenizer.vocab_size, **cfg["model"])
    train_cfg = TrainConfig(**cfg["train"])
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps
    if args.seed is not None:
        train_cfg.seed = args.seed

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_cfg.checkpoint_dir = str(out)
    tokenizer.save(out / "tokenizer.json")

    dataset = TokenDataset(tokens, model_cfg.block_size)
    model = GPT(model_cfg, seed=train_cfg.seed)
    print(f"parameters={model.num_parameters():,}")

    history = train(model, dataset, train_cfg)
    (out / "history.json").write_text(json.dumps(history), encoding="utf-8")
    final = save_checkpoint(out / "final.npz", model)
    print(f"saved {final}")
    return 0


def cmd_generate(args):
    from .gpt import generate

    model, tokenizer = _load_run(args.checkpoint)
    try:
        ids = tokenizer.encode(args.prompt) if args.prompt else [0]
    except KeyError as exc:
        print(f"prompt has characters outside the tokenizer vocabulary: {exc}", file=sys.stderr)
        return 1
    rng = np.random.default_rng(args.seed)
    out = generate(model, ids, args.max_new_tokens, args.temperature, args.top_k, rng)
    print(tokenizer.decode(out[0]))
    return 0


def cmd_compare(args):
    from .bridge import OllamaClient, OllamaError, run_local

    client = OllamaClient(args.host)
    if not client.is_available():
        print(f"cannot reach ollama at {args.host}; start it with: ollama serve", file=sys.stderr)
        return 1
    model, tokenizer = _load_run(args.checkpoint)
    try:
        local = run_local(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.seed,
        )
    except KeyError as exc:
        print(f"prompt has characters outside the tokenizer vocabulary: {exc}", file=sys.stderr)
        return 1
    try:
        remote = client.generate(
            args.ollama_model,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.seed,
        )
    except OllamaError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    n_params = model.num_parameters()
    print(f"[nanoforge | {n_params:,} params] {local.tokens} tokens, {local.tokens_per_sec:.1f} tok/s")
    print(args.prompt + local.text)
    print()
    print(f"[ollama | {args.ollama_model}] {remote.tokens} tokens, {remote.tokens_per_sec:.1f} tok/s")
    print(args.prompt + remote.text)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="nanoforge",
        description="From-scratch autograd, tensors and GPT in pure NumPy.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download Tiny Shakespeare")
    p.add_argument("--dest", default="data/tinyshakespeare.txt")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("train", help="train a GPT")
    p.add_argument("--config", default="configs/tiny.json")
    p.add_argument("--data", default="data/tinyshakespeare.txt")
    p.add_argument("--out", default="checkpoints/run")
    p.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    p.add_argument("--bpe-vocab", type=int, default=512)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("generate", help="sample from a checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("compare", help="compare a checkpoint against a local Ollama model")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ollama-model", required=True)
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_compare)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

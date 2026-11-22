import json

from nanoforge.cli import main


def test_train_then_generate(tmp_path, capsys):
    data = tmp_path / "data.txt"
    data.write_text("abc" * 500, encoding="utf-8")
    config = tmp_path / "cfg.json"
    config.write_text(
        json.dumps(
            {
                "model": {"block_size": 8, "n_layer": 1, "n_head": 2, "n_embd": 16},
                "train": {
                    "batch_size": 4,
                    "max_steps": 5,
                    "warmup_steps": 1,
                    "eval_interval": 5,
                    "eval_batches": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "run"
    rc = main(
        ["train", "--data", str(data), "--config", str(config), "--out", str(out)],
    )
    assert rc == 0
    for name in ("final.npz", "best.npz", "tokenizer.json", "history.json"):
        assert (out / name).exists()
    capsys.readouterr()

    rc = main(
        [
            "generate",
            "--checkpoint",
            str(out / "final.npz"),
            "--prompt",
            "a",
            "--max-new-tokens",
            "5",
            "--seed",
            "0",
        ]
    )
    assert rc == 0
    text = capsys.readouterr().out.strip()
    assert text.startswith("a")
    assert len(text) == 6


def test_generate_rejects_unknown_prompt_chars(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("abc" * 500, encoding="utf-8")
    config = tmp_path / "cfg.json"
    config.write_text(
        json.dumps(
            {
                "model": {"block_size": 8, "n_layer": 1, "n_head": 2, "n_embd": 16},
                "train": {"batch_size": 4, "max_steps": 1, "eval_interval": 1, "eval_batches": 1},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "run"
    main(["train", "--data", str(data), "--config", str(config), "--out", str(out)])
    rc = main(["generate", "--checkpoint", str(out / "final.npz"), "--prompt", "zzz"])
    assert rc == 1

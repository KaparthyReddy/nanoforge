import numpy as np
import pytest

from nanoforge.data import BPETokenizer, CharTokenizer, TokenDataset, load_tokenizer
from nanoforge.gpt import (
    GPT,
    GPTConfig,
    TrainConfig,
    generate,
    get_lr,
    load_checkpoint,
    save_checkpoint,
    train,
)


def _tiny(seed=0):
    cfg = GPTConfig(vocab_size=7, block_size=5, n_layer=2, n_head=2, n_embd=8, dropout=0.0)
    model = GPT(cfg, seed=seed)
    rng = np.random.default_rng(1)
    for p in model.parameters():
        p.data += rng.normal(0.0, 0.1, p.shape)
    return model


def test_char_tokenizer_roundtrip(tmp_path):
    tok = CharTokenizer.from_text("hello world")
    ids = tok.encode("low")
    assert tok.decode(ids) == "low"
    assert tok.vocab_size == 8
    with pytest.raises(KeyError):
        tok.encode("z")
    tok.save(tmp_path / "t.json")
    loaded = load_tokenizer(tmp_path / "t.json")
    assert loaded.encode("hello") == tok.encode("hello")


def test_bpe_roundtrip_and_compression(tmp_path):
    text = "the quick brown fox jumps over the lazy dog. " * 50
    tok = BPETokenizer.train(text, 300)
    assert 256 < tok.vocab_size <= 300
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert len(ids) < len(text.encode("utf-8"))
    sample = "héllo wörld ☃ the fox"
    assert tok.decode(tok.encode(sample)) == sample
    tok.save(tmp_path / "bpe.json")
    loaded = load_tokenizer(tmp_path / "bpe.json")
    assert loaded.encode(sample) == tok.encode(sample)


def test_bpe_untrained_is_bytes():
    tok = BPETokenizer()
    assert tok.encode("abc") == [97, 98, 99]
    assert tok.decode([97, 98, 99]) == "abc"


def test_dataset_batches():
    ds = TokenDataset(np.arange(1000), block_size=8, val_fraction=0.1)
    assert len(ds.train) == 900
    assert len(ds.val) == 100
    x, y = ds.get_batch("train", 4, np.random.default_rng(0))
    assert x.shape == (4, 8)
    assert np.array_equal(y, x + 1)
    with pytest.raises(ValueError):
        TokenDataset(np.arange(10), block_size=8)


def test_forward_shapes_and_loss():
    model = _tiny()
    idx = np.zeros((3, 5), dtype=np.int64)
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 5, 7)
    assert loss.item() > 0
    logits, loss = model(idx[:, :2])
    assert logits.shape == (3, 2, 7)
    assert loss is None
    with pytest.raises(ValueError):
        model(np.zeros((1, 6), dtype=np.int64))


def test_initial_loss_near_log_vocab():
    cfg = GPTConfig(vocab_size=50, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = GPT(cfg, seed=0)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 50, (4, 8))
    tgt = rng.integers(0, 50, (4, 8))
    _, loss = model(idx, tgt)
    assert loss.item() == pytest.approx(np.log(50), abs=0.15)


def test_causality():
    model = _tiny().eval()
    a = np.array([[1, 2, 3, 4, 5]])
    b = a.copy()
    b[0, -1] = 0
    la = model(a)[0].data
    lb = model(b)[0].data
    assert np.allclose(la[:, :-1], lb[:, :-1])
    assert not np.allclose(la[:, -1], lb[:, -1])


def test_param_gradients_match_numeric():
    model = _tiny()
    rng = np.random.default_rng(2)
    idx = rng.integers(0, 7, (2, 5))
    tgt = rng.integers(0, 7, (2, 5))
    _, loss = model(idx, tgt)
    model.zero_grad()
    loss.backward()
    eps = 1e-5
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        for _ in range(3):
            i = tuple(int(rng.integers(0, s)) for s in p.shape)
            orig = p.data[i]
            p.data[i] = orig + eps
            plus = model(idx, tgt)[1].item()
            p.data[i] = orig - eps
            minus = model(idx, tgt)[1].item()
            p.data[i] = orig
            numeric = (plus - minus) / (2 * eps)
            assert p.grad[i] == pytest.approx(numeric, rel=1e-4, abs=1e-7), (name, i)


def test_get_lr_schedule():
    tc = TrainConfig(max_steps=100, lr=1e-2, min_lr=1e-3, warmup_steps=10)
    assert get_lr(0, tc) < tc.lr
    assert get_lr(10, tc) == pytest.approx(tc.lr)
    assert get_lr(100, tc) == pytest.approx(tc.min_lr)
    assert tc.min_lr < get_lr(55, tc) < tc.lr


def test_overfits_repeating_pattern():
    text = "abc" * 400
    tok = CharTokenizer.from_text(text)
    ds = TokenDataset(tok.encode(text), block_size=8)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = GPT(cfg, seed=0)
    tc = TrainConfig(
        batch_size=8,
        max_steps=200,
        lr=1e-2,
        min_lr=1e-3,
        warmup_steps=10,
        weight_decay=0.0,
        eval_interval=100,
        eval_batches=2,
    )
    history = train(model, ds, tc, log=lambda *_: None)
    assert history[-1]["step"] == tc.max_steps
    assert history[-1]["val_loss"] < 0.5
    out = generate(model, [0], 9, temperature=0)
    assert tok.decode(out[0]) == "abcabcabca"


def test_generate_shapes_and_determinism():
    model = _tiny().eval()
    out = generate(model, [1, 2], 12, rng=np.random.default_rng(0))
    assert out.shape == (1, 14)
    assert out.min() >= 0
    assert out.max() < 7
    greedy = generate(model, [1, 2], 6, temperature=0)
    top1 = generate(model, [1, 2], 6, temperature=1.0, top_k=1, rng=np.random.default_rng(0))
    assert np.array_equal(greedy, top1)
    a = generate(model, [1], 5, rng=np.random.default_rng(3))
    b = generate(model, [1], 5, rng=np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_checkpoint_roundtrip(tmp_path):
    model = _tiny()
    path = save_checkpoint(tmp_path / "m", model)
    loaded = load_checkpoint(path)
    idx = np.array([[1, 2, 3, 4, 5]])
    assert np.allclose(model.eval()(idx)[0].data, loaded.eval()(idx)[0].data)

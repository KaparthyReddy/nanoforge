import numpy as np

from nanoforge.data import CharTokenizer, TokenDataset, download_tiny_shakespeare
from nanoforge.gpt import GPT, GPTConfig, TrainConfig, generate, train

text = download_tiny_shakespeare().read_text(encoding="utf-8")
tokenizer = CharTokenizer.from_text(text)
dataset = TokenDataset(tokenizer.encode(text), block_size=64)

config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=64, n_layer=2, n_head=2, n_embd=64)
model = GPT(config)
train(model, dataset, TrainConfig(max_steps=300, eval_interval=100))

ids = tokenizer.encode("ROMEO:")
out = generate(model, ids, 200, temperature=0.8, top_k=40, rng=np.random.default_rng(0))
print(tokenizer.decode(out[0]))

import torch
from sakicoder.config import Config
from sakicoder.model import GPT


def test_forward_shapes():
    cfg = Config( n_layer=2, n_head=2, n_embd=64, block_size=32, dropout=0.1)
    vocab_size = 100
    model = GPT(cfg, vocab_size)
    x = torch.randint(0, vocab_size, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, vocab_size)
    assert loss is not None

import argparse
import time
import torch
from pathlib import Path
from typing import Optional

from sakicoder.config import Config
from sakicoder.model import GPT


def _infer_vocab_size_from_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    for k, v in state.items():
        if k.endswith("tok_emb.weight") or k.endswith("head.weight"):
            return v.shape[0]
    raise RuntimeError("Could not infer vocab size from checkpoint")


def prepare_model_from_ckpt(checkpoint: str, config: str, device: str = "cpu"):
    cfg = Config.from_json(config)
    vocab_size = _infer_vocab_size_from_checkpoint(checkpoint)
    model = GPT(cfg, vocab_size)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.to(device)
    model.eval()
    return model


def _tokenize_prompt(prompt: str, tokenizer_path: Optional[str], max_len: int):
    if tokenizer_path is None:
        # fallback: return a short sequence of zeros
        return torch.zeros((1, min(8, max_len)), dtype=torch.long)
    try:
        from tokenizers import Tokenizer
    except Exception:
        print("tokenizers package not installed. Install with `pip install tokenizers` to use real tokenizer.")
        return torch.zeros((1, min(8, max_len)), dtype=torch.long)

    tok = Tokenizer.from_file(tokenizer_path)
    enc = tok.encode(prompt)
    ids = enc.ids[:max_len]
    import numpy as np
    arr = np.array(ids, dtype="int64")[None, :]
    return torch.from_numpy(arr)


def benchmark(checkpoint: str, config: str, tokenizer: Optional[str], prompt: str, max_new_tokens: int = 20, mode: str = "eager"):
    device = "cpu"
    if mode == "torchscript":
        # try to find corresponding torchscript file
        ts_path = Path("exports/torchscript/sakicoder_tiny.pt")
        if ts_path.exists():
            model = torch.jit.load(str(ts_path), map_location=device)
        else:
            raise FileNotFoundError("TorchScript model not found at exports/torchscript/sakicoder_tiny.pt")
    else:
        model = prepare_model_from_ckpt(checkpoint, config, device=device)
        if mode == "quantized":
            try:
                import torch.nn as nn
                model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
            except Exception as e:
                print("Quantization failed:", e)

    block_size = getattr(model, "block_size", 512)
    input_ids = _tokenize_prompt(prompt, tokenizer, block_size)

    # warmup
    with torch.no_grad():
        for _ in range(2):
            _ = model(input_ids)

    # generation (greedy) measure
    generated = []
    start = time.perf_counter()
    with torch.no_grad():
        cur = input_ids
        for i in range(max_new_tokens):
            logits, _ = model(cur)
            next_logits = logits[:, -1, :]
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            generated.append(int(next_token.item()))
            # next_token has shape (batch, 1)
            cur = torch.cat([cur, next_token], dim=1) if cur.shape[1] + 1 <= block_size else torch.cat([cur[:, 1:], next_token], dim=1)
    end = time.perf_counter()

    total_time = end - start
    toks = len(generated)
    tps = toks / total_time if total_time > 0 else float("inf")

    # parameter count
    try:
        num_params = sum(p.numel() for p in model.parameters())
    except Exception:
        num_params = None

    sample_out = generated[:20]

    return {
        "num_params": num_params,
        "prompt_len": input_ids.shape[1],
        "generated_tokens": toks,
        "total_time_s": total_time,
        "tokens_per_sec": tps,
        "sample_ids": sample_out,
    }


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--mode", choices=["eager", "torchscript", "quantized"], default="eager")
    args = p.parse_args()
    print("Running CPU benchmark mode=", args.mode)
    res = benchmark(args.checkpoint, args.config, args.tokenizer, args.prompt, args.max_new_tokens, args.mode)
    print(res)


if __name__ == "__main__":
    _cli()

import argparse
import torch
import os
from pathlib import Path

from sakicoder.config import Config
from sakicoder.model import GPT


def _infer_vocab_size_from_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    # try common keys
    for key in ("tok_emb.weight", "head.weight", "tok_emb.weight"):
        if key in state:
            return state[key].shape[0]
    # scan for embedding-like tensors
    for k, v in state.items():
        if k.endswith("tok_emb.weight") or k.endswith("head.weight"):
            return v.shape[0]
    raise RuntimeError("Could not infer vocab size from checkpoint")


def prepare_model(checkpoint: str, config: str, device: str = "cpu"):
    if not Path(checkpoint).exists():
        raise FileNotFoundError(checkpoint)
    cfg = Config.from_json(config)
    vocab_size = _infer_vocab_size_from_checkpoint(checkpoint)
    model = GPT(cfg, vocab_size)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.to(device)
    model.eval()
    return model


def export_torchscript(checkpoint: str, config: str, output: str, dry_run: bool = False):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return {"ok": True, "output": output}

    model = prepare_model(checkpoint, config, device="cpu")

    class InferenceWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, idx):
            logits, _ = self.m(idx)
            return logits

    wrapper = InferenceWrapper(model)

    example = torch.zeros((1, min(model.block_size, 8)), dtype=torch.long)
    try:
        ts = torch.jit.trace(wrapper, example)
        ts.save(output)
    except Exception:
        # fallback to scripting
        ts = torch.jit.script(wrapper)
        ts.save(output)

    return {"ok": True, "output": output}


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    print("Exporting TorchScript ->", args.output)
    res = export_torchscript(args.checkpoint, args.config, args.output)
    print("Done:", res)


if __name__ == "__main__":
    _cli()

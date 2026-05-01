import argparse
import torch
import torch.nn as nn
from pathlib import Path

from sakicoder.config import Config
from sakicoder.model import GPT


def _infer_vocab_size_from_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
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


def quantize_dynamic(checkpoint: str, config: str, output: str, dry_run: bool = False):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return {"ok": True, "output": output}

    model = prepare_model(checkpoint, config, device="cpu")
    # Only quantize supported modules (Linear)
    try:
        qmodel = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    except Exception as e:
        return {"ok": False, "reason": str(e)}

    # Save quantized model state dict
    try:
        torch.save({"model_state": qmodel.state_dict()}, output)
    except Exception as e:
        return {"ok": False, "reason": f"save_failed: {e}"}

    return {"ok": True, "output": output}


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    print("Running dynamic quantization ->", args.output)
    res = quantize_dynamic(args.checkpoint, args.config, args.output)
    print("Result:", res)


if __name__ == "__main__":
    _cli()

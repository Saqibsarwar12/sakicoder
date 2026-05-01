import torch
from pathlib import Path


def save_checkpoint(model, optimizer, step: int, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step
    }, path)


def load_checkpoint(model, optimizer, path: str, device="cpu"):
    import os
    if not Path(path).exists():
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("step", 0)

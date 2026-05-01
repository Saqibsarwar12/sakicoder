import argparse
import sys
import numpy as np
import torch
# Ensure repo root is on sys.path when running from scripts/
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from sakicoder.config import Config
from sakicoder.model import GPT
from sakicoder.trainer import Trainer
from sakicoder.utils import get_device, set_seed
from tokenizers import Tokenizer


def main(config_path: str, tokenizer_path: str, data_dir: str, out_dir: str, max_steps: int = 100, mode: str = "pretrain", resume_from: str | None = None):
    config_file = Path(config_path)
    tokenizer_file = Path(tokenizer_path)
    if mode not in {"pretrain", "instruct"}:
        raise ValueError("mode must be one of: pretrain, instruct")

    mode_file = Path(data_dir) / f"{mode}.npy"
    fallback_file = Path(data_dir) / "train.npy"
    data_file = mode_file if mode_file.exists() else fallback_file

    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    if not data_file.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_file}")

    set_seed(42)
    cfg = Config.from_json(str(config_file))
    tok = Tokenizer.from_file(str(tokenizer_file))
    vocab_size = len(tok.get_vocab())
    model = GPT(cfg, vocab_size=vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: total={n_params:,} trainable={n_trainable:,}")
    print(f"Training mode: {mode}")

    device = get_device()
    print(f"Training device: {device}")
    trainer = Trainer(model, cfg, device, mixed_precision=(device.type == "cuda"))
    arr = np.load(str(data_file))
    if arr.size == 0:
        raise ValueError("Processed dataset is empty")

    resume_ckpt = resume_from
    if resume_ckpt is None:
        auto_latest = Path(out_dir) / "latest.pt"
        if auto_latest.exists():
            resume_ckpt = str(auto_latest)

    trainer.train(arr, None, tok, out_dir, max_steps=max_steps, resume_checkpoint=resume_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--mode", choices=["pretrain", "instruct"], default="pretrain")
    parser.add_argument("--resume_from", default=None)
    args = parser.parse_args()
    try:
        main(args.config, args.tokenizer, args.data_dir, args.out_dir, args.max_steps, args.mode, args.resume_from)
    except Exception as exc:
        print(f"train failed: {exc}")
        raise

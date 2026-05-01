import argparse
import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from sakicoder.config import Config
from sakicoder.model import GPT
from tokenizers import Tokenizer


def main(checkpoint, tokenizer_path, config_path, out_path):
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    tok = Tokenizer.from_file(tokenizer_path)
    cfg = Config.from_json(config_path)
    model = GPT(cfg, vocab_size=len(tok.get_vocab()))
    ck = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ck.get("model_state", ck))
    model.eval()
    example = torch.zeros(1, cfg.block_size, dtype=torch.long)
    traced = torch.jit.trace(model, example)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_file))
    print(f"Saved torchscript to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        main(args.checkpoint, args.tokenizer, args.config, args.out)
    except Exception as exc:
        print(f"export_torchscript failed: {exc}")
        raise

import argparse
import sys
from tokenizers import Tokenizer
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from sakicoder.model import GPT
from sakicoder.generation import generate
from sakicoder.config import Config
from sakicoder.utils import get_device


def main(checkpoint: str, tokenizer_path: str, prompt: str, config_path: str, max_new_tokens: int = 64, output_path: str = "outputs/generation_sample.txt"):
    ckpt_path = Path(checkpoint)
    tok_path = Path(tokenizer_path)
    cfg_path = Path(config_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not prompt.strip():
        raise ValueError("Prompt must not be empty")

    tok = Tokenizer.from_file(str(tok_path))
    cfg = Config.from_json(str(cfg_path))
    vocab_size = len(tok.get_vocab())
    model = GPT(cfg, vocab_size=vocab_size)
    ck = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ck["model_state"]) if "model_state" in ck else model.load_state_dict(ck)
    device = get_device()
    model.to(device)
    out = generate(model, tok, device, prompt, max_new_tokens=max_new_tokens)
    print(out)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(out + "\n", encoding="utf-8")
    print(f"Saved generation sample to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--output", default="outputs/generation_sample.txt")
    args = parser.parse_args()
    try:
        main(args.checkpoint, args.tokenizer, args.prompt, args.config, args.max_new_tokens, args.output)
    except Exception as exc:
        print(f"generate failed: {exc}")
        raise

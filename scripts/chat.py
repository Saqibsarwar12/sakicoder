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


def chat_loop(model, tokenizer, device, config):
    history = []
    while True:
        try:
            prompt = input("user> ")
        except EOFError:
            break
        full = "<|user|>\n" + prompt + "\n<|assistant|>\n"
        out = generate(model, tokenizer, device, full, max_new_tokens=200)
        print(out)


def main(checkpoint: str, tokenizer_path: str, config_path: str):
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
    device = get_device()
    model.to(device)
    chat_loop(model, tok, device, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        main(args.checkpoint, args.tokenizer, args.config)
    except Exception as exc:
        print(f"chat failed: {exc}")
        raise

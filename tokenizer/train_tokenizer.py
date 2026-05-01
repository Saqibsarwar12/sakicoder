import argparse
import sys
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from pathlib import Path
# Ensure repo root is on sys.path so package imports work when running the script directly
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tokenizer.tokenizer_utils import SPECIAL_TOKENS


def train(input_dir: str, output: str, vocab_size: int = 8000):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = [str(p) for p in input_path.rglob("*.txt")] + [str(p) for p in input_path.rglob("*.jsonl")]
    if not files:
        raise SystemExit("No input files found in input_dir")

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS + ["<|unk|>"])
    tokenizer.train(files, trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output)
    print(f"Saved tokenizer to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab_size", type=int, default=8000)
    args = parser.parse_args()
    train(args.input_dir, args.output, args.vocab_size)

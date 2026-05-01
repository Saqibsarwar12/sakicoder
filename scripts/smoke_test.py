import subprocess
import sys
from pathlib import Path


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    os_py = sys.executable

    # 1) required imports
    run([os_py, "-c", "import torch, tokenizers, numpy, tqdm, pytest; print('import check OK')"])

    # 2) tokenizer training
    run([
        os_py,
        "tokenizer/train_tokenizer.py",
        "--input_dir",
        "data/raw",
        "--output",
        "tokenizer/tokenizer.json",
        "--vocab_size",
        "8000",
    ])

    # 3) data preparation (sliding window)
    run([
        os_py,
        "scripts/prepare_data.py",
        "--input_dir",
        "data/raw",
        "--tokenizer",
        "tokenizer/tokenizer.json",
        "--output_dir",
        "data/processed",
        "--block_size",
        "512",
        "--stride",
        "256",
    ])

    # 4) tiny training for 20 steps
    run([
        os_py,
        "scripts/train.py",
        "--config",
        "configs/tiny.json",
        "--tokenizer",
        "tokenizer/tokenizer.json",
        "--data_dir",
        "data/processed",
        "--out_dir",
        "checkpoints/tiny-test",
        "--max_steps",
        "20",
    ])

    # 5) generation
    run([
        os_py,
        "scripts/generate.py",
        "--checkpoint",
        "checkpoints/tiny-test/latest.pt",
        "--tokenizer",
        "tokenizer/tokenizer.json",
        "--config",
        "configs/tiny.json",
        "--prompt",
        "<|user|>Create a Python function that adds two numbers.<|assistant|>",
        "--max_new_tokens",
        "120",
        "--output",
        "outputs/generation_sample.txt",
    ])

    # 6) pytest
    run([os_py, "-m", "pytest", "-q"])
    print("\nSmoke test pipeline passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"smoke_test failed: {exc}", file=sys.stderr)
        raise

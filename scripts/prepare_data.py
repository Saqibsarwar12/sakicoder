import argparse
import json
from pathlib import Path
import sys
from tokenizers import Tokenizer
# Ensure repo root is on sys.path when running script directly from scripts/
sys.path.append(str(Path(__file__).resolve().parents[1]))
from sakicoder.dataset import SakiDataset
import numpy as np


def _format_instruct_example(obj: dict) -> str:
    prompt = (obj.get("prompt") or "").strip()
    response = (obj.get("response") or "").strip()
    if not prompt or not response:
        return ""
    return f"<|user|>\n{prompt}\n<|assistant|>\n{response}\n<|eos|>\n"


def _instruction_data_to_ids(input_path: Path, tokenizer: Tokenizer) -> list[int]:
    ids: list[int] = []
    files = sorted(input_path.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No instruction .jsonl files found in {input_path}")

    kept = 0
    broken = 0
    for file in files:
        for raw_line in file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                broken += 1
                continue

            text = _format_instruct_example(obj)
            if not text:
                broken += 1
                continue

            ids.extend(tokenizer.encode(text).ids)
            kept += 1

    if kept == 0:
        raise ValueError("No valid instruction examples found")
    print(f"Instruction examples kept={kept} broken={broken}")
    return ids


def _pretrain_data_to_ids(input_path: Path, tokenizer: Tokenizer, block_size: int) -> list[int]:
    files = [str(p) for p in input_path.rglob("*.txt")] + [str(p) for p in input_path.rglob("*.jsonl")]
    if not files:
        raise FileNotFoundError(f"No .txt/.jsonl files found in {input_path}")
    ds = SakiDataset(files, tokenizer, block_size)
    return list(ds.data)


def main(input_dir: str, tokenizer_path: str, output_dir: str, block_size: int, stride: int, mode: str):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    tok_path = Path(tokenizer_path)
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    if block_size <= 1:
        raise ValueError("block_size must be > 1")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if mode not in {"pretrain", "instruct"}:
        raise ValueError("mode must be one of: pretrain, instruct")

    tok = Tokenizer.from_file(str(tok_path))
    if mode == "instruct":
        token_ids = _instruction_data_to_ids(input_path, tok)
    else:
        token_ids = _pretrain_data_to_ids(input_path, tok, block_size)

    ds = SakiDataset([], tok, block_size)
    ds.data = token_ids
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    arr = ds.get_sliding_chunks(stride=stride)

    # If dataset is too small, create one padded/truncated chunk so training can run as a smoke test.
    if arr.size == 0:
        print("Warning: processed data is empty. Creating one padded sample for smoke test.")
        data_ids = ds.data if hasattr(ds, "data") else []
        ids = list(data_ids)
        if len(ids) >= block_size:
            chunk = np.array(ids[:block_size], dtype=np.int64)
        else:
            chunk = np.zeros(block_size, dtype=np.int64)
            chunk[: len(ids)] = ids
        arr = chunk.reshape(1, block_size)

    mode_file = out / f"{mode}.npy"
    np.save(str(mode_file), arr)
    # Keep compatibility with existing local scripts that expect train.npy
    np.save(str(out / "train.npy"), arr)
    print(f"Saved processed data to {mode_file} with shape={arr.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--mode", choices=["pretrain", "instruct"], default="pretrain")
    args = parser.parse_args()
    try:
        main(args.input_dir, args.tokenizer, args.output_dir, args.block_size, args.stride, args.mode)
    except Exception as exc:
        print(f"prepare_data failed: {exc}")
        raise

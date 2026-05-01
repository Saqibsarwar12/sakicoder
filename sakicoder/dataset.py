import json
from pathlib import Path
import numpy as np
from typing import List
from tokenizers import Tokenizer


class SakiDataset:
    def __init__(self, input_files: List[str], tokenizer: Tokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = []
        for p in input_files:
            p = Path(p)
            if not p.exists():
                continue
            if p.suffix == ".txt":
                text = p.read_text()
                self._add_text(text)
            elif p.suffix == ".jsonl":
                for line in p.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get("content") or obj.get("instruction") or json.dumps(obj)
                        self._add_text(text)
                    except Exception:
                        continue

    def _add_text(self, text: str):
        enc = self.tokenizer.encode(text)
        ids = enc.ids
        self.data.extend(ids)

    def get_chunks(self):
        arr = np.array(self.data, dtype=np.int64)
        n = len(arr) // self.block_size
        arr = arr[: n * self.block_size]
        chunks = arr.reshape(n, self.block_size)
        return chunks

    def get_sliding_chunks(self, stride: int):
        if stride <= 0:
            raise ValueError("stride must be > 0")
        ids = self.data
        if len(ids) < self.block_size:
            return np.empty((0, self.block_size), dtype=np.int64)

        chunks = []
        last_start = len(ids) - self.block_size
        for start in range(0, last_start + 1, stride):
            chunk = ids[start : start + self.block_size]
            if len(chunk) == self.block_size:
                chunks.append(chunk)
        return np.array(chunks, dtype=np.int64)

    def save_npz(self, out_path: str):
        chunks = self.get_chunks()
        np.save(out_path, chunks)

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class Config:
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    dropout: float = 0.1
    batch_size: int = 4
    learning_rate: float = 5e-4

    @staticmethod
    def from_json(path: str) -> "Config":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {path}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON config at {path}: {exc}")

        required = ["n_layer", "n_head", "n_embd", "block_size"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Invalid config: missing keys {missing}")
        return Config(**data)

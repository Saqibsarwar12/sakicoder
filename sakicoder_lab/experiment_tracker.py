from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class ExperimentRecord:
    run_id: str
    config: str
    dataset_path: str
    tokenizer_path: str
    checkpoint_path: str
    steps: int
    loss: float
    eval_path: str
    notes: str
    created_at: str


def log_experiment(run_id: str, config: str, dataset_path: str, tokenizer_path: str, checkpoint_path: str, steps: int, loss: float, eval_path: str, notes: str, created_at: str, path: str = "data/experiments/experiments.jsonl") -> ExperimentRecord:
    record = ExperimentRecord(
        run_id=run_id,
        config=config,
        dataset_path=dataset_path,
        tokenizer_path=tokenizer_path,
        checkpoint_path=checkpoint_path,
        steps=steps,
        loss=loss,
        eval_path=eval_path,
        notes=notes,
        created_at=created_at,
    )
    fp = Path(path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return record


def load_experiments(path: str = "data/experiments/experiments.jsonl") -> List[ExperimentRecord]:
    fp = Path(path)
    if not fp.exists():
        return []
    records: List[ExperimentRecord] = []
    for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        records.append(ExperimentRecord(**json.loads(line)))
    return records


def summarize_experiments(records: Iterable[ExperimentRecord]) -> dict:
    records = list(records)
    steps = [record.steps for record in records]
    losses = [record.loss for record in records]
    return {
        "runs": len(records),
        "avg_steps": sum(steps) / len(steps) if steps else 0,
        "best_loss": min(losses) if losses else None,
        "config_counts": dict(Counter(record.config for record in records)),
    }

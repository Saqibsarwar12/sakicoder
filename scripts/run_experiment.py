from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from sakicoder_lab.experiment_tracker import log_experiment
from scripts.train import main as train_main


def main(config: str, tokenizer: str, data_dir: str, out_dir: str, max_steps: int, mode: str, notes: str = ""):
    train_main(config, tokenizer, data_dir, out_dir, max_steps=max_steps, mode=mode)
    checkpoint_path = str(Path(out_dir) / "latest.pt")
    eval_path = "outputs/eval_results.md"
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
    record = log_experiment(
        run_id=run_id,
        config=config,
        dataset_path=str(Path(data_dir) / f"{mode}.npy"),
        tokenizer_path=tokenizer,
        checkpoint_path=checkpoint_path,
        steps=max_steps,
        loss=0.0,
        eval_path=eval_path,
        notes=notes or f"mode={mode}",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    print(f"Logged experiment {record.run_id}")
    print(f"Experiment log path: data/experiments/experiments.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--mode", choices=["pretrain", "instruct"], default="instruct")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    main(args.config, args.tokenizer, args.data_dir, args.out_dir, args.max_steps, args.mode, args.notes)

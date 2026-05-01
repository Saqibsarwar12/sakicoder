from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from sakicoder_lab.dataset_cards import generate_dataset_card
from sakicoder_lab.sources import load_sources


SOURCE_PATH = "data/sources/sources.jsonl"
OUTPUT_PATH = "data/instruction/DATASET_CARD.md"


def main():
    records = load_sources(SOURCE_PATH)
    if not records:
        raise FileNotFoundError(f"No sources found at {SOURCE_PATH}")

    allowed_examples = sum(1 for r in records if r.allowed_for_training)
    blocked_examples = len(records) - allowed_examples
    card = generate_dataset_card(
        dataset_name="SakiCoder Instruction Dataset",
        purpose="Train a coding and human-instruction model from safe, validated local and MCP-collected sources.",
        records=records,
        allowed_examples=allowed_examples,
        blocked_examples=blocked_examples,
        author="Developed by Saqib Sarwar; solo developer from Srinagar, J&K, India.",
        intended_use="Instruction tuning, coding help, and controlled research evaluation.",
        not_intended_use="Unfiltered scraping, unsafe code execution, or training on unvalidated content.",
    )
    out = Path(OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(card, encoding="utf-8")
    print(f"Saved dataset card to {out}")


if __name__ == "__main__":
    main()

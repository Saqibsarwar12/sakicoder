import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def _canonical_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def deduplicate_records(records: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    seen_prompt_response = set()
    seen_response = set()
    seen_near = set()

    deduped: List[Dict] = []
    exact_pair_duplicates = 0
    exact_code_duplicates = 0
    near_duplicates = 0

    for item in records:
        prompt = str(item.get("prompt", "")).strip()
        response = str(item.get("response", "")).strip()

        pair_key = (prompt, response)
        if pair_key in seen_prompt_response:
            exact_pair_duplicates += 1
            continue

        response_key = response
        if response_key in seen_response:
            exact_code_duplicates += 1
            continue

        near_key = (_canonical_text(prompt), _canonical_text(response))
        if near_key in seen_near:
            near_duplicates += 1
            continue

        seen_prompt_response.add(pair_key)
        seen_response.add(response_key)
        seen_near.add(near_key)
        deduped.append(item)

    stats = {
        "input": len(records),
        "output": len(deduped),
        "exact_pair_duplicates": exact_pair_duplicates,
        "exact_code_duplicates": exact_code_duplicates,
        "near_duplicates": near_duplicates,
        "removed_total": exact_pair_duplicates + exact_code_duplicates + near_duplicates,
    }
    return deduped, stats


def run_dedup(input_path: str, output_path: str) -> Dict[str, int]:
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    records: List[Dict] = []
    for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        records.append(json.loads(s))

    deduped, stats = deduplicate_records(records)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in deduped:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Input records: {stats['input']}")
    print(f"Deduplicated records: {stats['output']}")
    print(f"Removed duplicates: {stats['removed_total']}")
    print(f"Saved -> {output_path}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/instruction/sakicoder_coding_dataset.jsonl")
    parser.add_argument("--output", default="data/instruction/sakicoder_coding_dataset.dedup.jsonl")
    args = parser.parse_args()
    run_dedup(args.input, args.output)


if __name__ == "__main__":
    main()

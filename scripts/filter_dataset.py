from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from sakicoder_lab.quality_filters import check_quality, redact_secrets


def main(input_path: str, output_path: str, rejected_path: str):
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    kept = []
    rejected = []
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            rejected.append({"raw": s, "reason": ["invalid_json"], "redacted_text": ""})
            continue

        prompt = str(obj.get("prompt", ""))
        response = str(obj.get("response", ""))
        result = check_quality(f"{prompt}\n{response}")
        redacted_prompt = redact_secrets(prompt).strip()
        redacted_response = redact_secrets(response).strip()
        cleaned = {
            **obj,
            "prompt": redacted_prompt,
            "response": redacted_response,
        }
        if result["keep"]:
            kept.append(cleaned)
        else:
            rejected.append({**obj, "issues": result["issues"], "redacted_text": result["redacted_text"]})

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rej = Path(rejected_path)
    rej.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with rej.open("w", encoding="utf-8") as f:
        for item in rejected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Kept {len(kept)} examples -> {out}")
    print(f"Rejected {len(rejected)} examples -> {rej}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rejected", default="data/instruction/rejected_examples.jsonl")
    args = parser.parse_args()
    main(args.input, args.output, args.rejected)

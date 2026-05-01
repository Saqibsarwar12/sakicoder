import argparse
import json
from collections import Counter
from pathlib import Path


def main(input_path: str, tokenizer_path: str | None = None):
    fp = Path(input_path)
    if not fp.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    examples = 0
    broken = 0
    empty = 0
    by_type = Counter()
    by_lang = Counter()
    max_prompt = 0
    max_response = 0
    est_tokens = 0
    tokenizer = None

    if tokenizer_path:
        tok_fp = Path(tokenizer_path)
        if tok_fp.exists():
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(tok_fp))

    for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            broken += 1
            continue

        prompt = (obj.get("prompt") or "")
        response = (obj.get("response") or "")
        if not prompt.strip() or not response.strip():
            empty += 1
            continue

        examples += 1
        by_type[(obj.get("type") or "unknown")] += 1
        by_lang[(obj.get("language") or "unknown")] += 1
        max_prompt = max(max_prompt, len(prompt))
        max_response = max(max_response, len(response))

        if tokenizer is not None:
            est_tokens += len(tokenizer.encode(prompt + "\n" + response).ids)

    print(f"examples: {examples}")
    print(f"broken_examples: {broken}")
    print(f"empty_examples: {empty}")
    print("count_by_type:")
    for k, v in sorted(by_type.items()):
        print(f"  {k}: {v}")
    print("count_by_language:")
    for k, v in sorted(by_lang.items()):
        print(f"  {k}: {v}")
    print(f"longest_prompt_chars: {max_prompt}")
    print(f"longest_response_chars: {max_response}")
    if tokenizer is not None:
        print(f"estimated_token_count: {est_tokens}")
    else:
        print("estimated_token_count: tokenizer not provided or missing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    args = parser.parse_args()
    try:
        tok = args.tokenizer if Path(args.tokenizer).exists() else None
        main(args.input, tok)
    except Exception as exc:
        print(f"dataset_stats failed: {exc}")
        raise

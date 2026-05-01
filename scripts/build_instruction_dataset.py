import json
from pathlib import Path
from typing import Dict, List, Tuple

ALLOWED_TYPES = {"instruction", "code", "debug", "agent"}
ALLOWED_LANGUAGES = {
    "python",
    "javascript",
    "typescript",
    "kotlin",
    "dart",
    "html",
    "css",
    "bash",
    "json",
    "yaml",
    "general",
}


def normalize_example(obj: dict) -> dict | None:
    t = (obj.get("type") or "instruction").strip().lower()
    lang = (obj.get("language") or "general").strip().lower()
    prompt = (obj.get("prompt") or "").strip()
    response = (obj.get("response") or "").strip()

    # Try common alternate keys.
    if not prompt:
        prompt = (obj.get("instruction") or obj.get("input") or obj.get("question") or "").strip()
    if not response:
        response = (obj.get("output") or obj.get("answer") or obj.get("completion") or "").strip()

    if t not in ALLOWED_TYPES:
        t = "instruction"
    if lang not in ALLOWED_LANGUAGES:
        lang = "general"

    if not prompt or not response:
        return None

    return {
        "type": t,
        "language": lang,
        "prompt": prompt,
        "response": response,
    }


def parse_role_messages(lines: List[str], default_type: str = "instruction", default_language: str = "general") -> List[dict]:
    out: List[dict] = []
    pending_prompt = None
    for line in lines:
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue

        role = (obj.get("role") or "").strip().lower()
        content = (obj.get("content") or "").strip()
        if not content:
            continue

        if role == "user":
            pending_prompt = content
        elif role == "assistant" and pending_prompt:
            out.append(
                {
                    "type": default_type,
                    "language": default_language,
                    "prompt": pending_prompt,
                    "response": content,
                }
            )
            pending_prompt = None
    return out


def build_dataset(raw_dir: str = "data/raw", seed_dir: str = "data/seed", output_path: str = "data/instruction/sakicoder_instructions.jsonl") -> Tuple[int, int]:
    input_files: List[Path] = []
    for d in [Path(raw_dir), Path(seed_dir)]:
        if d.exists():
            input_files.extend(sorted(d.rglob("*.jsonl")))

    if not input_files:
        raise FileNotFoundError("No .jsonl files found under data/raw or data/seed")

    normalized: List[dict] = []
    broken = 0

    for fp in input_files:
        lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()

        # If file uses role/content chat records, parse pairs.
        role_pairs = parse_role_messages(lines)
        if role_pairs:
            normalized.extend(role_pairs)
            continue

        for line in lines:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                broken += 1
                continue

            item = normalize_example(obj)
            if item is None:
                broken += 1
                continue
            normalized.append(item)

    # Deduplicate by tuple key.
    seen = set()
    deduped = []
    for ex in normalized:
        key = (ex["type"], ex["language"], ex["prompt"], ex["response"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ex)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for ex in deduped:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Built instruction dataset: {len(deduped)} examples -> {out}")
    print(f"Broken/skipped records: {broken}")
    return len(deduped), broken


if __name__ == "__main__":
    try:
        build_dataset()
    except Exception as exc:
        print(f"build_instruction_dataset failed: {exc}")
        raise

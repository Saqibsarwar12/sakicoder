import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sakicoder_lab.licenses import is_license_allowed, normalize_license
from sakicoder_lab.quality_filters import SECRET_PATTERNS, check_quality

ALLOWED_TYPES = {"code", "instruction", "debug", "agent"}
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

CODE_EXT_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".kt": "kotlin",
    ".dart": "dart",
    ".html": "html",
    ".css": "css",
    ".sh": "bash",
    ".bash": "bash",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
}

PERSONAL_DATA_PATTERNS = [
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b\+?\d{1,3}[ -]?\(?\d{2,4}\)?[ -]?\d{3,4}[ -]?\d{3,4}\b"),
]


def load_source_records(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Source records not found: {path}")

    out: List[Dict] = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(json.loads(s))
    return out


def contains_secret_or_pii(text: str) -> bool:
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        return True
    if any(pattern.search(text) for pattern in PERSONAL_DATA_PATTERNS):
        return True
    return False


def normalize_instruction_record(obj: Dict, source_id: str, license_name: str) -> Dict | None:
    t = str(obj.get("type", "instruction")).strip().lower()
    language = str(obj.get("language", "general")).strip().lower()
    prompt = str(obj.get("prompt") or obj.get("instruction") or obj.get("input") or "").strip()
    response = str(obj.get("response") or obj.get("output") or obj.get("answer") or "").strip()

    if t not in ALLOWED_TYPES:
        t = "instruction"
    if language not in ALLOWED_LANGUAGES:
        language = "general"

    if not prompt or not response:
        return None

    merged = f"{prompt}\n{response}"
    if contains_secret_or_pii(merged):
        return None

    quality = check_quality(merged)
    if not quality["keep"]:
        return None

    return {
        "type": t,
        "language": language,
        "prompt": prompt,
        "response": response,
        "source_id": source_id,
        "license": license_name,
    }


def iter_local_content(path: Path) -> Iterable[Tuple[str, str]]:
    if path.is_file():
        yield str(path), path.read_text(encoding="utf-8", errors="ignore")
        return

    for f in sorted(path.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in CODE_EXT_LANGUAGE and f.suffix.lower() != ".jsonl":
            continue
        yield str(f), f.read_text(encoding="utf-8", errors="ignore")


def to_code_example(file_path: str, content: str, source_id: str, license_name: str, declared_language: str) -> Dict | None:
    body = content.strip()
    if not body:
        return None
    if contains_secret_or_pii(body):
        return None

    quality = check_quality(body)
    if not quality["keep"]:
        return None

    suffix = Path(file_path).suffix.lower()
    language = CODE_EXT_LANGUAGE.get(suffix, declared_language if declared_language in ALLOWED_LANGUAGES else "general")
    return {
        "type": "code",
        "language": language,
        "prompt": f"Read and learn from this {language} example from {Path(file_path).name}.",
        "response": body,
        "source_id": source_id,
        "license": license_name,
    }


def normalize_dataset(
    sources_path: str = "data/sources/coding_sources.jsonl",
    output_path: str = "data/instruction/sakicoder_coding_dataset.jsonl",
) -> Dict[str, int]:
    records = load_source_records(sources_path)
    accepted: List[Dict] = []
    skipped = 0

    for record in records:
        source_id = str(record.get("source_id", "unknown"))
        source_type = str(record.get("source_type", "")).strip().lower()
        path_or_url = str(record.get("url_or_path", ""))
        declared_language = str(record.get("language", "general")).strip().lower()
        license_name = normalize_license(record.get("license"))
        allowed_for_training = bool(record.get("allowed_for_training", False))

        if not allowed_for_training:
            skipped += 1
            continue
        if not is_license_allowed(license_name):
            skipped += 1
            continue

        # Safe-by-default: this script does not fetch remote content.
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            skipped += 1
            continue

        local_path = Path(path_or_url)
        if not local_path.exists():
            skipped += 1
            continue

        for file_path, content in iter_local_content(local_path):
            if file_path.endswith(".jsonl"):
                for line in content.splitlines():
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    normalized = normalize_instruction_record(obj, source_id, license_name)
                    if normalized is None:
                        skipped += 1
                        continue
                    accepted.append(normalized)
                continue

            # Code file path
            normalized_code = to_code_example(file_path, content, source_id, license_name, declared_language)
            if normalized_code is None:
                skipped += 1
                continue
            accepted.append(normalized_code)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in accepted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote normalized coding dataset: {len(accepted)} examples -> {output_path}")
    print(f"Skipped records/examples: {skipped}")
    return {"kept": len(accepted), "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="data/sources/coding_sources.jsonl")
    parser.add_argument("--output", default="data/instruction/sakicoder_coding_dataset.jsonl")
    args = parser.parse_args()
    normalize_dataset(sources_path=args.sources, output_path=args.output)


if __name__ == "__main__":
    main()

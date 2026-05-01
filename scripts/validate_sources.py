from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from sakicoder_lab.licenses import is_license_allowed, normalize_license
from sakicoder_lab.sources import load_sources, source_summary


SOURCE_PATH = "data/sources/sources.jsonl"


def main():
    records = load_sources(SOURCE_PATH)
    if not records:
        raise FileNotFoundError(f"No sources found at {SOURCE_PATH}")

    summary = source_summary(records)
    unknown = sum(1 for record in records if normalize_license(record.license) == "UNKNOWN")
    allowed = sum(1 for record in records if is_license_allowed(record.license) and record.allowed_for_training)
    blocked = len(records) - allowed

    print(f"total sources: {summary['total_sources']}")
    print(f"allowed for training: {allowed}")
    print(f"blocked: {blocked}")
    print(f"unknown licenses: {unknown}")
    print("language breakdown:")
    for language, count in sorted(summary["by_language"].items()):
        print(f"  {language}: {count}")


if __name__ == "__main__":
    main()

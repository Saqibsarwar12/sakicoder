import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sakicoder_lab.licenses import normalize_license

SUPPORTED_SOURCE_TYPES = {
    "local_files",
    "approved_github_repo",
    "official_docs_examples",
    "user_owned_project",
    "permissive_sample_code",
}


@dataclass
class CodingSourceRecord:
    source_id: str
    source_type: str
    url_or_path: str
    license: str
    language: str
    allowed_for_training: bool
    notes: str = ""


def validate_source_record(record: dict) -> tuple[bool, str]:
    required = {
        "source_id",
        "source_type",
        "url_or_path",
        "license",
        "language",
        "allowed_for_training",
        "notes",
    }
    missing = sorted(required - set(record.keys()))
    if missing:
        return False, f"missing fields: {missing}"

    if record["source_type"] not in SUPPORTED_SOURCE_TYPES:
        return False, f"unsupported source_type: {record['source_type']}"

    if not isinstance(record["allowed_for_training"], bool):
        return False, "allowed_for_training must be boolean"

    if not str(record["source_id"]).strip():
        return False, "source_id must be non-empty"

    return True, ""


def default_source_plan() -> List[CodingSourceRecord]:
    return [
        CodingSourceRecord(
            source_id="local-seed-001",
            source_type="local_files",
            url_or_path="data/seed",
            license="MIT",
            language="general",
            allowed_for_training=True,
            notes="Locally maintained seed examples.",
        ),
        CodingSourceRecord(
            source_id="user-project-001",
            source_type="user_owned_project",
            url_or_path="sakicoder",
            license="MIT",
            language="python",
            allowed_for_training=True,
            notes="User-owned project code snippets for style and structure.",
        ),
        CodingSourceRecord(
            source_id="approved-gh-001",
            source_type="approved_github_repo",
            url_or_path="https://github.com/example/permissive-repo",
            license="Apache-2.0",
            language="python",
            allowed_for_training=False,
            notes="Manual approval required before local ingestion. No scraping.",
        ),
        CodingSourceRecord(
            source_id="official-docs-001",
            source_type="official_docs_examples",
            url_or_path="https://docs.python.org/3/tutorial/",
            license="PSF-2.0",
            language="python",
            allowed_for_training=False,
            notes="Use only snippets copied with license review and attribution.",
        ),
        CodingSourceRecord(
            source_id="permissive-sample-001",
            source_type="permissive_sample_code",
            url_or_path="data/raw",
            license="MIT",
            language="general",
            allowed_for_training=True,
            notes="Local permissive sample code corpus.",
        ),
    ]


def save_records(records: Iterable[CodingSourceRecord], output_path: str) -> int:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            payload = asdict(record)
            payload["license"] = normalize_license(payload["license"])
            ok, reason = validate_source_record(payload)
            if not ok:
                raise ValueError(f"Invalid source record {payload.get('source_id', '?')}: {reason}")
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def main(output: str = "data/sources/coding_sources.jsonl") -> None:
    records = default_source_plan()
    count = save_records(records, output)
    print(f"Wrote {count} safe source-plan records -> {output}")
    print("This script creates a training source plan only. It does not scrape or execute remote code.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/sources/coding_sources.jsonl")
    args = parser.parse_args()
    main(output=args.output)

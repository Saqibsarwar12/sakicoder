from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from .sources import SourceRecord


def generate_dataset_card(
    dataset_name: str,
    purpose: str,
    records: Iterable[SourceRecord],
    allowed_examples: int,
    blocked_examples: int,
    author: str = "Developed by Saqib Sarwar; solo developer from Srinagar, J&K, India.",
    known_risks: list[str] | None = None,
    intended_use: str = "",
    not_intended_use: str = "",
) -> str:
    records = list(records)
    license_counts = Counter(record.license for record in records)
    language_counts = Counter(record.language for record in records)
    source_count = len(records)
    known_risks = known_risks or [
        "Small starter dataset may overfit on narrow patterns.",
        "Source quality depends on future MCP collection and validation.",
        "License checks must be re-run before any training expansion.",
    ]

    lines = [
        f"# {dataset_name}",
        "",
        f"Purpose: {purpose}",
        f"Author: {author}",
        "",
        f"Source count: {source_count}",
        f"Allowed examples: {allowed_examples}",
        f"Blocked examples: {blocked_examples}",
        "",
        "## License summary",
    ]
    for license_name, count in sorted(license_counts.items()):
        lines.append(f"- {license_name}: {count}")
    lines.extend(["", "## Language summary"])
    for language, count in sorted(language_counts.items()):
        lines.append(f"- {language}: {count}")
    lines.extend([
        "",
        "## Known risks",
    ])
    for risk in known_risks:
        lines.append(f"- {risk}")
    lines.extend([
        "",
        "## Intended use",
        intended_use or "Instruction tuning, coding help, debugging guidance, and controlled research evaluation.",
        "",
        "## Not intended use",
        not_intended_use or "Unfiltered scraping, unsafe code execution, or production-critical autonomous decisions.",
    ])
    return "\n".join(lines) + "\n"

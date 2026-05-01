from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class SourceRecord:
    source_id: str
    source_type: str
    url_or_path: str
    title: str
    license: str
    language: str
    collected_at: str
    allowed_for_training: bool
    notes: str = ""


def load_sources(path: str) -> List[SourceRecord]:
    fp = Path(path)
    if not fp.exists():
        return []
    records: List[SourceRecord] = []
    for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        records.append(SourceRecord(**obj))
    return records


def save_sources(records: Iterable[SourceRecord], path: str) -> None:
    fp = Path(path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def add_source_record(
    source_id: str,
    source_type: str,
    url_or_path: str,
    title: str,
    license: str,
    language: str,
    collected_at: str,
    allowed_for_training: bool,
    notes: str = "",
) -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        source_type=source_type,
        url_or_path=url_or_path,
        title=title,
        license=license,
        language=language,
        collected_at=collected_at,
        allowed_for_training=allowed_for_training,
        notes=notes,
    )


def source_summary(records: Iterable[SourceRecord]) -> dict:
    records = list(records)
    licenses = Counter(record.license for record in records)
    languages = Counter(record.language for record in records)
    types = Counter(record.source_type for record in records)
    allowed = sum(1 for record in records if record.allowed_for_training)
    return {
        "total_sources": len(records),
        "allowed_for_training": allowed,
        "blocked": len(records) - allowed,
        "by_license": dict(licenses),
        "by_language": dict(languages),
        "by_source_type": dict(types),
    }

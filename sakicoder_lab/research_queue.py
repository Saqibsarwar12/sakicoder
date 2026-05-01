from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class ResearchTask:
    task_id: str
    task_type: str
    query: str
    target_source: str
    priority: str
    status: str
    notes: str = ""


def create_task(task_id: str, task_type: str, query: str, target_source: str, priority: str = "medium", status: str = "pending", notes: str = "") -> ResearchTask:
    return ResearchTask(
        task_id=task_id,
        task_type=task_type,
        query=query,
        target_source=target_source,
        priority=priority,
        status=status,
        notes=notes,
    )


def load_queue(path: str) -> List[ResearchTask]:
    fp = Path(path)
    if not fp.exists():
        return []
    tasks: List[ResearchTask] = []
    for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        tasks.append(ResearchTask(**json.loads(line)))
    return tasks


def save_queue(tasks: Iterable[ResearchTask], path: str) -> None:
    fp = Path(path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(asdict(task), ensure_ascii=False) + "\n")


def next_pending_task(tasks: Iterable[ResearchTask]) -> Optional[ResearchTask]:
    for task in tasks:
        if task.status == "pending":
            return task
    return None

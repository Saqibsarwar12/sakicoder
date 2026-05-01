from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from sakicoder_lab.research_queue import create_task, save_queue


OUTPUT_PATH = "data/research/research_queue.jsonl"


def main():
    tasks = [
        create_task("task-python-docs", "docs", "Python official docs examples for typing, pathlib, dataclasses", "python docs", "high", "pending", "Collect only official docs examples."),
        create_task("task-js-docs", "docs", "JavaScript MDN examples for arrays, promises, fetch", "javascript docs", "high", "pending", "Collect only MDN or official docs."),
        create_task("task-ts-docs", "docs", "TypeScript handbook examples for types, unions, generics", "typescript docs", "high", "pending", "Collect official handbook examples."),
        create_task("task-react-docs", "docs", "React docs examples for hooks and components", "react docs", "high", "pending", "Collect official React docs examples."),
        create_task("task-fastapi-docs", "docs", "FastAPI docs examples for endpoints and validation", "fastapi docs", "high", "pending", "Collect official FastAPI examples."),
        create_task("task-kotlin-docs", "docs", "Kotlin Android docs examples for activities and viewmodels", "kotlin android docs", "high", "pending", "Collect official Android/Kotlin docs examples."),
        create_task("task-flutter-docs", "docs", "Flutter docs examples for widgets and state", "flutter docs", "high", "pending", "Collect official Flutter examples."),
        create_task("task-gradle-errors", "research", "Gradle error examples and fixes from official docs", "gradle docs", "medium", "pending", "Collect safe error messages and official fixes."),
        create_task("task-bash-linux", "docs", "Bash and Linux command examples from official man pages", "bash linux docs", "medium", "pending", "Collect command docs only."),
    ]
    save_queue(tasks, OUTPUT_PATH)
    print(f"Saved research queue to {OUTPUT_PATH} ({len(tasks)} tasks)")


if __name__ == "__main__":
    main()

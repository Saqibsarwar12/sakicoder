import os
import shlex
import subprocess
from typing import List


SAFE_BLOCKED = ["rm -rf", "shutdown", "reboot", "mkfs", "del /s", "format"]


def _is_safe(cmd: str) -> bool:
    for b in SAFE_BLOCKED:
        if b in cmd:
            return False
    return True


def run_command(command: str, cwd: str = None) -> dict:
    if not _is_safe(command):
        return {"ok": False, "error": "Command blocked by safety policy"}
    print(f"Running (safe mode): {command}")
    try:
        proc = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=30)
        return {"ok": True, "stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def list_files(path: str) -> List[str]:
    return [str(p) for p in __import__('pathlib').Path(path).rglob('*')]


def read_file(path: str) -> str:
    return open(path, 'r', encoding='utf-8', errors='ignore').read()


def write_file(path: str, content: str) -> dict:
    Path = __import__('pathlib').Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {"ok": True}

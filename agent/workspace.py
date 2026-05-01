from pathlib import Path


def safe_list(path: str):
    p = Path(path)
    if not p.exists():
        return []
    return [str(x) for x in p.iterdir()]

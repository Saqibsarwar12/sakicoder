import argparse
import zipfile
from pathlib import Path
from typing import Iterable, List

EXCLUDED_PARTS = {
    "checkpoints",
    "outputs",
    "exports",
    ".pytest_cache",
    "__pycache__",
    ".git",
}


def should_exclude(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDED_PARTS:
        return True
    if path.suffix in {".pyc", ".pyo"}:
        return True
    if path.name.endswith(".zip") and "data" in parts and "colab_package" in parts:
        return True
    return False


def iter_included_paths(repo_root: Path, include_items: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for rel in include_items:
        target = repo_root / rel
        if not target.exists():
            continue
        if target.is_file():
            if not should_exclude(target.relative_to(repo_root)):
                files.append(target)
            continue

        for f in sorted(target.rglob("*")):
            if not f.is_file():
                continue
            rel_path = f.relative_to(repo_root)
            if should_exclude(rel_path):
                continue
            files.append(f)
    return files


def colab_readme_text() -> str:
    return (
        "# SakiCoder Colab Dataset Package\n\n"
        "This package contains safe, local training assets for Colab runs.\n"
        "Steps:\n"
        "1. Upload this zip to Colab and unzip into the repo folder.\n"
        "2. Train tokenizer and prepare data.\n"
        "3. Start training with scripts/train.py.\n\n"
        "Security:\n"
        "- No remote scraping is required.\n"
        "- Unknown or blocked licenses should be excluded before packaging.\n"
    )


def create_package(
    output_path: str = "data/colab_package/sakicoder_colab_training_package.zip",
    repo_root: str = ".",
) -> Path:
    root = Path(repo_root).resolve()
    output = Path(output_path)
    if not output.is_absolute():
        output = (root / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    include_items = [
        "data/instruction/sakicoder_coding_dataset.dedup.jsonl",
        "data/seed",
        "configs",
        "tokenizer",
        "scripts",
        "sakicoder",
    ]

    files = iter_included_paths(root, include_items)

    readme_rel = Path("data/colab_package/COLAB_PACKAGE_README.md")
    readme_abs = root / readme_rel
    readme_abs.parent.mkdir(parents=True, exist_ok=True)
    readme_abs.write_text(colab_readme_text(), encoding="utf-8")
    files.append(readme_abs)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            arcname = f.relative_to(root)
            zf.write(f, arcname.as_posix())

    print(f"Created Colab package: {output}")
    print(f"Included files: {len(files)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/colab_package/sakicoder_colab_training_package.zip")
    args = parser.parse_args()
    create_package(output_path=args.output)


if __name__ == "__main__":
    main()

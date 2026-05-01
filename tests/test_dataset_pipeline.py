import json
from pathlib import Path

from scripts.collect_coding_sources import validate_source_record
from scripts.normalize_code_dataset import normalize_dataset
from scripts.deduplicate_dataset import deduplicate_records
from scripts.package_colab_dataset import create_package


def test_source_record_validation():
    ok, reason = validate_source_record(
        {
            "source_id": "s1",
            "source_type": "local_files",
            "url_or_path": "data/seed",
            "license": "MIT",
            "language": "python",
            "allowed_for_training": True,
            "notes": "safe local source",
        }
    )
    assert ok, reason


def test_dataset_normalization(tmp_path: Path):
    source_dir = tmp_path / "sample_src"
    source_dir.mkdir(parents=True)
    sample_jsonl = source_dir / "samples.jsonl"
    sample_jsonl.write_text(
        "\n".join(
            [
                json.dumps({
                    "type": "instruction",
                    "language": "python",
                    "prompt": "Write add function",
                    "response": "def add(a, b): return a + b",
                }),
                json.dumps({
                    "type": "instruction",
                    "language": "python",
                    "prompt": "",
                    "response": "invalid empty prompt",
                }),
            ]
        ),
        encoding="utf-8",
    )

    sources = tmp_path / "sources.jsonl"
    sources.write_text(
        json.dumps(
            {
                "source_id": "tmp-src",
                "source_type": "local_files",
                "url_or_path": str(source_dir),
                "license": "MIT",
                "language": "python",
                "allowed_for_training": True,
                "notes": "tmp test source",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "normalized.jsonl"
    stats = normalize_dataset(str(sources), str(out))
    assert stats["kept"] == 1
    lines = [x for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 1


def test_deduplication():
    records = [
        {"prompt": "p", "response": "r"},
        {"prompt": "p", "response": "r"},
        {"prompt": "P ", "response": "R"},
        {"prompt": "x", "response": "r"},
    ]
    deduped, stats = deduplicate_records(records)
    assert len(deduped) == 1
    assert stats["removed_total"] == 3


def test_package_creation_path_handling(tmp_path: Path):
    # Build a miniature repo structure expected by the package script.
    (tmp_path / "data/instruction").mkdir(parents=True)
    (tmp_path / "data/seed").mkdir(parents=True)
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "tokenizer").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "sakicoder").mkdir(parents=True)

    (tmp_path / "data/instruction/sakicoder_coding_dataset.dedup.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "data/seed/sample.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "configs/tiny.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "tokenizer/train_tokenizer.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "scripts/train.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "sakicoder/model.py").write_text("print('ok')\n", encoding="utf-8")

    out_zip = tmp_path / "data/colab_package/pkg.zip"
    result = create_package(str(out_zip), repo_root=str(tmp_path))
    assert result.exists()

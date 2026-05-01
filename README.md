# SakiCoder

SakiCoder is a from-scratch PyTorch Transformer project focused on coding and instruction-following. It provides a minimal, trainable GPT-style model built from first principles so you can train and extend it locally or in Colab.

## Author

Developed by Saqib Sarwar  
Solo developer from Srinagar, J&K, India

Key points:
- Implemented from scratch in PyTorch (no pretrained model imports).
- Tokenizer training using Hugging Face `tokenizers` (BPE).
- Data pipeline supports .txt and JSONL instruction/chat examples.
- Training loop with AdamW, checkpointing, and optional mixed precision.
- CPU-friendly tiny config for development, larger configs for Colab GPU training.

This repo is a foundation (v0.1). The model quality will be small until trained on more/better data.

See the top-level `README.md` for usage examples and the `notebooks/colab_train_sakicoder.ipynb` for a Colab workflow.

## End-to-end smoke test

Run the full local verification pipeline (imports, tokenizer training, data prep, tiny training, generation, pytest):

```bash
python scripts/smoke_test.py
```

This command will also write a sample generation output to `outputs/generation_sample.txt`.

## Continuous integration

GitHub Actions runs lightweight verification on every push and pull request.
The CI pipeline installs dependencies, runs the smoke test, runs `pytest -q`,
and validates export imports with `pytest tests/test_export_imports.py -q`.
It also runs a short CPU benchmark after smoke test completion.

## Dataset preparation for Colab training

Build a safe coding dataset pipeline before long Colab GPU runs:

```bash
python scripts/collect_coding_sources.py
python scripts/normalize_code_dataset.py
python scripts/deduplicate_dataset.py
python scripts/package_colab_dataset.py
```

The pipeline is safe-by-default: it does not scrape random repositories, skips
unknown/blocked licenses, filters low-quality/minified/binary-looking content,
and excludes examples with secrets or personal data.

## Training stages

Stage 1: tokenizer training
- Train a code/instruction-aware BPE tokenizer from local data.

Stage 2: tiny smoke training
- Run a short CPU training job to verify model, checkpointing, and generation.

Stage 3: instruction dataset building
- Build and normalize instruction JSONL into `data/instruction/sakicoder_instructions.jsonl`.

Stage 4: Colab GPU training
- Use Colab notebook flow for longer runs and larger configs with checkpoint resume.

Stage 5: evaluation
- Run category-based generation evaluation and save reports to `outputs/eval_results.md`.

Stage 6: future quantization/export
- Export TorchScript now and add later quantization targets for CPU deployment.

## Safe MCP-ready research workflow

1. Create a research queue.
2. Use MCP/search tools manually or through an agent to collect candidate sources.
3. Save source metadata.
4. Validate licenses.
5. Filter and redact dataset examples.
6. Build a dataset card.
7. Train only on approved filtered data.
8. Log experiment results.

Commands:

```bash
python scripts/create_research_queue.py
python scripts/validate_sources.py
python scripts/filter_dataset.py --input data/instruction/sakicoder_instructions.jsonl --output data/instruction/sakicoder_instructions.filtered.jsonl
python scripts/make_dataset_card.py
python scripts/run_experiment.py --config configs/tiny.json --tokenizer tokenizer/tokenizer.json --data_dir data/processed --out_dir checkpoints/experiment-test --max_steps 20 --mode instruct
```
# CPU export and quantization

SakiCoder training runs in PyTorch. This `export/` folder provides a lightweight foundation
to export trained checkpoints for CPU inference and to measure performance.

Paths and commands

- TorchScript export (eager/scripted wrapper):

```bash
python export/export_torchscript.py \
  --checkpoint checkpoints/tiny-test/latest.pt \
  --config configs/tiny.json \
  --output exports/torchscript/sakicoder_tiny.pt
```

- ONNX export:

```bash
python export/export_onnx.py \
  --checkpoint checkpoints/tiny-test/latest.pt \
  --config configs/tiny.json \
  --output exports/onnx/sakicoder_tiny.onnx
```

If the `onnx` package is not installed the script will inform you and skip.

- Dynamic quantization (PyTorch):

```bash
python export/quantize_dynamic.py \
  --checkpoint checkpoints/tiny-test/latest.pt \
  --config configs/tiny.json \
  --output exports/quantized/sakicoder_dynamic.pt
```

- CPU benchmark (simple greedy generation):

```bash
python export/benchmark_cpu.py \
  --checkpoint checkpoints/tiny-test/latest.pt \
  --config configs/tiny.json \
  --tokenizer tokenizer/tokenizer.json \
  --prompt "<|user|>Create a Python function that adds two numbers.<|assistant|>" \
  --max_new_tokens 50 \
  --mode eager
```

Notes
- These tools are for inference/export only and do not replace the PyTorch training code.
- The quality of exported models depends on the training you run locally or in Colab.
- Future exports (e.g. GGUF/llama.cpp) may require architecture conversion and are not included here.

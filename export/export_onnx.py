import argparse
import torch
from pathlib import Path

from sakicoder.config import Config
from sakicoder.model import GPT


def _infer_vocab_size_from_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    for k, v in state.items():
        if k.endswith("tok_emb.weight") or k.endswith("head.weight"):
            return v.shape[0]
    raise RuntimeError("Could not infer vocab size from checkpoint")


def prepare_model(checkpoint: str, config: str, device: str = "cpu"):
    if not Path(checkpoint).exists():
        raise FileNotFoundError(checkpoint)
    cfg = Config.from_json(config)
    vocab_size = _infer_vocab_size_from_checkpoint(checkpoint)
    model = GPT(cfg, vocab_size)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.to(device)
    model.eval()
    return model


def export_onnx(checkpoint: str, config: str, output: str, opset: int = 17, dry_run: bool = False):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return {"ok": True, "output": output}

    try:
        import onnx  # noqa: F401
    except Exception:
        print("ONNX package is not installed. To enable ONNX export please `pip install onnx`.")
        return {"ok": False, "reason": "missing_onnx"}

    model = prepare_model(checkpoint, config, device="cpu")
    example = torch.zeros((1, min(model.block_size, 8)), dtype=torch.long)

    def _forward(idx):
        logits, _ = model(idx)
        return logits

    # Use torch.onnx.export
    try:
        torch.onnx.export(model, example, output, opset_version=opset, input_names=["input_ids"], output_names=["logits"])
    except Exception as e:
        return {"ok": False, "reason": str(e)}

    return {"ok": True, "output": output}


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()
    print("Exporting ONNX ->", args.output)
    res = export_onnx(args.checkpoint, args.config, args.output, opset=args.opset)
    print("Result:", res)


if __name__ == "__main__":
    _cli()

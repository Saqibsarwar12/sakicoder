import argparse
import sys
from datetime import datetime, timezone
from tokenizers import Tokenizer
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from sakicoder.model import GPT
from sakicoder.config import Config
from sakicoder.generation import generate


def main(checkpoint: str, tokenizer_path: str, config_path: str):
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    tok = Tokenizer.from_file(tokenizer_path)
    cfg = Config.from_json(config_path)
    model = GPT(cfg, vocab_size=len(tok.get_vocab()))
    ck = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ck.get("model_state", ck))

    eval_prompts = [
        ("Python function generation", "<|user|>Write a Python function to compute factorial recursively.<|assistant|>"),
        ("JavaScript bug fix", "<|user|>Fix this JS bug: function add(a,b){ return a-b }<|assistant|>"),
        ("React component creation", "<|user|>Create a React button component with loading state.<|assistant|>"),
        ("FastAPI endpoint creation", "<|user|>Create a FastAPI GET endpoint /health returning status ok.<|assistant|>"),
        ("Kotlin Android explanation", "<|user|>Explain what onCreate does in an Android Activity.<|assistant|>"),
        ("Flutter widget generation", "<|user|>Create a Flutter StatelessWidget for a profile card.<|assistant|>"),
        ("Bash command explanation", "<|user|>Explain: grep -R \"TODO\" src/ <|assistant|>"),
        ("Gradle error fix", "<|user|>Fix this error: Could not resolve com.android.tools.build:gradle.<|assistant|>"),
        ("Human instruction understanding", "<|user|>Plan steps to build a small notes app with auth and search.<|assistant|>"),
    ]

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "eval_results.md"
    lines = [
        "# SakiCoder Evaluation Results",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    for category, prompt in eval_prompts:
        text = generate(model, tok, torch.device("cpu"), prompt, max_new_tokens=180)
        lines.extend([
            f"## {category}",
            "",
            "### Prompt",
            "```text",
            prompt,
            "```",
            "",
            "### Output",
            "```text",
            text,
            "```",
            "",
        ])
        print(f"[{category}]\n{text}\n")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved evaluation results to {out_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", default="configs/tiny.json")
    args = parser.parse_args()
    try:
        main(args.checkpoint, args.tokenizer, args.config)
    except Exception as exc:
        print(f"evaluate failed: {exc}")
        raise

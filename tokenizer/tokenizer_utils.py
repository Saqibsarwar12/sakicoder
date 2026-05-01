from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from typing import List

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|system|>", "<|file|>", "<|error|>", "<|tool_call|>", "<|tool_result|>"]


def load_tokenizer(path: str) -> Tokenizer:
    t = Tokenizer.from_file(path)
    t.post_processor = TemplateProcessing(
        single="$A <|eos|>",
        pair="$A $B <|eos|>",
        special_tokens=[("<|eos|>", t.token_to_id("<|eos|>") if t.get_vocab() else 2)],
    )
    return t


def tokens_to_text(tokenizer: Tokenizer, ids: List[int]) -> str:
    text = tokenizer.decode(ids)
    # Byte-level fallback cleanup for display quality.
    return (
        text.replace("Ġ", " ")
        .replace("Ċ", "\n")
        .replace(" Â", "")
    )


def clean_decoded_text(text: str) -> str:
    return (
        text.replace("Ġ", " ")
        .replace("Ċ", "\n")
        .replace(" Â", "")
    )

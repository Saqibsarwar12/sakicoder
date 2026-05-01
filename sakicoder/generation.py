import torch
import torch.nn.functional as F
from typing import Optional
from tokenizer.tokenizer_utils import clean_decoded_text


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0):
    if top_k and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        vals, _ = torch.topk(logits, top_k)
        min_vals = vals[:, -1].unsqueeze(1)
        logits = torch.where(logits < min_vals, torch.full_like(logits, -float("inf")), logits)

    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_remove = cumulative_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove_mask = torch.zeros_like(logits, dtype=torch.bool)
        remove_mask.scatter_(1, sorted_indices, sorted_remove)
        logits = logits.masked_fill(remove_mask, -float("inf"))
    return logits


def generate(model, tokenizer, device, prompt: str, max_new_tokens: int = 64, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.95):
    model.eval()
    enc = tokenizer.encode(prompt)
    ids = enc.ids
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)
    for _ in range(max_new_tokens):
        logits, _ = model(input_ids)
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if next_id.item() == tokenizer.token_to_id("<|eos|>"):
            break
    return clean_decoded_text(tokenizer.decode(input_ids[0].tolist()))

import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .model import sample


TOKENIZER_PATH = Path("tokenizer/tokenizer.json")


@torch.inference_mode()
def stream(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
) -> None:
    """Greedily/temperature sample from `prompt` and print decoded deltas."""

    eot = tokenizer.token_to_id("<|endoftext|>")
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    ids = ([eot] if eot is not None else []) + ids
    tokens = torch.tensor([ids or [0]], dtype=torch.long, device=device)

    sys.stdout.write(prompt)
    sys.stdout.write("\033[2m|\033[0m")
    sys.stdout.flush()

    new_ids, printed = [], ""
    for _ in range(max_new_tokens):
        logits = model(tokens[:, -model.config.seq_len :])[:, -1]
        next_token = sample(logits, temperature, top_k if top_k > 0 else None)
        token_id = int(next_token.item())
        if eot is not None and token_id == eot:
            break
        tokens = torch.cat((tokens, next_token), dim=1)
        new_ids.append(token_id)
        decoded = tokenizer.decode(new_ids, skip_special_tokens=False)
        sys.stdout.write(decoded[len(printed) :])
        sys.stdout.flush()
        printed = decoded
    sys.stdout.write("\n")

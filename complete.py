"""Stream a completion from the trained model, token-by-token to stdout.

Usage:
    python complete.py "Once upon a time"
    python complete.py "Once upon a time" -n 200 -t 0.7 -k 40
    echo "Once upon a time" | python complete.py
"""

import argparse
import sys

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from llm import LLM, LLMConfig

TOKENIZER_PATH = "tokenizer/tokenizer.json"


def stream_complete(model, tk, prompt, max_new_tokens, temperature, top_k, device):
    eot_id = tk.token_to_id("<|endoftext|>")

    # Match training-time encoding: skip the TemplateProcessor auto-EOT suffix
    # and prepend a single EOT as a "start of document" marker.
    enc_ids = list(tk.encode(prompt, add_special_tokens=False).ids)
    if eot_id is not None:
        enc_ids = [eot_id] + enc_ids

    if not enc_ids:
        # Empty prompt: seed with just EOT so the model has at least one token.
        enc_ids = [eot_id] if eot_id is not None else [0]

    idx = torch.tensor([enc_ids], dtype=torch.long, device=device)

    # Echo the prompt first so the user sees it followed by the streaming
    # continuation. Use a dim separator so prompt vs. completion is visible.
    sys.stdout.write(prompt)
    sys.stdout.write("\033[2m|\033[0m")  # dim "|" between prompt and completion
    sys.stdout.flush()

    new_ids = []
    printed = ""

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.config.seq_len :]
            logits = model(idx_cond)[:, -1, :]

            if temperature == 0.0:
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, k)
                    logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)

            tok_id = int(next_tok.item())
            if eot_id is not None and tok_id == eot_id:
                break

            new_ids.append(tok_id)
            idx = torch.cat([idx, next_tok], dim=1)

            # Decode the entire continuation so far and print only the new
            # characters. This avoids emitting partial UTF-8 sequences when a
            # single BPE token doesn't decode to a clean character on its own.
            decoded = tk.decode(new_ids, skip_special_tokens=False)
            if len(decoded) > len(printed):
                sys.stdout.write(decoded[len(printed) :])
                sys.stdout.flush()
                printed = decoded

    sys.stdout.write("\n")
    sys.stdout.flush()


def main():
    p = argparse.ArgumentParser(
        description="Stream an autocompletion from the trained model."
    )
    p.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="prompt text. If omitted, read from stdin.",
    )
    p.add_argument("--run-name", default="v1", help="checkpoint subdir under checkpoints/")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="explicit .pth path (overrides --run-name)",
    )
    p.add_argument(
        "-n", "--max-new-tokens", type=int, default=128, help="tokens to generate"
    )
    p.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.8,
        help="0.0 = greedy argmax; >0 sample from softmax(logits/T)",
    )
    p.add_argument(
        "-k", "--top-k", type=int, default=40, help="restrict sampling to top-k logits"
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    prompt = args.prompt if args.prompt is not None else sys.stdin.read()

    tk = Tokenizer.from_file(TOKENIZER_PATH)

    model = LLM(LLMConfig())
    ckpt_path = args.checkpoint or f"checkpoints/{args.run_name}/final.pth"
    state = torch.load(ckpt_path, map_location=args.device)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(args.device)

    stream_complete(
        model,
        tk,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        device=args.device,
    )


if __name__ == "__main__":
    main()

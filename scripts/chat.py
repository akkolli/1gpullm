import argparse
import sys

import torch
from tokenizers import Tokenizer

from one_gpu_lm.checkpoint import load_model
from one_gpu_lm.data import TOKENIZER_PATH
from one_gpu_lm.sft import stream_chat


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    model = load_model(args.run_name, args.checkpoint, args.device)
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    stream_chat(
        model,
        tokenizer,
        prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.device,
        args.system_prompt,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--run-name", default="chat-sft")
    parser.add_argument("--checkpoint")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("-n", "--max-new-tokens", type=int, default=256)
    parser.add_argument("-t", "--temperature", type=float, default=0.7)
    parser.add_argument("-k", "--top-k", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()

import argparse

import torch

from one_gpu_lm.checkpoint import load_model
from one_gpu_lm.core_eval import TOKENIZER_PATH, TokenizerAdapter, evaluate_core, write_outputs


def main() -> None:
    args = parse_args()
    tokenizer = TokenizerAdapter(TOKENIZER_PATH)
    model = load_model(args.run_name, args.checkpoint, args.device)
    output = evaluate_core(
        model,
        tokenizer,
        args.device,
        model.config.seq_len,
        args.max_per_task,
    )
    print(f"\nCORE = {output['core_metric']:.4f}")
    write_outputs(args.run_name, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="v1.9")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-per-task", type=int, default=-1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()

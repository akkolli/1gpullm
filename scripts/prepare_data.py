import argparse
from dataclasses import replace

from one_gpu_lm.data import DataBuildConfig, build_pretraining_data


def main() -> None:
    args = parse_args()
    cfg = replace(
        DataBuildConfig(),
        dataset_name=args.dataset,
        max_train_tokens=args.max_train_tokens,
        max_tokenizer_docs=args.max_tokenizer_docs,
        workers=args.workers,
    )
    build_pretraining_data(cfg)


def parse_args() -> argparse.Namespace:
    cfg = DataBuildConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=cfg.dataset_name)
    parser.add_argument("--max-train-tokens", type=int, default=cfg.max_train_tokens)
    parser.add_argument("--max-tokenizer-docs", type=int, default=cfg.max_tokenizer_docs)
    parser.add_argument("--workers", type=int, default=cfg.workers)
    return parser.parse_args()


if __name__ == "__main__":
    main()

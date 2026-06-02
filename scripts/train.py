import argparse
from dataclasses import replace

from one_gpu_lm.data import ShardedTokenDataset
from one_gpu_lm.model import LLM, LLMConfig
from one_gpu_lm.training import TrainConfig, train


def main() -> None:
    args = parse_args()
    cfg = replace(
        TrainConfig(),
        run_name=args.run_name,
        epochs=args.epochs,
        train_steps=args.train_steps,
        val_steps=args.val_steps,
        batch_size=args.batch_size,
        precision=args.precision,
        gradient_checkpointing=args.gradient_checkpointing,
        compile_model=not args.no_compile,
        checkpoint_interval=args.checkpoint_interval,
        mfu_peak_tflops=args.mfu_peak_tflops,
    )
    train(
        LLM(LLMConfig(gradient_checkpointing=cfg.gradient_checkpointing)),
        ShardedTokenDataset("train"),
        ShardedTokenDataset("val"),
        cfg,
        resume=args.resume,
    )


def parse_args() -> argparse.Namespace:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=cfg.run_name)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--train-steps", type=int, default=cfg.train_steps)
    parser.add_argument("--val-steps", type=int, default=cfg.val_steps)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--precision", choices=("bf16", "fp8"), default=cfg.precision)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=cfg.checkpoint_interval)
    parser.add_argument(
        "--mfu-peak-tflops",
        type=float,
        default=cfg.mfu_peak_tflops,
        help="override the built-in RTX 5090 precision-specific peak TFLOPS",
    )
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

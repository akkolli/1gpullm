import argparse
from dataclasses import replace

from tokenizers import Tokenizer

from one_gpu_lm.checkpoint import load_model
from one_gpu_lm.data import TOKENIZER_PATH
from one_gpu_lm.sft import ChatSFTDataset, SFTConfig, train_sft


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    cfg = replace(
        SFTConfig(),
        run_name=args.run_name,
        epochs=args.epochs,
        train_steps=args.train_steps,
        val_steps=args.val_steps,
        batch_size=args.batch_size,
        peak_lr=args.peak_lr,
        lr_warmup=args.lr_warmup,
        precision=args.precision,
        gradient_checkpointing=args.gradient_checkpointing,
        compile_model=not args.no_compile,
        checkpoint_interval=args.checkpoint_interval,
        mfu_peak_tflops=args.mfu_peak_tflops,
    )
    model = load_model(
        run_name=args.base_run_name,
        checkpoint=args.checkpoint,
        device="cpu",
    )
    train_data = ChatSFTDataset(
        args.dataset,
        args.train_split,
        tokenizer,
        model.config.seq_len,
        dataset_config=args.dataset_config,
        streaming=not args.no_streaming,
        shuffle=True,
        shuffle_seed=args.shuffle_seed,
        shuffle_buffer=args.shuffle_buffer,
        max_examples=args.max_train_examples,
        system_prompt=args.system_prompt,
    )
    val_data = None
    if cfg.val_steps > 0 and args.val_split:
        val_data = ChatSFTDataset(
            args.dataset,
            args.val_split,
            tokenizer,
            model.config.seq_len,
            dataset_config=args.dataset_config,
            streaming=not args.no_streaming,
            shuffle=False,
            max_examples=args.max_val_examples,
            system_prompt=args.system_prompt,
        )
    train_sft(model, train_data, val_data, cfg, resume=args.resume)


def parse_args() -> argparse.Namespace:
    cfg = SFTConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run-name", default="v1.9")
    parser.add_argument("--checkpoint")
    parser.add_argument("--run-name", default=cfg.run_name)
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--dataset-config")
    parser.add_argument("--train-split", default="train_sft")
    parser.add_argument("--val-split", default="test_sft")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--train-steps", type=int, default=cfg.train_steps)
    parser.add_argument("--val-steps", type=int, default=cfg.val_steps)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--peak-lr", type=float, default=cfg.peak_lr)
    parser.add_argument("--lr-warmup", type=int, default=cfg.lr_warmup)
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
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-val-examples", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()

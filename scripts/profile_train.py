import argparse
import time

import torch

from one_gpu_lm.model import LLM, LLMConfig
from one_gpu_lm.training import apply_precision


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("profile_train requires CUDA")

    cfg = LLMConfig(
        n_dim=args.n_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model = LLM(cfg).cuda().train()
    apply_precision(model, args.precision)
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    x = torch.randint(args.vocab_size, (args.batch_size, args.seq_len), device="cuda")
    y = torch.randint(args.vocab_size, (args.batch_size, args.seq_len), device="cuda")

    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup):
        step(model, opt, x, y)
    torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(args.steps):
        step(model, opt, x, y)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    tokens = args.steps * args.batch_size * args.seq_len
    print(f"step_ms={elapsed / args.steps * 1000:.2f}")
    print(f"tokens_per_second={tokens / elapsed:.0f}")
    print(f"max_allocated_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}")
    print(f"max_reserved_gb={torch.cuda.max_memory_reserved() / 1e9:.2f}")


def step(model: torch.nn.Module, opt: torch.optim.Optimizer, x, y) -> None:
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(x, targets=y)
    loss.backward()
    opt.step()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--precision", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--n-dim", type=int, default=768)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=32384)
    return parser.parse_args()


if __name__ == "__main__":
    main()

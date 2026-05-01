import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.optim import AdamW, Muon
from torch.optim.lr_scheduler import LambdaLR
from torchao.float8 import convert_to_float8_training

from .checkpoint import (
    load_training_state,
    save_model,
    save_training_state,
    unwrap_model,
)
from .model import LLM, param_breakdown


@dataclass(frozen=True)
class TrainConfig:
    """Single-run training configuration.

    The defaults target a short one-GPU run. `precision` controls module
    conversion before `torch.compile`; optimizer state remains in the optimizer
    implementation's chosen dtype.
    """

    run_name: str = "v1.9"
    epochs: int = 40
    train_steps: int = 3000
    val_steps: int = 50
    batch_size: int = 32
    val_interval: int = 1
    peak_lr: float = 5e-4
    lr_warmup: int = 6000
    min_lr_ratio: float = 0.1
    z_loss_coef: float = 1e-4
    gradient_checkpointing: bool = False
    precision: str = "fp8"
    compile_model: bool = True
    checkpoint_interval: int = 1

    @property
    def out_dir(self) -> Path:
        return Path("checkpoints") / self.run_name

    @property
    def total_steps(self) -> int:
        return self.epochs * self.train_steps


def train(
    model: LLM,
    train_data,
    val_data,
    cfg: TrainConfig,
    resume: bool = False,
) -> tuple[torch.nn.Module, dict]:
    """Train `model` on sharded next-token data and write final artifacts."""

    if not torch.cuda.is_available():
        raise RuntimeError("training requires a CUDA device")
    torch.set_float32_matmul_precision("medium")
    model.config.gradient_checkpointing = cfg.gradient_checkpointing
    seq_len = model.config.seq_len
    model = model.cuda()
    total_params = param_breakdown(model)
    # Convert precision before compile so Dynamo traces the final module graph.
    apply_precision(model, cfg.precision)
    if cfg.compile_model:
        model = torch.compile(model, mode="reduce-overhead")

    optimizers = build_optimizers(model, cfg)
    schedulers = [
        LambdaLR(
            opt,
            cosine_with_warmup(cfg.total_steps, cfg.lr_warmup, cfg.min_lr_ratio),
        )
        for opt in optimizers
    ]
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    state_path = cfg.out_dir / "train_state.pt"
    start_epoch, history = 0, {"train_loss": [], "val_loss": []}
    if resume and state_path.exists():
        start_epoch, history = load_training_state(
            state_path,
            model,
            optimizers,
            schedulers,
            device="cuda",
        )

    tokens_per_step = cfg.batch_size * seq_len
    tokens_seen = tokens_per_step * cfg.total_steps
    print(
        f"[train] {tokens_seen / 1e6:.1f}M tokens, "
        f"{tokens_seen / total_params:.2f} tok/param"
    )

    started = time.time()
    last_epoch_time = 0.0
    for epoch in range(start_epoch, cfg.epochs):
        train_loss, last_epoch_time = train_epoch(
            model,
            train_data,
            optimizers,
            schedulers,
            cfg,
            seq_len,
        )
        val_loss = float("nan")
        if (epoch + 1) % cfg.val_interval == 0 or epoch == cfg.epochs - 1:
            val_loss = validate(model, val_data, cfg, seq_len)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        tok_s = tokens_per_step / (last_epoch_time / cfg.train_steps)
        print(
            f"[{epoch + 1}/{cfg.epochs}] train={train_loss:.4f} "
            f"val={val_loss:.4f} {tok_s:.0f} tok/s"
        )
        if cfg.checkpoint_interval and (epoch + 1) % cfg.checkpoint_interval == 0:
            save_training_state(
                state_path,
                model,
                optimizers,
                schedulers,
                cfg,
                epoch + 1,
                history,
            )

    save_model(cfg.out_dir / "final.pth", model)
    metrics = {
        **history,
        "config": asdict(cfg),
        "total_training_time": time.time() - started,
        "last_epoch_time": last_epoch_time,
        "tokens_covered": tokens_seen,
        "total_params": total_params,
    }
    write_json(cfg.out_dir / "metrics.json", metrics)
    save_training_state(
        state_path,
        model,
        optimizers,
        schedulers,
        cfg,
        cfg.epochs,
        history,
        metrics,
    )
    return model, metrics


def train_epoch(
    model: torch.nn.Module,
    train_data,
    optimizers: list,
    schedulers: list[LambdaLR],
    cfg: TrainConfig,
    seq_len: int,
) -> tuple[float, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    loss_sum = torch.zeros((), device="cuda")
    for _ in range(cfg.train_steps):
        x, y = train_data.get_batch(cfg.batch_size, seq_len, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, targets=y, z_loss_coef=cfg.z_loss_coef)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()
            opt.zero_grad()
        for sched in schedulers:
            sched.step()
        loss_sum += loss.detach()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return (loss_sum / cfg.train_steps).item(), elapsed


@torch.inference_mode()
def validate(model: torch.nn.Module, val_data, cfg: TrainConfig, seq_len: int) -> float:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    for _ in range(cfg.val_steps):
        x, y = val_data.get_batch(cfg.batch_size, seq_len, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss_sum += model(x, targets=y).item()
    model.train(was_training)
    return loss_sum / cfg.val_steps


def build_optimizers(model: torch.nn.Module, cfg: TrainConfig) -> list:
    """Use Muon for hidden matrices and AdamW for embeddings/head/norms."""

    muon_params, adam_params = split_parameters(unwrap_model(model))
    return [
        Muon(
            muon_params,
            lr=cfg.peak_lr,
            weight_decay=0.1,
            momentum=0.95,
            adjust_lr_fn="match_rms_adamw",
        ),
        AdamW(
            adam_params,
            lr=cfg.peak_lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True,
        ),
    ]


def split_parameters(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Partition trainable parameters by optimizer.

    Muon is intended for hidden 2D weight matrices. AdamW keeps the tied
    embedding/head weights, norms, and any non-matrix parameters on the more
    standard update rule.
    """

    muon_params, adam_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        target = muon_params if is_hidden_matrix(name, param) else adam_params
        target.append(param)
    return muon_params, adam_params


def is_hidden_matrix(name: str, param: torch.nn.Parameter) -> bool:
    excluded = ("embeddings", "lm_head", "pos", "norm")
    return param.ndim == 2 and not any(token in name for token in excluded)


def apply_precision(model: torch.nn.Module, precision: str) -> None:
    if precision == "bf16":
        return
    if precision != "fp8":
        raise ValueError("precision must be 'fp8' or 'bf16'")
    convert_to_float8_training(model, module_filter_fn=fp8_module)


def fp8_module(_module: torch.nn.Module, fqn: str) -> bool:
    # Keep tied token weights and positional state out of low-precision rewrites.
    return not any(token in fqn for token in ("lm_head", "embed", "pos"))


def cosine_with_warmup(
    total_steps: int,
    warmup_steps: int,
    min_ratio: float,
) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return schedule


def write_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

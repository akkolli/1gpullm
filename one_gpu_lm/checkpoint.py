import random
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch

from .model import LLM, LLMConfig


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the original module when `torch.compile` wrapped it."""

    return getattr(model, "_orig_mod", model)


def strip_compile_prefix(state: dict) -> dict:
    """Normalize state dicts saved from compiled or uncompiled modules."""

    return {key.removeprefix("_orig_mod."): value for key, value in state.items()}


def save_model(path: Path, model: torch.nn.Module) -> None:
    """Save only model weights for eval/generation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(unwrap_model(model).state_dict(), path)


def load_model(
    run_name: str = "v1.9",
    checkpoint: str | Path | None = None,
    device: str = "cuda",
    config: LLMConfig | None = None,
) -> LLM:
    """Load an `LLM` from a run directory or explicit checkpoint path."""

    model = LLM(config or LLMConfig())
    path = Path(checkpoint) if checkpoint else Path("checkpoints") / run_name / "final.pth"
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(strip_compile_prefix(state))
    return model.to(device).eval()


def save_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizers: list,
    schedulers: list,
    cfg,
    epoch: int,
    history: dict,
    metrics: dict | None = None,
) -> None:
    """Save enough state to resume training exactly at an epoch boundary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": unwrap_model(model).state_dict(),
            "optimizers": [opt.state_dict() for opt in optimizers],
            "schedulers": [sched.state_dict() for sched in schedulers],
            "config": asdict(cfg) if is_dataclass(cfg) else dict(cfg),
            "history": history,
            "metrics": metrics or {},
            "rng": capture_rng_state(),
        },
        path,
    )


def load_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizers: list,
    schedulers: list,
    device: str,
) -> tuple[int, dict]:
    """Restore model, optimizer, scheduler, and RNG state."""

    state = torch.load(path, map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(strip_compile_prefix(state["model"]))
    for opt, opt_state in zip(optimizers, state["optimizers"], strict=True):
        opt.load_state_dict(opt_state)
    for sched, sched_state in zip(schedulers, state["schedulers"], strict=True):
        sched.load_state_dict(sched_state)
    restore_rng_state(state.get("rng", {}))
    return int(state["epoch"]), state.get("history", {"train_loss": [], "val_loss": []})


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])

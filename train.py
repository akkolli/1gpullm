import json
import math
import os
import pickle
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.optim import AdamW, Muon
from torch.optim.lr_scheduler import LambdaLR
from torchao.float8 import convert_to_float8_training
from tqdm import tqdm

from llm import LLM, LLMConfig, param_breakdown
from prepare_pretraining_data import ShardedTokenDataset

train_dataset = ShardedTokenDataset("train")
val_dataset = ShardedTokenDataset("val")


@dataclass
class TrainConfig:
    RUN_NAME: str = "v1.8:mega"
    epochs: int = 40
    train_steps: int = 2200
    val_steps: int = 100
    batch_size: int = 176
    val_interval: int = 1  # Epoch between val intervals
    peak_lr: float = 5e-4
    lr_warmup: int = 4000
    min_lr_ratio: float = 0.1
    # z-loss penalizes the softmax partition function magnitude. Stabilizes
    # bf16 logits and lets us push LR up. PaLM used 1e-4.
    z_loss_coeff: float = 1e-4


def train(model, train_dataloader, val_dataloader, train_config):
    torch.set_float32_matmul_precision("medium")

    model = model.to("cuda")
    total_params = param_breakdown(model)

    def fp8_filter(module, fqn: str) -> bool:
        return "lm_head" not in fqn and "embed" not in fqn and "pos" not in fqn

    convert_to_float8_training(model, module_filter_fn=fp8_filter)

    model = torch.compile(model, mode="reduce-overhead")

    # Muon goes on 2D hidden weights only. Embeddings (lookup, sparse-ish),
    # the tied lm_head, positional embeddings, and norms (1D) need AdamW.
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_2d_hidden = (
            p.ndim == 2
            and "embeddings" not in name
            and "lm_head" not in name
            and "pos" not in name
            and "norm" not in name
        )
        (muon_params if is_2d_hidden else adam_params).append(p)
    print(
        f"[opt] Muon: {sum(p.numel() for p in muon_params):,} params "
        f"across {len(muon_params)} tensors | "
        f"AdamW: {sum(p.numel() for p in adam_params):,} params "
        f"across {len(adam_params)} tensors"
    )

    # match_rms_adamw scales Muon's update RMS to match AdamW, so we can reuse
    # the AdamW LR (peak_lr) for both — no separate Muon LR to tune.
    opt_muon = Muon(
        muon_params,
        lr=train_config.peak_lr,
        weight_decay=0.1,
        momentum=0.95,
        adjust_lr_fn="match_rms_adamw",
    )
    opt_adam = AdamW(
        adam_params,
        lr=train_config.peak_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    optimizers = [opt_muon, opt_adam]
    losses = []
    val_losses = []
    check_point_path = f"./checkpoints/{train_config.RUN_NAME}"
    os.makedirs(check_point_path, exist_ok=True)
    total_steps = train_config.train_steps * train_config.epochs

    def lr_lamda(step):
        if step < train_config.lr_warmup:
            return (step + 1) / train_config.lr_warmup
        progress = (step - train_config.lr_warmup) / max(
            1, total_steps - train_config.lr_warmup
        )
        return train_config.min_lr_ratio + (1 - train_config.min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )

    schedulers = [LambdaLR(o, lr_lamda) for o in optimizers]
    tokens_per_step = train_config.batch_size * model.config.seq_len
    tokens_covered = tokens_per_step * train_config.train_steps * train_config.epochs
    total_tokens = train_dataloader.total_tokens
    print(
        f"Covering {(tokens_covered / total_tokens) * 100:.4f}% of the dataset "
        f"({tokens_covered / 1e6:.2f}M tokens)"
    )

    print(f"Tokens/param = {tokens_covered / total_params:.4f}")

    batch_load_total = 0.0
    model_total = 0.0
    last_epoch_time = None

    training_start_time = time.time()
    for epoch in range(train_config.epochs):
        is_last_epoch = epoch == train_config.epochs - 1
        if is_last_epoch:
            torch.cuda.synchronize()
            epoch_start = time.perf_counter()

        loss_accum = torch.zeros((), device="cuda")
        z_accum = torch.zeros((), device="cuda")
        for step in tqdm(range(train_config.train_steps)):
            if is_last_epoch:
                torch.cuda.synchronize()
                t_batch_start = time.perf_counter()

            x, y = train_dataloader.get_batch(
                train_config.batch_size, model.config.seq_len, device="cuda"
            )

            if is_last_epoch:
                torch.cuda.synchronize()
                batch_load_total += time.perf_counter() - t_batch_start
                t_model_start = time.perf_counter()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = model(x)
                flat = preds.view(-1, preds.size(-1))
                ce = F.cross_entropy(flat, y.view(-1))
                # All bf16. fp32 cast here would double the backward grad on
                # `flat` to 4.8GB at this batch size. logsumexp's max-shift
                # keeps bf16 numerically fine for the aux-loss purpose.
                log_z = torch.logsumexp(flat, dim=-1)
                z = (log_z * log_z).mean()
                loss = ce + train_config.z_loss_coeff * z

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for o in optimizers:
                o.step()
            for s in schedulers:
                s.step()
            for o in optimizers:
                o.zero_grad()

            if is_last_epoch:
                torch.cuda.synchronize()
                model_total += time.perf_counter() - t_model_start

            loss_accum += ce.detach()
            z_accum += z.detach()

        if is_last_epoch:
            torch.cuda.synchronize()
            last_epoch_time = time.perf_counter() - epoch_start

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for _ in range(train_config.val_steps):
                x, y = val_dataloader.get_batch(
                    train_config.batch_size, model.config.seq_len, device="cuda"
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    preds = model(x)
                    loss = F.cross_entropy(preds.view(-1, preds.size(-1)), y.view(-1))
                val_loss += loss.item()
        model.train()

        avg_train_l = loss_accum.item() / train_config.train_steps
        avg_z = z_accum.item() / train_config.train_steps
        avg_val_l = val_loss / train_config.val_steps
        train_ppl = math.exp(avg_train_l)
        val_ppl = math.exp(avg_val_l)

        print(
            f"Epoch {epoch}: Train CE {avg_train_l:.4f} PPL {train_ppl:.2f} Acc {1 - (train_ppl / model.config.vocab_size):.5f} |"
            f" Val CE {avg_val_l:.4f} PPL {val_ppl:.2f} Acc {1 - (val_ppl / model.config.vocab_size):.2f} |"
            f" z={avg_z:.3f}"
        )
        losses.append(avg_train_l)
        val_losses.append(avg_val_l)

    total_training_time = time.time() - training_start_time

    n = train_config.train_steps
    avg_batch = batch_load_total / n
    avg_model = model_total / n
    avg_step = last_epoch_time / n
    print(
        f"Last epoch ({last_epoch_time:.2f}s): "
        f"batch_load={avg_batch * 1000:.2f}ms ({100 * batch_load_total / last_epoch_time:.1f}%), "
        f"model={avg_model * 1000:.2f}ms ({100 * model_total / last_epoch_time:.1f}%), "
        f"step={avg_step * 1000:.2f}ms, {tokens_per_step / avg_step:.0f} tok/s"
    )

    raw_model = getattr(model, "_orig_mod", model)
    torch.save(raw_model.state_dict(), check_point_path + "/final.pth")
    with open(check_point_path + "/config.json", "w") as f:
        json.dump(asdict(train_config), f, indent=2)

    measurements = {
        "train_loss": losses,
        "val_loss": val_losses,
        "total_training_time": total_training_time,
        "last_epoch_time": last_epoch_time,
        "avg_batch_load_time": avg_batch,
        "avg_model_time": avg_model,
        "avg_step_time": avg_step,
        "tokens_covered": tokens_covered,
        "total_params": total_params,
    }
    with open(check_point_path + "/loss_curves.pkl", "wb") as f:
        pickle.dump(measurements, f)

    return model, measurements


def plot_graphs(loss_dict, train_config):
    import matplotlib.pyplot as plt

    check_point_path = f"./checkpoints/{train_config.RUN_NAME}"

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(loss_dict["train_loss"]) + 1)
    ax.plot(epochs, loss_dict["train_loss"], color="blue", marker="o", label="train")
    ax.plot(epochs, loss_dict["val_loss"], color="red", marker="o", label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(check_point_path + "/loss_curve.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    model_config = LLMConfig()
    train_config = TrainConfig()
    model = LLM(model_config)
    _, measurements = train(model, train_dataset, val_dataset, train_config)
    plot_graphs(measurements, train_config)

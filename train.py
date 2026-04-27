import json
import math
import os
import pickle
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from llm import LLM, LLMConfig, param_breakdown
from prepare_pretraining_data import ShardedTokenDataset

train_dataset = ShardedTokenDataset("train")
val_dataset = ShardedTokenDataset("val")


@dataclass
class TrainConfig:
    RUN_NAME: str = "v1.1"
    epochs = 10
    train_steps = 8000
    val_steps = 100
    batch_size = 512
    val_interval = 1  # Epoch between val intervals


def train(model, train_dataloader, val_dataloader, train_config):
    model.to("cuda")
    optimizer = AdamW(model.parameters(), lr=1e-4)
    losses = []
    val_losses = []
    check_point_path = f"./checkpoints/{train_config.RUN_NAME}"
    os.makedirs(check_point_path, exist_ok=True)

    tokens_per_step = train_config.batch_size * model.config.seq_len
    tokens_covered = tokens_per_step * train_config.train_steps * train_config.epochs
    total_tokens = train_dataloader.total_tokens
    print(
        f"Covering {(tokens_covered / total_tokens) * 100:.4f}% of the dataset "
        f"({tokens_covered / 1e6:.2f}M tokens)"
    )
    total_params = param_breakdown(model)
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

        train_loss = 0
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

            preds = model(x)
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            x.to("cpu")
            y.to("cpu")

            if is_last_epoch:
                torch.cuda.synchronize()
                model_total += time.perf_counter() - t_model_start

            train_loss += loss.item()

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
                preds = model(x)
                loss = F.cross_entropy(preds.view(-1, preds.size(-1)), y.view(-1))
                val_loss += loss.item()
        model.train()

        avg_train_l = train_loss / train_config.train_steps
        avg_val_l = val_loss / train_config.val_steps
        train_ppl = math.exp(avg_train_l)
        val_ppl = math.exp(avg_val_l)

        print(
            f"Epoch {epoch}: Train CE {avg_train_l:.4f} PPL {train_ppl:.2f} Acc {1 - (train_ppl / model.config.vocab_size):.2f} |"
            f" Val CE {avg_val_l:.4f} PPL {val_ppl:.2f} Acc {1 - (val_ppl / model.config.vocab_size):.2f}"
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

    torch.save(model.state_dict(), check_point_path + "/final.pth")
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

    plt.plot(loss_dict["train_loss"], color="blue")
    plt.plot(loss_dict["val_loss"], color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Perplexity")
    plt.savefig(check_point_path + "/loss_curve.png")


if __name__ == "__main__":
    model_config = LLMConfig()
    train_config = TrainConfig()
    model = LLM(model_config)
    _, measurements = train(model, train_dataset, val_dataset, train_config)
    plot_graphs(measurements, train_config)

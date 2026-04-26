import glob
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from llm import LLM, LLMConfig


class ShardedTokenDataset:
    """Memory-mapped reader for sharded uint32 token files.

    Layout on disk: dataset/fineweb/fineweb_{split}_{shard:04d}.bin
    Each file is a flat sequence of uint32 token IDs with <|endoftext|>
    between original documents.

    Sampling strategy: pick a shard weighted by its length, then a uniform
    random window inside it. This gives each token in the dataset equal
    probability of being the start of a sample.
    """

    def __init__(self, split: str, data_dir: str = "dataset/fineweb", dtype=np.uint32):
        pattern = os.path.join(data_dir, f"fineweb_{split}_*.bin")
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no shards matched {pattern}")

        self.dtype = dtype
        self.itemsize = np.dtype(dtype).itemsize  # bytes per token
        self.paths = paths
        # mmap each shard; mode="r" means read-only, no copy on write
        self.shards = [np.memmap(p, dtype=dtype, mode="r") for p in paths]
        self.lengths = np.array([len(s) for s in self.shards], dtype=np.int64)
        self.total_tokens = int(self.lengths.sum())
        # Sampling weights proportional to shard length
        self.weights = self.lengths / self.total_tokens

        print(
            f"[loader/{split}] {len(self.shards)} shards, "
            f"{self.total_tokens:,} tokens "
            f"({sum(os.path.getsize(p) for p in paths) / 1e9:.2f}GB)"
        )

    def __len__(self):
        return self.total_tokens

    def get_batch(
        self,
        batch_size: int,
        context_len: int,
        device: str = "cpu",
        rng: np.random.Generator = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of (input, target) pairs for next-token prediction.

        Returns:
          x: (batch_size, context_len) int64 token IDs
          y: (batch_size, context_len) int64 token IDs, shifted by 1
        """
        if rng is None:
            rng = np.random.default_rng()

        # Pick which shard each sample comes from, weighted by length
        shard_ids = rng.choice(len(self.shards), size=batch_size, p=self.weights)

        x = np.empty((batch_size, context_len), dtype=np.int64)
        y = np.empty((batch_size, context_len), dtype=np.int64)

        for b, sid in enumerate(shard_ids):
            shard = self.shards[sid]
            max_start = len(shard) - context_len - 1
            if max_start <= 0:
                # Tiny shard (shouldn't happen with 250M-token shards). Fall
                # back to a different shard.
                sid = int(rng.integers(0, len(self.shards)))
                shard = self.shards[sid]
                max_start = len(shard) - context_len - 1
            i = int(rng.integers(0, max_start))
            # astype(int64) materializes the slice into RAM — required because
            # PyTorch can't make tensors from uint32, and mmap'd arrays aren't
            # writable. The slice is small (just context_len tokens).
            x[b] = shard[i : i + context_len].astype(np.int64)
            y[b] = shard[i + 1 : i + 1 + context_len].astype(np.int64)

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        if device != "cpu":
            x_t = x_t.to(device, non_blocking=True)
            y_t = y_t.to(device, non_blocking=True)
        return x_t, y_t


train_dataset = ShardedTokenDataset("train")
val_dataset = ShardedTokenDataset("val")


@dataclass
class TrainConfig:
    epochs = 10
    batch_size = 512
    val_interval = 1  # Epoch between val intervals


def train(model, train_dataloader, val_dataloader, train_config):
    optimizer = AdamW(model.parameters(), lr=1e-4)
    losses = []

    for epoch in range(train_config.epochs):
        train_loss = 0
        for batch in train_dataloader:
            x, y = batch.to("cuda")
            preds = model(x)
            loss = F.kl_div(preds, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss

        model.eval()
        val_loss = 0
        for batch in val_dataloader:
            x, y = batch.to("cuda")
            preds = model(x)
            loss = F.kl_div(preds, y)
            val_loss += loss

        print(f"Epoch {epoch}: Train - {train_loss} Test - {val_loss}")

        losses.append(train_loss)

    return model


if __name__ == "__main__":
    config = LLMConfig()
    model = LLM(config)
    train(model, train_dataset, val_dataset, TrainConfig)

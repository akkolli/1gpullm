import json
import mmap
import multiprocessing as mp
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np
import torch

os.environ.setdefault("RAYON_NUM_THREADS", "16")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from .model import LLMConfig


TOKENIZER_PATH = Path("tokenizer/tokenizer.json")
TOKENIZER_CONFIG_PATH = Path("tokenizer/tokenizer_config.json")
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|fim_hole|>",
    "<|fim_end|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|tool_call|>",
    "<|tool_result|>",
    "<|think_begin|>",
    "<|think_end|>",
]
PRETOKENIZER_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|"""
    r"""[^\r\n\p{L}\p{N}]?+\p{L}+|"""
    r"""\p{N}{1,3}|"""
    r""" ?[^\s\p{L}\p{N}]++[\r\n]*|"""
    r"""\s*[\r\n]|\s+(?!\S)|\s+"""
)

_WORKER_TOKENIZER = None
_WORKER_EOT = None


@dataclass(frozen=True)
class DataBuildConfig:
    """Dataset/tokenizer build configuration.

    `max_train_tokens` is a hard cap for emitted train tokens. `val_every`
    sends every Nth document to validation before sharding.
    """

    dataset_name: str = "OptimalScale/ClimbMix"
    split: str = "train"
    max_tokenizer_docs: int = 100_000
    max_train_tokens: int = 4_000_000_000
    shard_tokens: int = 250_000_000
    val_every: int = 1000
    shuffle_seed: int = 42
    shuffle_buffer: int = 50_000
    worker_batch: int = 64
    status_every: int = 50_000
    workers: int = min(8, max(2, (os.cpu_count() or 4) - 2))

    @property
    def slug(self) -> str:
        return self.dataset_name.split("/")[-1].lower()

    @property
    def out_dir(self) -> Path:
        return Path("dataset") / self.slug


@dataclass
class PrefetchSlot:
    """Reusable host buffers plus the CUDA event for their device copy."""

    x_host: torch.Tensor
    y_host: torch.Tensor
    x_np: np.ndarray
    y_np: np.ndarray
    event: torch.cuda.Event | None = None

    @classmethod
    def allocate(cls, batch_size: int, seq_len: int) -> "PrefetchSlot":
        x = pinned_empty(batch_size, seq_len)
        y = pinned_empty(batch_size, seq_len)
        return cls(x, y, x.numpy(), y.numpy())


class ShardedTokenDataset:
    """Memory-mapped token shards with optional CUDA prefetching.

    Shards are flat `uint32` token streams on disk. `get_batch` returns adjacent
    next-token pairs shaped `[batch, seq]`, using pinned host buffers and a
    background CUDA stream when the requested device is not CPU.
    """

    def __init__(
        self,
        split: str,
        data_dir: str | Path | None = None,
        dataset_slug: str = DataBuildConfig().slug,
        dtype=np.uint32,
    ):
        root = Path(data_dir) if data_dir is not None else Path("dataset") / dataset_slug
        self.paths = sorted(root.glob(f"{dataset_slug}_{split}_*.bin"))
        if not self.paths:
            raise FileNotFoundError(f"no shards found in {root}")

        self.shards, self._handles = [], []
        for path in self.paths:
            file = path.open("rb")
            view = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            if hasattr(view, "madvise"):
                view.madvise(mmap.MADV_SEQUENTIAL)
            self.shards.append(np.frombuffer(view, dtype=dtype))
            self._handles.append((file, view))

        self.lengths = np.array([len(shard) for shard in self.shards], dtype=np.int64)
        self.total_tokens = int(self.lengths.sum())
        self._order = np.arange(len(self.shards))
        self._shard_pos = 0
        self._offset = 0
        self._reset(np.random.default_rng())

        self._pinned = {}
        self._pinned_idx = {}
        self._prefetch_thread = None
        self._prefetch_stop = threading.Event()
        self._prefetch_queue = None
        self._prefetch_shape = None
        self._prefetch_device = None
        print(f"[loader/{split}] {len(self.shards)} shards, {self.total_tokens:,} tokens")

    def __len__(self) -> int:
        return self.total_tokens

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        self._stop_prefetch()
        # Drop NumPy views before closing the mmaps they reference.
        self.shards = []
        self.lengths = []
        self.total_tokens = 0
        for file, view in getattr(self, "_handles", []):
            view.close()
            file.close()
        self._handles = []

    def get_batch(
        self,
        batch_size: int,
        seq_len: int,
        device: str = "cuda",
        rng=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if device != "cpu":
            return self._get_prefetched(batch_size, seq_len, device)
        if rng is None:
            rng = np.random.default_rng()
        x_host, y_host, x_np, y_np = self._pinned_pair(batch_size, seq_len)
        self._fill(x_np, y_np, batch_size, seq_len, rng)
        return x_host, y_host

    def _reset(self, rng):
        self._order = rng.permutation(len(self.shards))
        self._shard_pos = 0
        # Randomize the first offset so repeated loaders do not see the same
        # initial windows, while staying in the first half to leave room.
        shard = self.shards[int(self._order[0])]
        self._offset = int(rng.integers(0, max(1, len(shard) // 2)))

    def _advance_shard(self, rng):
        self._shard_pos += 1
        if self._shard_pos == len(self._order):
            self._reset(rng)
        else:
            self._offset = 0

    def _fill(self, x_np, y_np, batch_size, seq_len, rng):
        n = batch_size * seq_len
        for _ in range(2 * len(self.shards) + 2):
            shard = self.shards[int(self._order[self._shard_pos])]
            if self._offset + n + 1 <= len(shard):
                start = self._offset
                self._offset += n
                x = shard[start : start + n].reshape(batch_size, seq_len)
                y = shard[start + 1 : start + 1 + n].reshape(batch_size, seq_len)
                np.copyto(x_np, x, casting="unsafe")
                np.copyto(y_np, y, casting="unsafe")
                return
            self._advance_shard(rng)
        raise RuntimeError(f"no shard large enough for {n + 1:,} tokens")

    def _pinned_pair(self, batch_size, seq_len):
        key = (batch_size, seq_len)
        if key not in self._pinned:
            self._pinned[key] = []
            self._pinned_idx[key] = 0
            for _ in range(2):
                x = pinned_empty(batch_size, seq_len)
                y = pinned_empty(batch_size, seq_len)
                self._pinned[key].append((x, y, x.numpy(), y.numpy()))
        idx = self._pinned_idx[key]
        self._pinned_idx[key] = 1 - idx
        return self._pinned[key][idx]

    def _get_prefetched(self, batch_size, seq_len, device):
        shape = (batch_size, seq_len)
        if self._prefetch_thread is None:
            self._start_prefetch(batch_size, seq_len, device)
        elif self._prefetch_shape != shape or self._prefetch_device != device:
            self._stop_prefetch()
            self._start_prefetch(batch_size, seq_len, device)

        x, y, event = self._prefetch_queue.get()
        if isinstance(x, str) and x == "__error__":
            raise RuntimeError("prefetch thread crashed") from y
        event.wait()
        return x, y

    def _start_prefetch(self, batch_size, seq_len, device, slots=4, queue_size=2):
        self._prefetch_stop.clear()
        self._prefetch_queue = Queue(maxsize=queue_size)
        self._prefetch_shape = (batch_size, seq_len)
        self._prefetch_device = device
        # More slots than queued batches lets the copy stream overlap with
        # training without overwriting buffers still in flight.
        buffers = [PrefetchSlot.allocate(batch_size, seq_len) for _ in range(slots)]
        self._prefetch_thread = threading.Thread(
            target=self._prefetch,
            args=(buffers, batch_size, seq_len, device),
            daemon=True,
        )
        self._prefetch_thread.start()

    def _stop_prefetch(self):
        if getattr(self, "_prefetch_thread", None) is None:
            return
        self._prefetch_stop.set()
        try:
            while True:
                self._prefetch_queue.get_nowait()
        except Empty:
            pass
        self._prefetch_thread.join(timeout=2.0)
        self._prefetch_thread = None
        self._prefetch_queue = None
        self._prefetch_shape = None
        self._prefetch_device = None

    def _prefetch(self, slots, batch_size, seq_len, device):
        rng = np.random.default_rng()
        stream = torch.cuda.Stream()
        try:
            for slot in cycle(slots):
                if self._prefetch_stop.is_set():
                    return
                if slot.event is not None:
                    slot.event.synchronize()
                self._fill(slot.x_np, slot.y_np, batch_size, seq_len, rng)
                with torch.cuda.stream(stream):
                    x_dev = slot.x_host.to(device, non_blocking=True)
                    y_dev = slot.y_host.to(device, non_blocking=True)
                    slot.event = torch.cuda.Event()
                    slot.event.record(stream)
                self._put_prefetch(x_dev, y_dev, slot.event)
        except BaseException as exc:
            self._put_prefetch("__error__", exc, None)
            raise

    def _put_prefetch(self, x, y, event):
        while not self._prefetch_stop.is_set():
            try:
                self._prefetch_queue.put((x, y, event), timeout=0.5)
                return
            except Full:
                pass


class SplitShardWriter:
    """Buffered writer for one train/val split."""

    def __init__(self, split, out_dir, slug, shard_tokens):
        self.split = split
        self.out_dir = Path(out_dir)
        self.slug = slug
        self.shard_tokens = shard_tokens
        self.buffer = np.empty(shard_tokens, dtype=np.uint32)
        self.used = 0
        self.index = 0
        self.total_tokens = 0

    @property
    def path(self) -> Path:
        return self.out_dir / f"{self.slug}_{self.split}_{self.index:04d}.bin"

    def append(self, tokens: np.ndarray) -> None:
        pos = 0
        while pos < len(tokens):
            room = self.shard_tokens - self.used
            n = min(room, len(tokens) - pos)
            self.buffer[self.used : self.used + n] = tokens[pos : pos + n]
            self.used += n
            self.total_tokens += n
            pos += n
            if self.used == self.shard_tokens:
                self.flush()

    def flush(self) -> None:
        if self.used == 0:
            return
        self.buffer[: self.used].tofile(self.path)
        print(f"[shard/{self.split}] {self.path} ({self.used:,} tokens)")
        self.index += 1
        self.used = 0


def build_pretraining_data(cfg: DataBuildConfig = DataBuildConfig()) -> None:
    """Train/reuse the tokenizer, then encode and shard the streaming dataset."""

    stream = text_stream(cfg)
    tokenizer = load_or_train_tokenizer(cfg, stream)
    encode_stream(cfg, tokenizer)


def load_or_train_tokenizer(cfg: DataBuildConfig, stream) -> Tokenizer:
    """Load a matching tokenizer or train the repo's byte-level BPE."""

    if tokenizer_matches_vocab():
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        print(f"[tokenizer] loaded {TOKENIZER_PATH}")
        return tokenizer

    tokenizer = Tokenizer(BPE(unk_token=None, byte_fallback=False))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(PRETOKENIZER_PATTERN, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=LLMConfig.vocab_size,
        min_frequency=10,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    started = time.time()
    tokenizer.train_from_iterator(
        texts(stream, cfg.max_tokenizer_docs),
        trainer=trainer,
        length=cfg.max_tokenizer_docs,
    )
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A <|endoftext|>",
        pair="$A <|endoftext|> $B <|endoftext|>",
        special_tokens=[("<|endoftext|>", eot_id)],
    )
    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_PATH))
    write_json(TOKENIZER_CONFIG_PATH, {"vocab_size": LLMConfig.vocab_size})
    print(f"[tokenizer] trained in {(time.time() - started) / 60:.1f} min")
    return tokenizer


def encode_stream(cfg: DataBuildConfig, tokenizer: Tokenizer) -> None:
    """Encode documents in worker processes and write train/val shards."""

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    remove_existing_shards(cfg)
    writers = {
        "train": SplitShardWriter("train", cfg.out_dir, cfg.slug, cfg.shard_tokens),
        "val": SplitShardWriter("val", cfg.out_dir, cfg.slug, cfg.shard_tokens),
    }
    print(f"[encode] {cfg.workers} workers, cap={cfg.max_train_tokens:,} train tokens")
    started = time.time()
    processed = 0

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=cfg.workers,
        initializer=init_worker,
        initargs=(str(TOKENIZER_PATH),),
        maxtasksperchild=50,
    ) as pool:
        batches = document_batches(text_stream(cfg), cfg.worker_batch)
        for encoded_batch in pool.imap_unordered(encode_batch, batches, chunksize=2):
            for doc_index, token_ids in encoded_batch:
                split = "val" if doc_index % cfg.val_every == 0 else "train"
                writers[split].append(token_ids)
                processed += 1
                if processed % cfg.status_every == 0:
                    print_status(processed, writers, started)
                if writers["train"].total_tokens >= cfg.max_train_tokens:
                    pool.terminate()
                    flush_writers(writers)
                    print_status(processed, writers, started, done=True)
                    return
    flush_writers(writers)
    print_status(processed, writers, started, done=True)


def text_stream(cfg: DataBuildConfig):
    data = load_dataset(cfg.dataset_name, streaming=True, split=cfg.split)
    return data.shuffle(seed=cfg.shuffle_seed, buffer_size=cfg.shuffle_buffer)


def texts(stream, limit):
    for idx, row in enumerate(stream):
        if idx >= limit:
            return
        yield row["text"]


def document_batches(stream, batch_size):
    batch = []
    for idx, row in enumerate(stream):
        batch.append((idx, row["text"]))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def init_worker(tokenizer_path: str) -> None:
    """Initialize per-process tokenizer state for multiprocessing."""

    os.environ["RAYON_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    global _WORKER_TOKENIZER, _WORKER_EOT
    _WORKER_TOKENIZER = Tokenizer.from_file(tokenizer_path)
    _WORKER_EOT = _WORKER_TOKENIZER.token_to_id("<|endoftext|>")


def encode_batch(batch: list[tuple[int, str]]) -> list[tuple[int, np.ndarray]]:
    indices = [item[0] for item in batch]
    batch_texts = [item[1] for item in batch]
    encodings = _WORKER_TOKENIZER.encode_batch(batch_texts, add_special_tokens=False)
    encoded = []
    for idx, enc in zip(indices, encodings):
        arr = np.empty(len(enc.ids) + 1, dtype=np.uint32)
        arr[:-1] = enc.ids
        arr[-1] = _WORKER_EOT
        encoded.append((idx, arr))
    return encoded


def tokenizer_matches_vocab() -> bool:
    try:
        if not TOKENIZER_PATH.exists():
            return False
        with open(TOKENIZER_CONFIG_PATH) as f:
            return json.load(f).get("vocab_size") == LLMConfig.vocab_size
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return False


def pinned_empty(batch_size: int, seq_len: int) -> torch.Tensor:
    return torch.empty(
        (batch_size, seq_len),
        dtype=torch.int64,
        pin_memory=torch.cuda.is_available(),
    )


def remove_existing_shards(cfg: DataBuildConfig) -> None:
    for path in cfg.out_dir.glob(f"{cfg.slug}_*.bin"):
        path.unlink()


def flush_writers(writers: dict[str, SplitShardWriter]) -> None:
    for writer in writers.values():
        writer.flush()


def print_status(
    processed: int,
    writers: dict[str, SplitShardWriter],
    started: float,
    done: bool = False,
) -> None:
    elapsed = max(time.time() - started, 1e-9)
    prefix = "[done]" if done else "[status]"
    train = writers["train"].total_tokens
    val = writers["val"].total_tokens
    print(
        f"{prefix} docs={processed:,} train={train:,} val={val:,} "
        f"rate={(train + val) / elapsed / 1e6:.2f}M tok/s"
    )


def write_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def cycle(items):
    while True:
        yield from items

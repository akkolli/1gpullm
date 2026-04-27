import glob
import json
import mmap
import multiprocessing as mp
import os
import threading
from queue import Empty, Queue

import numpy as np
import torch

# Set thread environment vars BEFORE importing tokenizers — the Rust extension
# reads these at load time and they're sticky after that.
os.environ["RAYON_NUM_THREADS"] = "24"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import time

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

from llm import LLMConfig

MAX_DOCS = 1000
TOKENIZER_PATH = "tokenizer/tokenizer.json"
TOKENIZER_CONFIG_PATH = "tokenizer/tokenizer_config.json"


def dataset_iterator():
    i = 0
    for i, data in enumerate(dataset):
        if i >= MAX_DOCS:
            break
        yield data["text"]


def check_tokenizer_config():
    try:
        with open(TOKENIZER_CONFIG_PATH, "r") as f:
            config = json.load(f)
            return LLMConfig.vocab_size == config["vocab_size"]
    except:
        return False


# Pool worker globals — populated by _worker_init in each child process.
_TK = None
_EOT = None


def _worker_init(tokenizer_path: str) -> None:
    """Pool worker setup. Force each worker to use a single Rayon thread so
    process-level parallelism (the pool) doesn't fight thread-level parallelism
    (the tokenizer's internal Rayon pool). With N workers × 1 thread we get
    clean N-way parallelism; with N workers × 24 threads we'd thrash.

    The env vars must be set BEFORE the Tokenizer is loaded — Rayon reads
    RAYON_NUM_THREADS lazily on first thread-pool use, but once it's read the
    value is sticky.
    """
    os.environ["RAYON_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    global _TK, _EOT
    _TK = Tokenizer.from_file(tokenizer_path)
    _EOT = _TK.token_to_id("<|endoftext|>")


def _worker_encode_batch(batch):
    """Tokenize a chunk of (i, text) tuples. Returns a list of (i, np.uint32 array).

    Each returned array carries the doc's tokens plus a trailing <|endoftext|>
    separator. The tokenizer's TemplateProcessor already appends one EOT, so
    each doc ends up with two consecutive EOTs in the shard — preserved here
    to match the on-disk format the existing v1.4 shards used.
    """
    indices = [b[0] for b in batch]
    texts = [b[1] for b in batch]
    encs = _TK.encode_batch(texts)
    out = []
    for idx, enc in zip(indices, encs):
        ids = enc.ids
        arr = np.empty(len(ids) + 1, dtype=np.uint32)
        arr[: len(ids)] = ids
        arr[-1] = _EOT
        out.append((idx, arr))
    return out


class ShardedTokenDataset:
    """Memory-mapped reader for sharded uint32 token files.

    Layout on disk: dataset/fineweb/fineweb_{split}_{shard:04d}.bin
    Each file is a flat sequence of uint32 token IDs with <|endoftext|>
    between original documents.

    Sampling strategy: sequential cursor over a shuffled shard order, with a
    random offset into the first shard each epoch. Each step reads one
    contiguous (batch_size * context_len + 1)-token slice, which gets reshaped
    into the (B, T) batch. Sequential I/O is ~30x faster than random on NVMe,
    and fineweb's documents are already shuffled at write time so adjacent
    windows are statistically independent enough for pretraining.
    """

    def __init__(self, split: str, data_dir: str = "dataset/fineweb", dtype=np.uint32):
        pattern = os.path.join(data_dir, f"fineweb_{split}_*.bin")
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no shards matched {pattern}")

        self.dtype = dtype
        self.itemsize = np.dtype(dtype).itemsize  # bytes per token
        self.paths = paths

        # mmap with MADV_SEQUENTIAL: tells the kernel to aggressively read
        # ahead of the cursor and drop pages we've passed. This is what makes
        # the sequential read pattern actually fast — without it the kernel's
        # default heuristics still treat each access as potentially random.
        self.shards = []
        self._mmaps = []  # keep refs alive; np.frombuffer doesn't own them
        for p in paths:
            f = open(p, "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            mm.madvise(mmap.MADV_SEQUENTIAL)
            self.shards.append(np.frombuffer(mm, dtype=dtype))
            self._mmaps.append((f, mm))

        self.lengths = np.array([len(s) for s in self.shards], dtype=np.int64)
        self.total_tokens = int(self.lengths.sum())

        # Sequential cursor state (see _reset_epoch / get_batch). Owned by the
        # prefetch thread once it starts; nothing else should mutate it.
        self._shard_order = np.arange(len(self.shards))
        self._cursor_shard_pos = 0
        self._cursor_offset = 0
        self._reset_epoch(np.random.default_rng())

        # Synchronous pinned-buffer cache (used only on the device='cpu' path).
        self._pinned: dict = {}
        self._pinned_idx: dict = {}

        # Background prefetch state — lazily started on first GPU get_batch.
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_stop = threading.Event()
        self._prefetch_ready: Queue | None = None
        self._prefetch_shape: tuple | None = None
        self._prefetch_device: str | None = None

        print(
            f"[loader/{split}] {len(self.shards)} shards, "
            f"{self.total_tokens:,} tokens "
            f"({sum(os.path.getsize(p) for p in paths) / 1e9:.2f}GB)"
        )

    def __len__(self):
        return self.total_tokens

    def _reset_epoch(self, rng: np.random.Generator) -> None:
        """Shuffle shard order and pick a fresh random offset into shard 0."""
        self._shard_order = rng.permutation(len(self.shards))
        self._cursor_shard_pos = 0
        sid = int(self._shard_order[0])
        # Random start inside the first shard so we don't see the same windows
        # epoch after epoch. Cap below shard length so a full batch still fits.
        max_off = max(1, len(self.shards[sid]) // 2)
        self._cursor_offset = int(rng.integers(0, max_off))

    def _advance_shard(self, rng: np.random.Generator) -> None:
        self._cursor_shard_pos += 1
        if self._cursor_shard_pos >= len(self._shard_order):
            self._reset_epoch(rng)
        else:
            self._cursor_offset = 0

    def _get_pinned(self, batch_size: int, context_len: int):
        """Ping-pong pinned buffer cache, used only by the synchronous path."""
        key = (batch_size, context_len)
        bufs = self._pinned.get(key)
        if bufs is None:
            bufs = []
            for _ in range(2):
                x_pin = torch.empty(
                    (batch_size, context_len), dtype=torch.int64, pin_memory=True
                )
                y_pin = torch.empty(
                    (batch_size, context_len), dtype=torch.int64, pin_memory=True
                )
                bufs.append((x_pin, y_pin, x_pin.numpy(), y_pin.numpy()))
            self._pinned[key] = bufs
            self._pinned_idx[key] = 0
        idx = self._pinned_idx[key]
        self._pinned_idx[key] = 1 - idx
        return bufs[idx]

    def _fill_one(
        self,
        x_np: np.ndarray,
        y_np: np.ndarray,
        batch_size: int,
        context_len: int,
        rng: np.random.Generator,
    ) -> None:
        """Advance cursor and copy one (B*T + 1)-token contiguous slice."""
        n = batch_size * context_len
        sid = int(self._shard_order[self._cursor_shard_pos])
        shard = self.shards[sid]
        if self._cursor_offset + n + 1 > len(shard):
            self._advance_shard(rng)
            sid = int(self._shard_order[self._cursor_shard_pos])
            shard = self.shards[sid]
        i = self._cursor_offset
        np.copyto(
            x_np, shard[i : i + n].reshape(batch_size, context_len), casting="unsafe"
        )
        np.copyto(
            y_np,
            shard[i + 1 : i + 1 + n].reshape(batch_size, context_len),
            casting="unsafe",
        )
        self._cursor_offset += n

    def _start_prefetch(
        self,
        batch_size: int,
        context_len: int,
        device: str,
        num_slots: int = 4,
        queue_size: int = 2,
    ) -> None:
        # num_slots > queue_size so the producer always has a free buffer to
        # write into (we synchronize on each slot's prior H2D event before
        # reusing it).
        slots = []
        for _ in range(num_slots):
            x_pin = torch.empty(
                (batch_size, context_len), dtype=torch.int64, pin_memory=True
            )
            y_pin = torch.empty(
                (batch_size, context_len), dtype=torch.int64, pin_memory=True
            )
            slots.append(
                {
                    "x_pin": x_pin,
                    "y_pin": y_pin,
                    "x_np": x_pin.numpy(),
                    "y_np": y_pin.numpy(),
                    "event": None,  # CUDA event recorded after the H2D using this slot
                }
            )
        self._prefetch_ready = Queue(maxsize=queue_size)
        self._prefetch_stop.clear()
        self._prefetch_shape = (batch_size, context_len)
        self._prefetch_device = device
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop,
            args=(slots, batch_size, context_len, device),
            daemon=True,
        )
        self._prefetch_thread.start()

    def _stop_prefetch(self) -> None:
        if self._prefetch_thread is None:
            return
        self._prefetch_stop.set()
        # Drain the ready queue so the producer isn't stuck on Queue.put.
        try:
            while True:
                self._prefetch_ready.get_nowait()
        except Empty:
            pass
        self._prefetch_thread.join(timeout=2.0)
        self._prefetch_thread = None
        self._prefetch_ready = None
        self._prefetch_shape = None
        self._prefetch_device = None

    def _prefetch_loop(
        self,
        slots: list,
        batch_size: int,
        context_len: int,
        device: str,
    ) -> None:
        rng = np.random.default_rng()
        slot_idx = 0
        while not self._prefetch_stop.is_set():
            slot = slots[slot_idx]
            slot_idx = (slot_idx + 1) % len(slots)
            # Block CPU until the previous DMA reading from this pinned buffer
            # is done. Without this we'd race: producer overwrites bytes the
            # GPU is still copying out.
            if slot["event"] is not None:
                slot["event"].synchronize()

            self._fill_one(
                slot["x_np"], slot["y_np"], batch_size, context_len, rng
            )

            # Async H2D, then record an event so the consumer's stream can
            # wait on it. Issued from this thread; the event carries the
            # ordering across threads.
            x_gpu = slot["x_pin"].to(device, non_blocking=True)
            y_gpu = slot["y_pin"].to(device, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            slot["event"] = ev

            # Bounded queue: producer blocks here once it's `queue_size` ahead.
            # Use timeout so we periodically re-check the stop flag.
            while not self._prefetch_stop.is_set():
                try:
                    self._prefetch_ready.put((x_gpu, y_gpu, ev), timeout=0.5)
                    break
                except Exception:
                    continue

    def get_batch(
        self,
        batch_size: int,
        context_len: int,
        device: str = "cuda",
        rng: np.random.Generator = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next (x, y) batch.

        On CUDA: served from a background prefetch queue (~µs per call). The
        prefetcher fills pinned buffers and issues async H2D so the work
        overlaps with the previous step's forward/backward.

        On CPU: synchronous fill into a pinned buffer.
        """
        if device != "cpu":
            shape = (batch_size, context_len)
            if self._prefetch_thread is None:
                self._start_prefetch(batch_size, context_len, device)
            elif self._prefetch_shape != shape or self._prefetch_device != device:
                self._stop_prefetch()
                self._start_prefetch(batch_size, context_len, device)

            x_gpu, y_gpu, ev = self._prefetch_ready.get()
            # Make the consumer's current stream wait for the H2D event. The
            # CPU returns immediately; the next op on this stream (model
            # forward) will be ordered after the copy.
            ev.wait()
            return [x_gpu, y_gpu]

        # Synchronous CPU path (no prefetch).
        if rng is None:
            rng = np.random.default_rng()
        x_pin, y_pin, x_np, y_np = self._get_pinned(batch_size, context_len)
        self._fill_one(x_np, y_np, batch_size, context_len, rng)
        return [x_pin, y_pin]

    def __del__(self):
        try:
            self._stop_prefetch()
        except Exception:
            pass


if __name__ == "__main__":
    print(f"[init] rayon threads = {os.environ['RAYON_NUM_THREADS']}")
    print(f"[init] loading dataset...")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", streaming=True, split="train"
    )
    print(f"[init] dataset ready")

    # If tokenizer does not exist, make one
    if not os.path.exists(TOKENIZER_PATH) or not check_tokenizer_config():
        print(f"[tokenizer] tokenizer.json not found, training from scratch")
        tokenizer = Tokenizer(BPE(unk_token=None, byte_fallback=False))

        claude_regex_magic = (
            r"""'(?i:[sdmt]|ll|ve|re)|"""
            r"""[^\r\n\p{L}\p{N}]?+\p{L}+|"""
            r"""\p{N}{1,3}|"""
            r""" ?[^\s\p{L}\p{N}]++[\r\n]*|"""
            r"""\s*[\r\n]|\s+(?!\S)|\s+"""
        )

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    pattern=claude_regex_magic, behavior="isolated", invert=False
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )

        tokenizer.decoder = decoders.ByteLevel()

        SPECIALS = [
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
        trainer = BpeTrainer(
            vocab_size=LLMConfig.vocab_size,
            min_frequency=2,
            special_tokens=SPECIALS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )

        print(f"[tokenizer] training BPE on up to {MAX_DOCS:,} docs...")
        t0 = time.time()
        # Run training. This is CPU-bound and parallelized internally (Rust).
        # `length=MAX_DOCS` is just for the progress bar — it doesn't enforce
        # the limit; that's the iterator's job.
        tokenizer.train_from_iterator(
            dataset_iterator(), trainer=trainer, length=MAX_DOCS
        )
        print(f"[tokenizer] BPE training done in {(time.time() - t0) / 60:.1f} min")

        eot_id = tokenizer.token_to_id("<|endoftext|>")
        tokenizer.post_processor = processors.TemplateProcessing(
            single="$A <|endoftext|>",
            pair="$A <|endoftext|> $B <|endoftext|>",
            special_tokens=[("<|endoftext|>", eot_id)],
        )
        os.makedirs("tokenizer", exist_ok=True)
        tokenizer.save(TOKENIZER_PATH)
        print(f"[tokenizer] saved to tokenizer.json")
        with open(TOKENIZER_CONFIG_PATH, "w") as f:
            json.dump({"vocab_size": LLMConfig.vocab_size}, f)

        text = "def hello():\n    print('Hello, world!')  # 你好"
        enc = tokenizer.encode(text)
        print(f"vocab_size = {tokenizer.get_vocab_size()}")
        print(f"tokens     = {enc.tokens}")
        print(f"ids        = {enc.ids}")
        print(f"decoded    = {tokenizer.decode(enc.ids)}")

        print(f"[tokenizer] computing bytes/token on 10K held-out docs...")
        total_bytes, total_tokens = 0, 0
        for j, ex in enumerate(
            load_dataset(
                "HuggingFaceFW/fineweb",
                name="sample-10BT",
                streaming=True,
                split="train",
            ).skip(MAX_DOCS)
        ):
            if j >= 10_000:
                break
            total_bytes += len(ex["text"].encode("utf-8"))
            total_tokens += len(tokenizer.encode(ex["text"]).ids)
        print(f"bytes/token = {total_bytes / total_tokens:.2f}")

    else:
        print(f"[tokenizer] loading existing tokenizer.json")
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

        eot_id = tokenizer.token_to_id("<|endoftext|>")
        print(
            f"[tokenizer] vocab_size = {tokenizer.get_vocab_size()}, eot_id = {eot_id}"
        )
        exit()

    SHARD_SIZE = 250_000_000  # ~1GB per shard at uint32
    VAL_EVERY = 1000
    STATUS_EVERY = 50_000  # high-level progress print every N docs
    NPROCS = max(2, (os.cpu_count() or 4) - 2)
    WORKER_BATCH = 64  # docs per task sent to a worker; amortizes IPC

    def encode_both_splits():
        """Single pass over the dataset. The main process iterates the stream
        and dispatches doc batches to a worker pool; workers tokenize in
        parallel; the main process accumulates results into a preallocated
        per-split shard buffer and writes full shards to disk.
        """
        out_dir = "dataset/fineweb"
        os.makedirs(out_dir, exist_ok=True)

        print(f"[encode] output dir: {out_dir}/")
        print(
            f"[encode] shard size = {SHARD_SIZE:,} tokens (~{SHARD_SIZE * 4 / 1e9:.1f}GB each)"
        )
        print(f"[encode] {NPROCS} workers, worker_batch = {WORKER_BATCH}")
        print(f"[encode] val ratio = 1/{VAL_EVERY}")

        # Per-split state. One preallocated SHARD_SIZE buffer; on overflow,
        # write to disk and reset the cursor.
        state = {}
        for split in ("train", "val"):
            state[split] = {
                "buf": np.empty(SHARD_SIZE, dtype=np.uint32),
                "n": 0,  # tokens written into the current shard buffer
                "shard_idx": 0,
                "total_tokens": 0,
                "path": os.path.join(out_dir, f"fineweb_{split}_0000.bin"),
            }
            print(f"[encode] {split} shard 0: {state[split]['path']}")

        def flush_full(split):
            s = state[split]
            s["buf"].tofile(s["path"])
            size_gb = os.path.getsize(s["path"]) / 1e9
            print(
                f"  [shard/{split}] FINISHED {s['path']} "
                f"({s['n']:,} tokens, {size_gb:.2f}GB)"
            )
            s["shard_idx"] += 1
            s["path"] = os.path.join(
                out_dir, f"fineweb_{split}_{s['shard_idx']:04d}.bin"
            )
            s["n"] = 0

        def append(split, arr):
            """Append a doc's token array, splitting across shard boundaries
            when needed. Doc ordering within a shard isn't meaningful for
            pretraining, so the imap_unordered out-of-order delivery is fine.
            """
            s = state[split]
            offset = 0
            m = len(arr)
            while offset < m:
                room = SHARD_SIZE - s["n"]
                chunk = (m - offset) if (m - offset) < room else room
                s["buf"][s["n"] : s["n"] + chunk] = arr[offset : offset + chunk]
                s["n"] += chunk
                s["total_tokens"] += chunk
                offset += chunk
                if s["n"] >= SHARD_SIZE:
                    flush_full(split)

        def doc_batches():
            """Yield (i, text) lists of size WORKER_BATCH from the dataset."""
            batch = []
            for i, ex in enumerate(dataset):
                batch.append((i, ex["text"]))
                if len(batch) >= WORKER_BATCH:
                    yield batch
                    batch = []
            if batch:
                yield batch

        print(f"[encode] starting at {time.strftime('%H:%M:%S')}")
        t_start = time.time()
        last_status = t_start
        last_status_docs = 0
        processed = 0

        # fork is the default on Linux and skips re-import overhead. The worker
        # initializer pins each child to RAYON_NUM_THREADS=1 before it loads
        # the tokenizer, so per-process Rayon doesn't oversubscribe the box.
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=NPROCS,
            initializer=_worker_init,
            initargs=(TOKENIZER_PATH,),
        ) as pool:
            try:
                for results in pool.imap_unordered(
                    _worker_encode_batch, doc_batches(), chunksize=2
                ):
                    for i, arr in results:
                        split = "val" if i % VAL_EVERY == 0 else "train"
                        append(split, arr)
                        processed += 1

                        if processed % STATUS_EVERY == 0:
                            now = time.time()
                            interval_docs = processed - last_status_docs
                            interval_dt = now - last_status
                            rate = (
                                interval_docs / interval_dt if interval_dt > 0 else 0
                            )
                            elapsed_min = (now - t_start) / 60
                            print(
                                f"[status] doc {processed:,} | {rate:.0f} docs/s | "
                                f"elapsed {elapsed_min:.1f}min | "
                                f"train={state['train']['total_tokens']:,} tok "
                                f"({state['train']['shard_idx'] + 1} shards) | "
                                f"val={state['val']['total_tokens']:,} tok "
                                f"({state['val']['shard_idx'] + 1} shards)"
                            )
                            last_status = now
                            last_status_docs = processed
                print(f"[encode] stream exhausted")
            finally:
                print(f"[encode] finalizing...")
                for split, s in state.items():
                    if s["n"] > 0:
                        s["buf"][: s["n"]].tofile(s["path"])
                        size_gb = os.path.getsize(s["path"]) / 1e9
                        print(
                            f"  [shard/{split}] FINISHED {s['path']} "
                            f"({s['n']:,} tokens, {size_gb:.2f}GB)"
                        )
                    print(
                        f"[done/{split}] {s['total_tokens']:,} tokens "
                        f"across {s['shard_idx'] + 1} shard(s)"
                    )

                total_dt = time.time() - t_start
                total_tok = sum(s["total_tokens"] for s in state.values())
                rate_m = total_tok / total_dt / 1e6 if total_dt > 0 else 0
                print(
                    f"[done] total: {total_tok:,} tokens in {total_dt / 60:.1f} min "
                    f"({rate_m:.2f}M tok/s)"
                )

    encode_both_splits()

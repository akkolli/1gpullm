import json
import os

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

print(f"[init] rayon threads = {os.environ['RAYON_NUM_THREADS']}")
print(f"[init] loading dataset...")

dataset = load_dataset(
    "HuggingFaceFW/fineweb", name="sample-10BT", streaming=True, split="train"
)
print(f"[init] dataset ready")

MAX_DOCS = 200_000


def dataset_iterator():
    i = 0
    for i, data in enumerate(dataset):
        if i >= MAX_DOCS:
            break
        yield data["text"]


# If tokenizer does not exist, make one
if not os.path.exists("./tokenizer.json"):
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
    tokenizer.train_from_iterator(dataset_iterator(), trainer=trainer, length=MAX_DOCS)
    print(f"[tokenizer] BPE training done in {(time.time() - t0) / 60:.1f} min")

    eot_id = tokenizer.token_to_id("<|endoftext|>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A <|endoftext|>",
        pair="$A <|endoftext|> $B <|endoftext|>",
        special_tokens=[("<|endoftext|>", eot_id)],
    )

    tokenizer.save("tokenizer.json")
    print(f"[tokenizer] saved to tokenizer.json")

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
            "HuggingFaceFW/fineweb", name="sample-10BT", streaming=True, split="train"
        ).skip(MAX_DOCS)
    ):
        if j >= 10_000:
            break
        total_bytes += len(ex["text"].encode("utf-8"))
        total_tokens += len(tokenizer.encode(ex["text"]).ids)
    print(f"bytes/token = {total_bytes / total_tokens:.2f}")

else:
    print(f"[tokenizer] loading existing tokenizer.json")
    tokenizer = Tokenizer.from_file("tokenizer.json")
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    print(f"[tokenizer] vocab_size = {tokenizer.get_vocab_size()}, eot_id = {eot_id}")


SHARD_SIZE = 250_000_000  # ~1GB per shard at uint32
ENCODE_BATCH = 4000  # bumped from 1000 to give rayon more parallelism per call
FLUSH_EVERY = 20_000_000  # bumped to amortize numpy/tofile overhead
VAL_EVERY = 1000
STATUS_EVERY = 50_000  # high-level progress print every N docs


def encode_both_splits():
    """Single pass over dataset, routing each doc to train or val shards."""
    out_dir = "dataset/fineweb"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[encode] output dir: {out_dir}/")
    print(
        f"[encode] shard size = {SHARD_SIZE:,} tokens (~{SHARD_SIZE * 4 / 1e9:.1f}GB each)"
    )
    print(f"[encode] encode_batch = {ENCODE_BATCH}, flush_every = {FLUSH_EVERY:,}")
    print(f"[encode] val ratio = 1/{VAL_EVERY}")

    # Per-split state. Keeps the two outputs independent but lets us share
    # the iteration over the source dataset.
    state = {}
    for split in ("train", "val"):
        path = os.path.join(out_dir, f"fineweb_{split}_0000.bin")
        state[split] = {
            "buf": [],
            "batch": [],
            "shard_idx": 0,
            "shard_tokens": 0,
            "total_tokens": 0,
            "f": open(path, "wb"),
            "path": path,
        }
        print(f"[encode] opened {path}")

    def open_shard(split, idx):
        path = os.path.join(out_dir, f"fineweb_{split}_{idx:04d}.bin")
        return open(path, "wb"), path

    def flush_batch(split):
        s = state[split]
        if not s["batch"]:
            return
        t0 = time.time()
        n_docs = len(s["batch"])
        encs = tokenizer.encode_batch(s["batch"])
        for enc in encs:
            s["buf"].extend(enc.ids)
            s["buf"].append(eot_id)
        s["batch"].clear()
        if split == "train":
            dt = time.time() - t0
            rate = n_docs / dt if dt > 0 else 0
            print(f"  [encode/{split}] {n_docs} docs in {dt:.2f}s ({rate:.0f} docs/s)")

    def flush_buf(split):
        s = state[split]
        if not s["buf"]:
            return
        t0 = time.time()
        n_tokens = len(s["buf"])
        offset = 0
        buf = s["buf"]
        while offset < len(buf):
            room = SHARD_SIZE - s["shard_tokens"]
            chunk = buf[offset : offset + room]
            np.array(chunk, dtype=np.uint32).tofile(s["f"])
            s["shard_tokens"] += len(chunk)
            s["total_tokens"] += len(chunk)
            offset += len(chunk)
            if s["shard_tokens"] >= SHARD_SIZE:
                s["f"].close()
                size_gb = os.path.getsize(s["path"]) / 1e9
                print(
                    f"  [shard/{split}] FINISHED {s['path']} "
                    f"({s['shard_tokens']:,} tokens, {size_gb:.2f}GB)"
                )
                s["shard_idx"] += 1
                s["f"], s["path"] = open_shard(split, s["shard_idx"])
                s["shard_tokens"] = 0
                print(f"  [shard/{split}] opened {s['path']}")
        s["buf"].clear()
        if split == "train":
            dt = time.time() - t0
            print(f"  [flush/{split}] wrote {n_tokens:,} tokens in {dt:.2f}s")

    print(f"[encode] starting at {time.strftime('%H:%M:%S')}")
    t_start = time.time()
    last_status = t_start
    last_status_docs = 0

    try:
        for i, ex in enumerate(tqdm(dataset, desc="encoding")):
            split = "val" if i % VAL_EVERY == 0 else "train"
            s = state[split]
            s["batch"].append(ex["text"])
            if len(s["batch"]) >= ENCODE_BATCH:
                flush_batch(split)
            if len(s["buf"]) >= FLUSH_EVERY:
                flush_buf(split)

            if (i + 1) % STATUS_EVERY == 0:
                now = time.time()
                interval_docs = (i + 1) - last_status_docs
                interval_dt = now - last_status
                rate = interval_docs / interval_dt if interval_dt > 0 else 0
                train_tok = state["train"]["total_tokens"] + len(state["train"]["buf"])
                val_tok = state["val"]["total_tokens"] + len(state["val"]["buf"])
                elapsed_min = (now - t_start) / 60
                print(
                    f"[status] doc {i + 1:,} | {rate:.0f} docs/s | "
                    f"elapsed {elapsed_min:.1f}min | "
                    f"train={train_tok:,} tok ({state['train']['shard_idx'] + 1} shards) | "
                    f"val={val_tok:,} tok ({state['val']['shard_idx'] + 1} shards)"
                )
                last_status = now
                last_status_docs = i + 1

        print(f"[encode] stream exhausted, draining buffers...")
        for split in state:
            flush_batch(split)
            flush_buf(split)
    finally:
        print(f"[encode] finalizing...")
        for split, s in state.items():
            if not s["f"].closed:
                s["f"].close()
                size_gb = os.path.getsize(s["path"]) / 1e9
                print(
                    f"  [shard/{split}] FINISHED {s['path']} "
                    f"({s['shard_tokens']:,} tokens, {size_gb:.2f}GB)"
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

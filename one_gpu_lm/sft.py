import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .checkpoint import (
    load_training_state,
    save_model,
    save_training_state,
)
from .model import LLM, param_breakdown, sample
from .training import (
    apply_precision,
    compute_mfu,
    cosine_with_warmup,
    DEFAULT_GPU_NAME,
    ensure_history_keys,
    estimate_training_flops_per_token,
    format_flops,
    format_mfu,
    resolve_mfu_peak_tflops,
    write_json,
)


IGNORE_INDEX = -100
CHAT_ROLES = ("system", "user", "assistant")
ROLE_ALIASES = {
    "system": "system",
    "human": "user",
    "user": "user",
    "prompter": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "model": "assistant",
}


@dataclass(frozen=True)
class SFTConfig:
    """Single-GPU supervised fine-tuning configuration."""

    run_name: str = "chat-sft"
    epochs: int = 1
    train_steps: int = 10_000
    val_steps: int = 50
    batch_size: int = 16
    val_interval: int = 1
    peak_lr: float = 2e-5
    lr_warmup: int = 100
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.0
    z_loss_coef: float = 1e-5
    gradient_checkpointing: bool = False
    precision: str = "bf16"
    compile_model: bool = True
    checkpoint_interval: int = 1
    # Set > 0 to override GPU_PEAK_TFLOPS_BY_PRECISION.
    mfu_peak_tflops: float = 0.0

    @property
    def out_dir(self) -> Path:
        return Path("checkpoints") / self.run_name

    @property
    def total_steps(self) -> int:
        return self.epochs * self.train_steps


class ChatSFTDataset:
    """On-the-fly chat SFT batches from common Hugging Face dataset schemas."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer: Tokenizer,
        seq_len: int,
        dataset_config: str | None = None,
        streaming: bool = True,
        shuffle: bool = False,
        shuffle_seed: int = 42,
        shuffle_buffer: int = 10_000,
        max_examples: int | None = None,
        system_prompt: str = "",
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.streaming = streaming
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.shuffle_buffer = shuffle_buffer
        self.max_examples = max_examples
        self.system_prompt = system_prompt
        self.pad_id = special_id(tokenizer, "<|endoftext|>")
        self._epoch = 0
        self._iterator = self._new_iterator()

    def get_batch(
        self,
        batch_size: int,
        seq_len: int,
        device: str = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len != self.seq_len:
            raise ValueError(f"dataset seq_len={self.seq_len}, requested {seq_len}")

        xs, ys = [], []
        skipped = 0
        max_skips = max(1000, batch_size * 100)
        while len(xs) < batch_size:
            row = self._next_row()
            pair = encode_chat_row(
                row,
                self.tokenizer,
                self.seq_len,
                self.pad_id,
                self.system_prompt,
            )
            if pair is None:
                skipped += 1
                if skipped > max_skips:
                    raise RuntimeError("could not find enough assistant-labeled examples")
                continue
            x, y = pair
            xs.append(x)
            ys.append(y)

        return torch.stack(xs).to(device), torch.stack(ys).to(device)

    def _next_row(self):
        while True:
            try:
                return next(self._iterator)
            except StopIteration:
                self._iterator = self._new_iterator()

    def _new_iterator(self):
        kwargs = {
            "split": self.split,
            "streaming": self.streaming,
        }
        if self.dataset_config:
            kwargs["name"] = self.dataset_config
        data = load_dataset(self.dataset_name, **kwargs)

        if self.shuffle:
            seed = self.shuffle_seed + self._epoch
            if self.streaming:
                data = data.shuffle(seed=seed, buffer_size=self.shuffle_buffer)
            else:
                data = data.shuffle(seed=seed)
        if self.max_examples is not None:
            if self.streaming:
                data = data.take(self.max_examples)
            else:
                data = data.select(range(min(self.max_examples, len(data))))
        self._epoch += 1
        return iter(data)


def train_sft(
    model: LLM,
    train_data: ChatSFTDataset,
    val_data: ChatSFTDataset | None,
    cfg: SFTConfig,
    resume: bool = False,
) -> tuple[torch.nn.Module, dict]:
    """Supervised fine-tune `model` on chat batches and write artifacts."""

    if not torch.cuda.is_available():
        raise RuntimeError("SFT requires a CUDA device")
    torch.set_float32_matmul_precision("medium")
    model.config.gradient_checkpointing = cfg.gradient_checkpointing
    seq_len = model.config.seq_len
    model = model.cuda().train()
    total_params = param_breakdown(model)
    flops_per_token = estimate_training_flops_per_token(model, total_params)
    mfu_peak_tflops = resolve_mfu_peak_tflops(cfg.precision, cfg.mfu_peak_tflops)
    apply_precision(model, cfg.precision)
    if cfg.compile_model:
        model = torch.compile(model, mode="reduce-overhead")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.peak_lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        fused=True,
    )
    schedulers = [
        LambdaLR(
            optimizer,
            cosine_with_warmup(cfg.total_steps, cfg.lr_warmup, cfg.min_lr_ratio),
        )
    ]

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    state_path = cfg.out_dir / "train_state.pt"
    start_epoch, history = 0, {"train_loss": [], "val_loss": []}
    if resume and state_path.exists():
        start_epoch, history = load_training_state(
            state_path,
            model,
            [optimizer],
            schedulers,
            device="cuda",
        )
    ensure_history_keys(history)
    supervised_history = history.setdefault("supervised_tokens_per_second", [])
    if len(supervised_history) < len(history.get("train_loss", [])):
        supervised_history.extend(
            [float("nan")] * (len(history["train_loss"]) - len(supervised_history))
        )
    supervised_tokens_history = history.setdefault("supervised_tokens", [])
    if len(supervised_tokens_history) < len(history.get("train_loss", [])):
        supervised_tokens_history.extend(
            [float("nan")]
            * (len(history["train_loss"]) - len(supervised_tokens_history))
        )

    print(
        f"[sft] {cfg.total_steps:,} steps, "
        f"{cfg.batch_size} batch, seq_len={seq_len}, "
        f"{format_flops(flops_per_token)} flops/token, "
        f"mfu_peak={mfu_peak_tflops:.1f} TFLOPS"
    )
    started = time.time()
    last_epoch_time = 0.0
    for epoch in range(start_epoch, cfg.epochs):
        train_loss, supervised_tokens, last_epoch_time = train_sft_epoch(
            model,
            train_data,
            optimizer,
            schedulers,
            cfg,
            seq_len,
        )
        val_loss = float("nan")
        if val_data is not None and cfg.val_steps > 0:
            if (epoch + 1) % cfg.val_interval == 0 or epoch == cfg.epochs - 1:
                val_loss = validate_sft(model, val_data, cfg, seq_len)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        processed_tokens = cfg.batch_size * seq_len * cfg.train_steps
        total_processed_tokens = processed_tokens * (epoch + 1)
        tok_s = processed_tokens / max(last_epoch_time, 1e-9)
        supervised_tok_s = supervised_tokens / max(last_epoch_time, 1e-9)
        flops_s = tok_s * flops_per_token
        mfu = compute_mfu(flops_s, mfu_peak_tflops)
        history["tokens_processed"].append(total_processed_tokens)
        history["supervised_tokens"].append(supervised_tokens)
        history["tokens_per_second"].append(tok_s)
        history["supervised_tokens_per_second"].append(supervised_tok_s)
        history["flops_per_second"].append(flops_s)
        history["mfu"].append(mfu)
        print(
            f"[{epoch + 1}/{cfg.epochs}] train={train_loss:.4f} "
            f"val={val_loss:.4f} tokens={total_processed_tokens / 1e6:.1f}M "
            f"tok/s={tok_s:.0f} "
            f"supervised_tok/s={supervised_tok_s:.0f} "
            f"flops/s={format_flops(flops_s)} mfu={format_mfu(mfu)}"
        )
        if cfg.checkpoint_interval and (epoch + 1) % cfg.checkpoint_interval == 0:
            save_training_state(
                state_path,
                model,
                [optimizer],
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
        "total_params": total_params,
        "flops_per_token": flops_per_token,
        "mfu_peak_tflops": mfu_peak_tflops,
        "mfu_peak_gpu": DEFAULT_GPU_NAME if cfg.mfu_peak_tflops <= 0 else "override",
    }
    write_json(cfg.out_dir / "metrics.json", metrics)
    save_training_state(
        state_path,
        model,
        [optimizer],
        schedulers,
        cfg,
        cfg.epochs,
        history,
        metrics,
    )
    return model, metrics


def train_sft_epoch(
    model: torch.nn.Module,
    train_data: ChatSFTDataset,
    optimizer: torch.optim.Optimizer,
    schedulers: list[LambdaLR],
    cfg: SFTConfig,
    seq_len: int,
) -> tuple[float, int, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    loss_sum = torch.zeros((), device="cuda")
    supervised_tokens = 0
    for _ in range(cfg.train_steps):
        x, y = train_data.get_batch(cfg.batch_size, seq_len, device="cuda")
        supervised_tokens += int((y != IGNORE_INDEX).sum().item())
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(
                x,
                targets=y,
                z_loss_coef=cfg.z_loss_coef,
                ignore_index=IGNORE_INDEX,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        for sched in schedulers:
            sched.step()
        loss_sum += loss.detach()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return (loss_sum / cfg.train_steps).item(), supervised_tokens, elapsed


@torch.inference_mode()
def validate_sft(
    model: torch.nn.Module,
    val_data: ChatSFTDataset,
    cfg: SFTConfig,
    seq_len: int,
) -> float:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    for _ in range(cfg.val_steps):
        x, y = val_data.get_batch(cfg.batch_size, seq_len, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss_sum += model(x, targets=y, ignore_index=IGNORE_INDEX).item()
    model.train(was_training)
    return loss_sum / cfg.val_steps


def encode_chat_row(
    row: dict,
    tokenizer: Tokenizer,
    seq_len: int,
    pad_id: int,
    system_prompt: str = "",
) -> tuple[torch.Tensor, torch.Tensor] | None:
    messages = normalize_messages(row)
    if system_prompt and not any(msg["role"] == "system" for msg in messages):
        messages = [{"role": "system", "content": system_prompt}] + messages
    return encode_chat_messages(tokenizer, messages, seq_len, pad_id)


def encode_chat_messages(
    tokenizer: Tokenizer,
    messages: list[dict],
    seq_len: int,
    pad_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    pad = special_id(tokenizer, "<|endoftext|>") if pad_id is None else pad_id
    eot = special_id(tokenizer, "<|endoftext|>")
    ids, loss_mask = [], []

    for msg in messages:
        role = msg.get("role")
        content = clean_text(msg.get("content"))
        if role not in CHAT_ROLES or not content:
            continue
        append_special(ids, loss_mask, tokenizer, f"<|{role}|>", train=False)
        append_text(ids, loss_mask, tokenizer, "\n", train=False)
        train = role == "assistant"
        append_text(ids, loss_mask, tokenizer, content, train=train)
        if train:
            ids.append(eot)
            loss_mask.append(True)
        append_text(ids, loss_mask, tokenizer, "\n", train=False)

    if len(ids) < 2:
        return None

    limit = seq_len + 1
    if len(ids) > limit:
        ids = ids[-limit:]
        loss_mask = loss_mask[-limit:]

    if not any(loss_mask[1:]):
        return None

    x_ids = ids[:-1]
    y_ids = ids[1:]
    y_mask = loss_mask[1:]
    y_ids = [target if keep else IGNORE_INDEX for target, keep in zip(y_ids, y_mask)]

    pad_len = seq_len - len(x_ids)
    if pad_len > 0:
        x_ids.extend([pad] * pad_len)
        y_ids.extend([IGNORE_INDEX] * pad_len)

    return torch.tensor(x_ids, dtype=torch.long), torch.tensor(y_ids, dtype=torch.long)


def normalize_messages(row: dict) -> list[dict]:
    for key in ("messages", "conversations", "chosen"):
        value = maybe_json(row.get(key))
        if isinstance(value, list):
            messages = normalize_message_list(value)
            if messages:
                return messages
        if isinstance(value, dict):
            nested = maybe_json(value.get("messages"))
            if isinstance(nested, list):
                messages = normalize_message_list(nested)
                if messages:
                    return messages

    instruction = clean_text(row.get("instruction"))
    output = clean_text(row.get("output") or row.get("response"))
    if instruction and output:
        input_text = clean_text(row.get("input"))
        user = f"{instruction}\n\n{input_text}" if input_text else instruction
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": output},
        ]

    prompt = clean_text(row.get("prompt") or row.get("question"))
    answer = clean_text(
        row.get("response")
        or row.get("completion")
        or row.get("answer")
        or row.get("output")
    )
    if prompt and answer:
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]

    return []


def normalize_message_list(items: list) -> list[dict]:
    messages = []
    for item in items:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or item.get("from") or item.get("speaker")
        role = ROLE_ALIASES.get(clean_text(role).lower())
        content = clean_text(item.get("content") or item.get("value") or item.get("text"))
        if role in CHAT_ROLES and content:
            messages.append({"role": role, "content": content})
    return messages


def build_chat_prompt_ids(
    tokenizer: Tokenizer,
    prompt: str,
    system_prompt: str = "",
) -> list[int]:
    ids, mask = [], []
    if system_prompt:
        append_special(ids, mask, tokenizer, "<|system|>", train=False)
        append_text(ids, mask, tokenizer, "\n", train=False)
        append_text(ids, mask, tokenizer, system_prompt, train=False)
        append_text(ids, mask, tokenizer, "\n", train=False)
    append_special(ids, mask, tokenizer, "<|user|>", train=False)
    append_text(ids, mask, tokenizer, "\n", train=False)
    append_text(ids, mask, tokenizer, prompt, train=False)
    append_text(ids, mask, tokenizer, "\n", train=False)
    append_special(ids, mask, tokenizer, "<|assistant|>", train=False)
    append_text(ids, mask, tokenizer, "\n", train=False)
    return ids or [special_id(tokenizer, "<|endoftext|>")]


@torch.inference_mode()
def stream_chat(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
    system_prompt: str = "",
) -> None:
    eot = special_id(tokenizer, "<|endoftext|>")
    ids = build_chat_prompt_ids(tokenizer, prompt, system_prompt)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)

    new_ids, printed = [], ""
    for _ in range(max_new_tokens):
        logits = model(tokens[:, -model.config.seq_len :])[:, -1]
        next_token = sample(logits, temperature, top_k if top_k > 0 else None)
        token_id = int(next_token.item())
        if token_id == eot:
            break
        tokens = torch.cat((tokens, next_token), dim=1)
        new_ids.append(token_id)
        decoded = tokenizer.decode(new_ids, skip_special_tokens=False)
        sys.stdout.write(decoded[len(printed) :])
        sys.stdout.flush()
        printed = decoded
    sys.stdout.write("\n")


def append_special(
    ids: list[int],
    loss_mask: list[bool],
    tokenizer: Tokenizer,
    token: str,
    train: bool,
) -> None:
    ids.append(special_id(tokenizer, token))
    loss_mask.append(train)


def append_text(
    ids: list[int],
    loss_mask: list[bool],
    tokenizer: Tokenizer,
    text: str,
    train: bool,
) -> None:
    token_ids = tokenizer.encode(text, add_special_tokens=False).ids
    ids.extend(token_ids)
    loss_mask.extend([train] * len(token_ids))


def special_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"tokenizer is missing required special token {token}")
    return token_id


def clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def maybe_json(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        import json

        return json.loads(text)
    except json.JSONDecodeError:
        return value

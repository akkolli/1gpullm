from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class LLMConfig:
    """Architecture knobs for the decoder-only language model.

    `n_dim` must divide evenly across `n_heads`; `seq_len` is both the RoPE
    cache length and the maximum context used by generation/evaluation.
    """

    n_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    vocab_size: int = 32384
    seq_len: int = 1024
    gradient_checkpointing: bool = False


class SwiGLU(nn.Module):
    """Feed-forward block used by modern decoder LMs."""

    def __init__(self, dim: int, hidden_mult: float = 8 / 3):
        super().__init__()
        hidden_dim = round_up(int(dim * hidden_mult), multiple=64)
        self.gate_up = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    """Pre-norm attention + SwiGLU block with RoPE and causal SDPA."""

    def __init__(self, config: LLMConfig):
        super().__init__()
        if config.n_dim % config.n_heads:
            raise ValueError("n_dim must be divisible by n_heads")
        self.config = config
        self.head_dim = config.n_dim // config.n_heads
        self.attn_norm = nn.RMSNorm(config.n_dim)
        self.qkv = nn.Linear(config.n_dim, 3 * config.n_dim, bias=False)
        self.proj = nn.Linear(config.n_dim, config.n_dim, bias=False)
        self.ffn_norm = nn.RMSNorm(config.n_dim)
        self.ffn = SwiGLU(config.n_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(self.attn_norm(x))
        qkv = qkv.view(batch, seq_len, 3, self.config.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = apply_rope(q, k, rope_cos[:seq_len], rope_sin[:seq_len])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(batch, seq_len, dim))
        return x + self.ffn(self.ffn_norm(x))


class LLM(nn.Module):
    """Small GPT-style decoder.

    Forward accepts token ids shaped `[batch, seq]`. In inference mode it
    returns logits shaped `[batch, seq, vocab]`; in training mode, pass
    `targets` to return the memory-efficient scalar LM loss directly.
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.embeddings = nn.Embedding(config.vocab_size, config.n_dim)
        self.norm = nn.RMSNorm(config.n_dim)
        self.lm_head = nn.Linear(config.n_dim, config.vocab_size, bias=False)
        self.register_rope()
        self.apply(init_weights)
        # Weight tying reduces parameters and usually improves sample quality.
        self.lm_head.weight = self.embeddings.weight

    def register_rope(self) -> None:
        cos, sin = rope_cache(
            head_dim=self.config.n_dim // self.config.n_heads,
            seq_len=self.config.seq_len,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        z_loss_coef: float = 0.0,
        loss_chunk_size: int = 8192,
        ignore_index: int | None = None,
    ) -> torch.Tensor:
        x = self.embeddings(x)
        for block in self.layers:
            x = self.run_block(block, x)
        x = self.norm(x)
        if targets is None:
            return self.lm_head(x)
        return chunked_lm_loss(
            x,
            self.lm_head.weight,
            targets,
            z_loss_coef,
            loss_chunk_size,
            ignore_index,
        )

    def run_block(self, block: TransformerBlock, x: torch.Tensor) -> torch.Tensor:
        if not (self.training and self.config.gradient_checkpointing):
            return block(x, self.rope_cos, self.rope_sin)
        # No dropout in this model, so preserving RNG state is unnecessary.
        return checkpoint(
            lambda h: block(h, self.rope_cos, self.rope_sin),
            x,
            use_reentrant=False,
            preserve_rng_state=False,
        )

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(tokens[:, -self.config.seq_len :])[:, -1]
            next_token = sample(logits, temperature=temperature, top_k=top_k)
            tokens = torch.cat((tokens, next_token), dim=1)
        self.train(was_training)
        return tokens


class ChunkedLMLoss(torch.autograd.Function):
    """Cross entropy + optional z-loss without materializing full logits.

    `hidden` is flattened to `[tokens, dim]`, `weight` is `[vocab, dim]`, and
    `target` is `[tokens]`. The backward pass recomputes each logits chunk,
    trading extra matmul work for much lower peak memory.
    """

    @staticmethod
    def forward(ctx, hidden, weight, target, z_coef, chunk_size):
        ctx.save_for_backward(hidden, weight, target)
        ctx.z_coef = float(z_coef)
        ctx.chunk_size = int(chunk_size)

        loss_sum = torch.zeros((), device=hidden.device, dtype=torch.float32)
        z_sum = torch.zeros_like(loss_sum)
        for start, end in chunks(hidden.size(0), ctx.chunk_size):
            logits = hidden[start:end] @ weight.t()
            log_z = torch.logsumexp(logits.float(), dim=-1)
            target_logits = logits.float().gather(1, target[start:end, None]).squeeze(1)
            loss_sum = loss_sum + (log_z - target_logits).sum()
            z_sum = z_sum + log_z.square().sum()
        return loss_sum / hidden.size(0) + ctx.z_coef * z_sum / hidden.size(0)

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, target = ctx.saved_tensors
        n_tokens = hidden.size(0)
        grad_hidden = torch.empty_like(hidden)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        ce_scale = grad_out / n_tokens
        z_scale = grad_out * ctx.z_coef / n_tokens

        for start, end in chunks(n_tokens, ctx.chunk_size):
            # Recompute the same chunk of logits instead of saving it in forward.
            logits = (hidden[start:end] @ weight.t()).float()
            log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
            probs = (logits - log_z).exp()
            grad_logits = probs * ce_scale
            rows = torch.arange(end - start, device=hidden.device)
            grad_logits[rows, target[start:end]] -= ce_scale.to(grad_logits.dtype)
            grad_logits.add_(probs * (2.0 * log_z * z_scale))
            grad_logits = grad_logits.to(weight.dtype)
            grad_hidden[start:end] = grad_logits @ weight
            grad_weight.add_(grad_logits.t().float() @ hidden[start:end].float())

        return grad_hidden, grad_weight.to(weight.dtype), None, None, None


class MaskedChunkedLMLoss(torch.autograd.Function):
    """Chunked LM loss variant that skips ignored target positions."""

    @staticmethod
    def forward(ctx, hidden, weight, target, z_coef, chunk_size, ignore_index):
        ctx.save_for_backward(hidden, weight, target)
        ctx.z_coef = float(z_coef)
        ctx.chunk_size = int(chunk_size)
        ctx.ignore_index = int(ignore_index)
        ctx.valid_count = int((target != ctx.ignore_index).sum().item())

        if ctx.valid_count == 0:
            return torch.zeros((), device=hidden.device, dtype=torch.float32)

        loss_sum = torch.zeros((), device=hidden.device, dtype=torch.float32)
        z_sum = torch.zeros_like(loss_sum)
        for start, end in chunks(hidden.size(0), ctx.chunk_size):
            chunk_target = target[start:end]
            valid = chunk_target != ctx.ignore_index
            if not valid.any():
                continue
            chunk_hidden = hidden[start:end][valid]
            logits = chunk_hidden @ weight.t()
            log_z = torch.logsumexp(logits.float(), dim=-1)
            target_logits = logits.float().gather(1, chunk_target[valid].unsqueeze(1))
            target_logits = target_logits.squeeze(1)
            loss_sum = loss_sum + (log_z - target_logits).sum()
            z_sum = z_sum + log_z.square().sum()
        return loss_sum / ctx.valid_count + ctx.z_coef * z_sum / ctx.valid_count

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, target = ctx.saved_tensors
        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)

        if ctx.valid_count == 0:
            return grad_hidden, grad_weight.to(weight.dtype), None, None, None, None

        ce_scale = grad_out / ctx.valid_count
        z_scale = grad_out * ctx.z_coef / ctx.valid_count

        for start, end in chunks(hidden.size(0), ctx.chunk_size):
            chunk_target = target[start:end]
            valid = chunk_target != ctx.ignore_index
            if not valid.any():
                continue
            valid_rows = valid.nonzero(as_tuple=False).flatten()
            valid_target = chunk_target[valid_rows]
            chunk_hidden = hidden[start:end][valid_rows]

            logits = (chunk_hidden @ weight.t()).float()
            log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
            probs = (logits - log_z).exp()
            grad_logits = probs * ce_scale
            rows = torch.arange(valid_target.numel(), device=hidden.device)
            grad_logits[rows, valid_target] -= ce_scale.to(grad_logits.dtype)
            grad_logits.add_(probs * (2.0 * log_z * z_scale))
            grad_logits = grad_logits.to(weight.dtype)

            grad_chunk = grad_logits @ weight
            grad_hidden[start:end].index_copy_(0, valid_rows, grad_chunk)
            grad_weight.add_(grad_logits.t().float() @ chunk_hidden.float())

        return grad_hidden, grad_weight.to(weight.dtype), None, None, None, None


def chunked_lm_loss(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    z_coef: float = 0.0,
    chunk_size: int = 8192,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """Return scalar LM loss for `hidden: [batch, seq, dim]`."""

    hidden = hidden.reshape(-1, hidden.size(-1))
    targets = targets.reshape(-1)
    if ignore_index is not None:
        return MaskedChunkedLMLoss.apply(
            hidden,
            weight,
            targets,
            z_coef,
            chunk_size,
            ignore_index,
        )
    return ChunkedLMLoss.apply(
        hidden,
        weight,
        targets,
        z_coef,
        chunk_size,
    )


_chunked_lm_loss = chunked_lm_loss


def sample(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int | None = None,
) -> torch.Tensor:
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        values = torch.topk(logits, min(top_k, logits.size(-1))).values
        logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


def rope_cache(
    head_dim: int,
    seq_len: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv_freq)
    phases = torch.cat((freqs, freqs), dim=-1)
    return phases.cos(), phases.sin()


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.to(q.dtype)[None, None]
    sin = sin.to(q.dtype)[None, None]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, std=0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


def round_up(value: int, multiple: int):
    return multiple * ((value + multiple - 1) // multiple)


def chunks(size: int, chunk_size: int):
    for start in range(0, size, chunk_size):
        yield start, min(start + chunk_size, size)


def param_breakdown(model: nn.Module) -> int:
    seen = set()
    counts = {}
    for name, module in model.named_children():
        for param in module.parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            counts[name] = counts.get(name, 0) + param.numel()
    total = sum(counts.values())
    for name, count in counts.items():
        print(f"  {name:20s} {count:>12,}  ({100 * count / total:.1f}%)")
    print(f"  {'TOTAL':20s} {total:>12,}")
    return total

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LLMConfig:
    n_dim: int = 768  # Dimensions of the token vectors
    n_layers: int = 12  # Number of layers in the language model
    n_heads: int = 24
    vocab_size: int = 32384  # Number of unique tokens
    seq_len: int = 128  # Context length


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 8 / 3):
        super().__init__()
        hidden = ((int(dim * hidden_mult) + 63) // 64) * 64
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


def precompute_rope_cache(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute (cos, sin) tables of shape (max_seq_len, head_dim) for RoPE.
    Frequencies are computed on the half-dim and duplicated so the rotation is
    a plain elementwise multiply against `rotate_half(x)`.
    """
    assert head_dim % 2 == 0, "RoPE requires even head_dim"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # q, k: (B, H, T, Dh). cos/sin: (T, Dh), broadcast over B, H.
    cos = cos.to(q.dtype)[None, None, :, :]
    sin = sin.to(q.dtype)[None, None, :, :]
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class Layer(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        assert self.config.n_dim % self.config.n_heads == 0
        self.config.head_dim = self.config.n_dim // self.config.n_heads
        self.qkv = nn.Linear(self.config.n_dim, self.config.n_dim * 3, bias=False)
        self.mix = nn.Linear(self.config.n_dim, self.config.n_dim, bias=False)
        self.norm1 = nn.RMSNorm(normalized_shape=self.config.n_dim)
        self.norm2 = nn.RMSNorm(normalized_shape=self.config.n_dim)
        # self.ffn = nn.Sequential(
        #     nn.Linear(self.config.n_dim, self.config.n_dim * 4, bias=False),
        #     nn.GELU(),
        #     nn.Linear(self.config.n_dim * 4, self.config.n_dim, bias=False),
        # )
        self.ffn = SwiGLU(self.config.n_dim)
        # self.register_buffer(
        #     "mask",
        #     torch.triu(
        #         torch.ones(
        #             self.config.seq_len,
        #             self.config.seq_len,
        #             dtype=torch.bool,
        #         ),
        #         diagonal=1,
        #     ),
        # )

    def forward(self, x, cos, sin, causal=True):

        B, T, C = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.view(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        k = k.view(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        v = v.view(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        out = x + self.mix(attn)
        out = out + self.ffn(self.norm2(out))
        return out


class LLM(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Layer(self.config) for _ in range(self.config.n_layers)]
        )
        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.n_dim)
        self.norm = nn.RMSNorm(normalized_shape=self.config.n_dim)
        self.lm_head = nn.Linear(self.config.n_dim, self.config.vocab_size, bias=False)

        head_dim = self.config.n_dim // self.config.n_heads
        cos, sin = precompute_rope_cache(head_dim, self.config.seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        # self.init_std = math.sqrt(2 / self.config.n_dim)
        self.init_std = 0.02
        self.apply(self._init_weights)
        # self.lm_head.weight = self.embeddings.weight

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.init_std)

    def forward(self, x):
        """
        Args:
            x: Input tokens
            These tokens represent the input sequence, usually in the shape of B, S, D
        Returns:
            Log probs, in the shape of B, S
        """
        B, S = x.shape
        x = self.embeddings(x)
        for l in self.layers:
            x = l(x, self.rope_cos, self.rope_sin)

        x = self.norm(x)
        x = self.lm_head(x)
        return x

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.0, top_k=None):
        """Autoregressively extend `idx` by `max_new_tokens` tokens.

        Args:
            idx: (B, S) int64 token ids — the prompt
            max_new_tokens: how many tokens to append
            temperature: 0.0 = greedy argmax; >0 = sample from softmax(logits/T)
            top_k: if set, restrict sampling to the top-k logits before softmax

        Returns:
            (B, S + max_new_tokens) int64 tensor — prompt + generated tokens
        """
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.seq_len :]
            logits = self.forward(idx_cond)[:, -1, :]  # (B, V)
            if temperature == 0.0:
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, k)
                    logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        if was_training:
            self.train()
        return idx


def param_breakdown(model):
    total = 0
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        total += n
        print(
            f"  {name:20s} {n:>12,}  ({100 * n / sum(p.numel() for p in model.parameters()):.1f}%)"
        )
    print(f"  {'TOTAL':20s} {total:>12,}")
    return total

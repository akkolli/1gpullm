import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LLMConfig:
    n_dim = 64  # Dimensions of the token vectors
    n_layers = 10  # Number of layers in the language model
    vocab_size = 1024  # Number of unique tokens
    seq_len = 128  # Context length


class Layer(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(self.config.n_dim, self.config.n_dim * 3)
        self.ffn = nn.Sequential(
            nn.Linear(self.config.n_dim, self.config.n_dim * 4),
            nn.ReLU(),
            nn.Linear(self.config.n_dim * 4, self.config.n_dim),
        )
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(
                    self.config.seq_len,
                    self.config.seq_len,
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        T = q.size(-2)
        scores = (q @ k.transpose(-1, -2)) * self.config.n_dim**-0.5
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = x + attn @ v
        out = out + self.ffn(out)
        return out


class LLM(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Layer(self.config) for _ in range(self.config.n_layers)]
        )
        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.n_dim)
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(normalized_shape=self.config.n_dim)
                for _ in range(self.config.n_layers)
            ]
        )
        self.lm_head = nn.Linear(self.config.n_dim, self.config.vocab_size)

        self.register_buffer("pos_idx", torch.arange(self.config.seq_len))
        self.pos = nn.Embedding(self.config.seq_len, self.config.n_dim)
        # self.init_std = math.sqrt(2 / self.config.n_dim)
        self.init_std = 0.02
        self.apply(self._init_weights)

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
        pos_tok = self.pos(self.pos_idx[:S])
        x = x + pos_tok
        for l, norm in zip(self.layers, self.norms):
            x = norm(l(x))

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

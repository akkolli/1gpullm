from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LLMConfig:
    n_dim = 10  # Dimensions of the token vectors
    n_layers = 10  # Number of layers in the language model
    vocab_size = 1024  # Number of unique tokens
    seq_len = 64  # Context length


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

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scores = (
            F.softmax(q @ k.transpose(-1, -2), dim=-1) / self.config.n_dim**-0.5
        ).tril()
        out = scores @ v
        out = self.ffn(out)
        return out


class LLM(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Layer(self.config) for _ in range(self.config.n_layers)]
        )
        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.n_dim)
        self.norm = nn.LayerNorm(normalized_shape=self.config.n_dim)
        self.lm_head = nn.Linear(self.config.n_dim, self.config.vocab_size)

    def forward(self, x):
        """
        Args:
            x: Input tokens
            These tokens represent the input sequence, usually in the shape of B, S, D
        Returns:
            Log probs, in the shape of B, S
        """
        x = self.embeddings(x)
        for l in self.layers:
            x = self.norm(x + l(x))

        x = self.lm_head(x)
        return x


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

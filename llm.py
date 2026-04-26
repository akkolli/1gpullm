from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LLMConfig:
    n_dim = 10  # Dimensions of the token vectors
    n_layers = 10  # Number of layers in the language model
    vocab_size = 128_000  # Number of unique tokens


class Layer(nn.Module):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.qkv = nn.Linear(self.config.n_dim, self.config.n_dim * 3)
        self.ffn = nn.Sequential(
            nn.Linear(self.config.n_dim, self.config.n_dim * 4),
            nn.ReLU(),
            nn.Linear(self.config.n_dim * 4, self.config.n_dim),
        )

    def forward(self, x):
        q, k, v = self.qkv(x).split(split_size_or_sections=3, dim=-1)
        scores = (F.softmax(q @ k.T) / self.config.n_dim**-0.5).tril()
        out = v @ scores
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

        return x

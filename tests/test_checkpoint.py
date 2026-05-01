import tempfile
import unittest
from pathlib import Path

import torch

from one_gpu_lm.checkpoint import load_model, save_model
from one_gpu_lm.model import LLM, LLMConfig


class CheckpointTest(unittest.TestCase):
    def test_model_weights_round_trip(self):
        cfg = LLMConfig(n_dim=16, n_layers=1, n_heads=4, vocab_size=32, seq_len=8)
        model = LLM(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "final.pth"
            save_model(path, model)
            loaded = load_model(checkpoint=path, device="cpu", config=cfg)

        x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
        self.assertTrue(torch.allclose(model(x), loaded(x)))


if __name__ == "__main__":
    unittest.main()

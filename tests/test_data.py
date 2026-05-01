import tempfile
import unittest
from pathlib import Path

import numpy as np

from one_gpu_lm.data import ShardedTokenDataset


class ShardedTokenDatasetTest(unittest.TestCase):
    def test_cpu_batch_is_next_token_shifted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy_train_0000.bin"
            np.arange(128, dtype=np.uint32).tofile(path)

            ds = ShardedTokenDataset("train", data_dir=tmp, dataset_slug="toy")
            x, y = ds.get_batch(2, 8, device="cpu", rng=np.random.default_rng(0))
            ds.close()

        self.assertEqual(tuple(x.shape), (2, 8))
        self.assertEqual(tuple(y.shape), (2, 8))
        self.assertTrue(np.array_equal(y.numpy(), x.numpy() + 1))


if __name__ == "__main__":
    unittest.main()

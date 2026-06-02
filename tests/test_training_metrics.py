import unittest

from one_gpu_lm.training import (
    GPU_PEAK_TFLOPS_BY_PRECISION,
    resolve_mfu_peak_tflops,
)


class TrainingMetricsTest(unittest.TestCase):
    def test_default_mfu_peak_tracks_precision(self):
        self.assertEqual(
            resolve_mfu_peak_tflops("bf16"),
            GPU_PEAK_TFLOPS_BY_PRECISION["bf16"],
        )
        self.assertEqual(
            resolve_mfu_peak_tflops("fp8"),
            GPU_PEAK_TFLOPS_BY_PRECISION["fp8"],
        )

    def test_mfu_peak_can_be_overridden(self):
        self.assertEqual(resolve_mfu_peak_tflops("bf16", 123.0), 123.0)


if __name__ == "__main__":
    unittest.main()

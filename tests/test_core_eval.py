import unittest

from one_gpu_lm.core_eval import center, common_prefix, common_suffix, crop_spans


class CoreEvalHelpersTest(unittest.TestCase):
    def test_common_prefix_and_suffix(self):
        self.assertEqual(common_prefix([[1, 2, 3], [1, 2, 4]]), 2)
        self.assertEqual(common_suffix([[1, 2, 3], [0, 2, 3]]), 2)

    def test_center_random_baseline(self):
        self.assertAlmostEqual(center(0.55, 0.10), 0.5)

    def test_crop_invalidates_lost_scored_span(self):
        rows, starts, ends = crop_spans([[1, 2, 3, 4]], [1], [4], max_len=2)
        self.assertEqual(rows, [[3, 4]])
        self.assertEqual(starts, [None])
        self.assertEqual(ends, [None])


if __name__ == "__main__":
    unittest.main()

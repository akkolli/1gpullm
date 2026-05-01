import unittest

import torch
import torch.nn.functional as F

from one_gpu_lm.model import chunked_lm_loss


class ChunkedLMLossTest(unittest.TestCase):
    def test_matches_reference_loss_and_gradients(self):
        torch.manual_seed(0)
        hidden = torch.randn(2, 5, 7, requires_grad=True)
        weight = torch.randn(11, 7, requires_grad=True)
        targets = torch.randint(0, 11, (2, 5))

        ref_hidden = hidden.detach().clone().requires_grad_()
        ref_weight = weight.detach().clone().requires_grad_()

        actual = chunked_lm_loss(hidden, weight, targets, z_coef=1e-4, chunk_size=3)
        logits = ref_hidden @ ref_weight.t()
        log_z = torch.logsumexp(logits.float(), dim=-1)
        expected = F.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1))
        expected = expected + 1e-4 * log_z.square().mean()

        actual.backward()
        expected.backward()

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(hidden.grad, ref_hidden.grad, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(weight.grad, ref_weight.grad, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()

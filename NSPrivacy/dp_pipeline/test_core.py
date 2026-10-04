"""Focused regression tests for manuscript-critical behavior."""

import math
import unittest

import torch

from nsprivacy_dp import _record_cross_entropy


class CoreTests(unittest.TestCase):
    def test_class_weight_does_not_cancel(self):
        logits = torch.tensor([[0.25, -0.25]])
        target = torch.tensor(1)
        unit = _record_cross_entropy(logits, target, torch.tensor([1.0, 1.0]))
        weighted = _record_cross_entropy(logits, target, torch.tensor([1.0, 3.0]))
        self.assertTrue(math.isclose(weighted.item(), 3.0 * unit.item(), rel_tol=1e-6))


if __name__ == "__main__":
    unittest.main()

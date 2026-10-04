"""Regression tests for preprocessing and evaluation utilities."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import pue
from prepare_tabular import prepare


class ExperimentTests(unittest.TestCase):
    def test_pue_reference_values(self):
        self.assertAlmostEqual(pue(0.9732, 0.530, 1.0)["pue"], 0.96528)
        self.assertAlmostEqual(pue(0.7123, 0.508, 0.0)["pue"], 0.67852)

    def test_preprocessing_split_and_class_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            output = Path(directory) / "prepared.npz"
            rows = 100
            frame = pd.DataFrame(
                {
                    "a": np.arange(rows, dtype=float),
                    "b": np.tile([1.0, 2.0], rows // 2),
                    "constant": 7.0,
                    "label": np.tile(["x", "y"], rows // 2),
                }
            )
            frame.to_csv(source, index=False)
            prepare(str(source), "label", str(output), seed=42)
            arrays = np.load(output, allow_pickle=False)
            self.assertEqual(len(arrays["x_train"]), 80)
            self.assertEqual(len(arrays["x_val"]), 10)
            self.assertEqual(len(arrays["x_test"]), 10)
            self.assertEqual(arrays["x_train"].shape[1], 2)
            self.assertEqual(len(arrays["class_weights"]), 2)


if __name__ == "__main__":
    unittest.main()

# Experiment Utilities

This directory contains the public preprocessing and evaluation utilities used with the NSPrivacy training pipeline.

- [`prepare_tabular.py`](prepare_tabular.py) creates stratified 80/10/10 splits, fits preprocessing on the training split, and records balanced class weights.
- [`run_seeds.py`](run_seeds.py) launches the five manuscript seeds for one privacy budget.
- [`evaluation.py`](evaluation.py) provides an output-based membership-inference classifier, PUE calculation, bootstrap intervals, and paired testing.
- [`run_nonprivate_baselines.py`](run_nonprivate_baselines.py) runs Random Forest, gradient-boosting, and MLP reference models.
- [`test_experiments.py`](test_experiments.py) checks the preprocessing split and PUE calculations.

Dataset-specific cohort construction and label consolidation must be completed before the generic tabular preprocessor is used. See the repository [data guide](../../DATA.md). New runs write to user-selected output paths and do not overwrite the manuscript result records.

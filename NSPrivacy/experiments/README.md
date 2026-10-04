# Experiment Utilities

These utilities make the public companion executable beyond the synthetic
smoke test.

- `prepare_tabular.py` creates leakage-safe 80/10/10 splits, fits preprocessing
  on the training split, and records balanced class weights.
- `run_seeds.py` launches the five manuscript seeds for one privacy budget.
- `evaluation.py` provides an output-based membership attack, PUE calculation,
  bootstrap intervals, and Bonferroni-corrected paired tests.
- `run_nonprivate_baselines.py` runs reproducible Random Forest, gradient
  boosting, and MLP reference models. It does not relabel the gradient-boosting
  reference as XGBoost.

Dataset-specific cohort construction for MIMIC-IV must follow its access and
data-use requirements. Reported result files remain frozen and are never
overwritten by these utilities.

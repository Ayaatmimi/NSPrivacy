# NSPrivacy implementation

This directory contains the model, private-training pipeline, experiment utilities, and manuscript result records.

## Components

- [`dp_pipeline/`](dp_pipeline/): NSPrivacy model, DP-SGD, privacy filtering, RDP accounting, evaluation, and audit export
- [`experiments/`](experiments/): preprocessing, repeated-seed execution, reference baselines, and evaluation utilities
- [`reported_results/`](reported_results/): manuscript result tables and analyses
- [`CODE_STATUS.md`](CODE_STATUS.md): implementation and accounting notes

## Quick check

From the repository root:

```bash
python -m pip install -r NSPrivacy/requirements.txt
python NSPrivacy/dp_pipeline/smoke_test.py
```

## Manuscript configuration

The selected configuration uses 100 epochs, an expected Poisson batch size of 256, clipping bound 1.0, `erase_sigma=0.1`, `lambda_sparse=1e-4`, `lambda_design=1e-3`, and 20 mask-adaptation epochs. The DP-SGD noise multiplier is calibrated separately for each target privacy budget.

See the root [reproducibility guide](../REPRODUCIBILITY.md), [data guide](../DATA.md), and [paper-to-code map](../MANUSCRIPT_ALIGNMENT.md) for the complete workflow.

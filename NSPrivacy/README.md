# NSPrivacy

NSPrivacy combines data minimization, individual non-retention, structural privacy design, and DP-SGD with RDP accounting.

## Components

- `dp_pipeline/`: private training and privacy accounting
- `reported_results/`: manuscript result tables
- `notebooks/`: experiment notebook directory
- `CODE_STATUS.md`: implementation notes

## Quick start

```bash
cd dp_pipeline
python -m pip install -r requirements.txt
python smoke_test.py
```

Prepare an NPZ file containing `x_train` and `y_train`, then follow `dp_pipeline/README.md` for training.

The manuscript configuration uses 100 epochs, an expected batch size of 256,
clipping bound 1.0, `erase_sigma=0.1`, `lambda_sparse=1e-4`,
`lambda_design=1e-3`, and 20 mask-adaptation epochs.

Raw datasets are not distributed. MIMIC-IV access remains subject to its data-use requirements.

# NSPrivacy research artifacts

NSPrivacy combines data minimization, individual non-retention, structural privacy design, and DP-SGD with RDP accounting.

## Important status

The files in `reported_results/` preserve the values reported in the manuscript. The code in `dp_pipeline/` is a candidate implementation aligned with the final training algorithm. It is not yet a verified end-to-end reproduction of all manuscript results.

Read `CODE_STATUS.md` before using or citing the code.

## Quick check

```bash
cd dp_pipeline
python -m pip install -r requirements.txt
python smoke_test.py
```

For real data, prepare an NPZ file containing `x_train` and `y_train`, then follow `dp_pipeline/README.md`.

Raw datasets are not distributed. MIMIC-IV access remains subject to its data-use requirements.

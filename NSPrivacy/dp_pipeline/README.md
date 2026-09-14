# Candidate DP-SGD and RDP pipeline

This directory contains the candidate implementation aligned with the manuscript training algorithm.

## Input format

Create an NPZ file with:

- `x_train`: a two-dimensional float array
- `y_train`: integer labels in `0,...,C-1`
- optionally `x_test` and `y_test`

Preprocessing is intentionally external. If preprocessing depends on protected records, account for it separately or use parameters fixed from public data.

## Install and test

```bash
python -m pip install -r requirements.txt
python smoke_test.py
```

## Train

```bash
python train.py \
  --data path/to/preprocessed_data.npz \
  --output outputs/run_01 \
  --num-classes 8 \
  --epsilon 0.114 \
  --delta YOUR_FIXED_DELTA \
  --epochs 100 \
  --batch-size YOUR_BATCH_SIZE \
  --max-grad-norm YOUR_CLIPPING_NORM \
  --erase-sigma YOUR_ERASURE_SCALE \
  --warm-epochs YOUR_MASK_WARMUP
```

Replace every `YOUR_...` value with the fixed experimental setting. Do not infer missing settings from the reported tables.

## Privacy notes

- The RDP accountant is updated after every private optimizer step.
- Sampling and DP-gradient noise use operating-system-seeded generators by default.
- A fixed `--private-seed` is available only for controlled debugging. It should not be used for a privacy-critical release.
- Fixed epochs are used. Private validation-based early stopping is not included.
- Class weights are accepted only as fixed values. Computing them from protected labels requires separate analysis.
- Evaluation data and published evaluation statistics are outside this training accountant.

The pipeline is not yet a verified five-seed reproduction of the manuscript.

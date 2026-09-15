# NSPrivacy DP-SGD and RDP pipeline

This directory implements the NSPrivacy private-training procedure.

## Input format

Create an NPZ file with:

- `x_train`: a two-dimensional float array
- `y_train`: integer labels in `0,...,C-1`
- optionally `x_test` and `y_test`

Preprocessing is maintained as a separate dataset-specific stage.

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

Set the `YOUR_...` arguments to the corresponding experimental configuration.

## Privacy accounting

- The RDP accountant is updated after every private optimizer step.
- Sampling and DP-gradient noise use operating-system-seeded generators by default.
- Fixed epochs are used for the private training schedule.
- Class weights can be supplied in encoded class order.
- The generated run record stores the target budget, achieved budget, sampling rate, noise multiplier, clipping norm, and completed updates.

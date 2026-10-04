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
  --delta 1e-5 \
  --epochs 100 \
  --batch-size 256 \
  --learning-rate 1e-3 \
  --min-learning-rate 1e-5 \
  --max-grad-norm 1.0 \
  --erase-sigma 0.1 \
  --lambda-sparse 1e-4 \
  --lambda-design 1e-3 \
  --warm-epochs 20
```

The command shows the selected manuscript configuration. Change the input and
output paths and set the target privacy budget for the run.

## Privacy accounting

- The RDP accountant is updated after every private optimizer step.
- Sampling and DP-gradient noise use operating-system-seeded generators by default.
- Fixed epochs are used for the private training schedule.
- Class weights can be supplied in encoded class order.
- Empty Poisson batches receive a noise-only update and are conservatively counted.
- Gradients and noise are normalized by the fixed expected batch size.
- The released run record omits private RNG seeds and class-weight values.
- The audit record stores only public configuration, accountant outputs, and
  epoch-level module status. It does not store masks, sensitivities,
  representations, per-record gradients, or mini-batch statistics.

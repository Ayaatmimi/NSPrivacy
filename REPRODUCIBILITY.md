# Reproducibility Guide

## Environment

The automated workflow uses Python 3.11 and the package versions pinned in `NSPrivacy/requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r NSPrivacy/requirements.txt
```

## Automated checks

From the repository root:

```bash
python NSPrivacy/dp_pipeline/smoke_test.py

cd NSPrivacy/dp_pipeline
python -m unittest -v test_core.py

cd ../experiments
python -m unittest -v test_experiments.py
```

The same checks run in [GitHub Actions](https://github.com/Ayaatmimi/NSPrivacy/actions/workflows/dp-smoke-test.yml).

## Data preparation

Follow [`DATA.md`](DATA.md) to obtain and derive CIC-IDS2017 or MIMIC-IV inputs. For an already derived tabular CSV:

```bash
python NSPrivacy/experiments/prepare_tabular.py \
  --csv path/to/data.csv \
  --label-column label \
  --output prepared/data.npz \
  --seed 42
```

The command removes exact duplicates and constant features, creates a stratified 80/10/10 split, fits median imputation, 1st/99th-percentile winsorization, and standardization on the training split, and records balanced class weights.

## Private training

Run one target budget over the five manuscript seeds:

```bash
python NSPrivacy/experiments/run_seeds.py \
  --data prepared/data.npz \
  --output outputs/epsilon_0114 \
  --num-classes 8 \
  --epsilon 0.114 \
  --delta 1e-5
```

The manuscript seeds are 42, 123, 456, 789, and 1024. The public seed controls data-independent initialization and reporting. Sampling and DP-gradient noise use a separate operating-system-seeded generator unless a debug seed is explicitly supplied.

Each run writes:

- `model_state.pt`;
- `run_record.json`, containing the public configuration, evaluation metrics, and privacy audit record.

## Evaluation

[`NSPrivacy/experiments/evaluation.py`](NSPrivacy/experiments/evaluation.py) provides:

- an output-based membership-inference classifier;
- PUE calculation;
- bootstrap confidence intervals;
- paired tests with Bonferroni correction.

[`NSPrivacy/experiments/run_nonprivate_baselines.py`](NSPrivacy/experiments/run_nonprivate_baselines.py) provides reproducible non-private reference models. External private baselines should be run from their official implementations under the common protocol described in the manuscript.

## Paper result records

The Markdown files under [`NSPrivacy/reported_results/`](NSPrivacy/reported_results/) mirror the result values reported in the current manuscript. They are kept separate from new run outputs, which are written to user-selected directories.

## Privacy-accounting scope

The RDP accountant covers all scheduled DP-SGD updates in the released private trainer, including mask adaptation and conservatively counted noise-only updates from empty Poisson batches. The released audit record is restricted to public configuration, accountant outputs, and module status.

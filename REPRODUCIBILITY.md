# Reproducibility Guide

## Environment

The repository pins the Python packages used by the automated tests. Python
3.11 is used in GitHub Actions.

```bash
python -m pip install -r NSPrivacy/requirements.txt
```

## Data preparation

Obtain CIC-IDS2017 or MIMIC-IV from its official source. MIMIC-IV remains
subject to its credentialing and data-use requirements. For an already derived
tabular CSV:

```bash
python NSPrivacy/experiments/prepare_tabular.py \
  --csv path/to/data.csv \
  --label-column label \
  --output prepared/data.npz \
  --seed 42
```

The command removes exact duplicates and constant features, creates a
stratified 80/10/10 split, fits median imputation, 1st/99th-percentile
winsorization, and standardization on the training split, and records balanced
class weights.

## Private training

Run one target budget over the five manuscript seeds:

```bash
python NSPrivacy/experiments/run_seeds.py \
  --data prepared/data.npz \
  --output outputs/epsilon_0114 \
  --num-classes 8 \
  --epsilon 0.114
```

The private RNG remains operating-system-seeded unless a debug seed is
explicitly supplied. The released run record does not disclose that seed.

## Evaluation

`NSPrivacy/experiments/evaluation.py` contains the output-based membership
attack, PUE calculation, bootstrap confidence intervals, and paired testing.
`run_nonprivate_baselines.py` supplies reproducible reference baselines.

The Markdown files under `reported_results/` are frozen manuscript records.
New runs write to user-selected output directories and never overwrite them.

## Scope

The repository provides the NSPrivacy model, private optimizer, accountant,
audit records, data-preparation utilities, repeated-seed launcher, and core
evaluation utilities. External private baselines should be run from their
official implementations under the protocol stated in the manuscript.

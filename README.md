# NSPrivacy

[![DP pipeline smoke test](https://github.com/Ayaatmimi/NSPrivacy/actions/workflows/dp-smoke-test.yml/badge.svg)](https://github.com/Ayaatmimi/NSPrivacy/actions/workflows/dp-smoke-test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation and experimental companion for:

**NSPrivacy: A Neuro-Symbolic Framework for Privacy-Preserving Data Classification**  
Zhi-Jie Wang and Mimuna Mimi

NSPrivacy integrates a differentiable policy engine with private optimization for tabular classification. The policy engine combines latent data minimization, sensitivity-aware non-retention perturbation, and structural representation control. The complete per-record objective is trained with DP-SGD and accounted with Rényi differential privacy (RDP).

## Repository structure

| Path | Contents |
| --- | --- |
| [`NSPrivacy/dp_pipeline/`](NSPrivacy/dp_pipeline/) | NSPrivacy model, per-record clipping, Gaussian DP noise, privacy filtering, and RDP accounting |
| [`NSPrivacy/experiments/`](NSPrivacy/experiments/) | Tabular preprocessing, repeated-seed execution, baseline utilities, membership inference, PUE, bootstrap intervals, and paired tests |
| [`NSPrivacy/reported_results/`](NSPrivacy/reported_results/) | Result tables and analyses reported in the manuscript |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Environment, commands, and evaluation workflow |
| [`DATA.md`](DATA.md) | Dataset access and input preparation |
| [`MANUSCRIPT_ALIGNMENT.md`](MANUSCRIPT_ALIGNMENT.md) | Paper-to-code map |
| [`CITATION.cff`](CITATION.cff) | Citation metadata |

## Quick start

```bash
git clone https://github.com/Ayaatmimi/NSPrivacy.git
cd NSPrivacy

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r NSPrivacy/requirements.txt

python NSPrivacy/dp_pipeline/smoke_test.py
```

The smoke test uses synthetic data and checks the complete private-training path. The GitHub Actions workflow also runs the core and experiment regression tests.

## Prepare data

Obtain CIC-IDS2017 or MIMIC-IV from its official source and prepare the derived tabular file described in [`DATA.md`](DATA.md). For a derived CSV:

```bash
python NSPrivacy/experiments/prepare_tabular.py \
  --csv path/to/data.csv \
  --label-column label \
  --output prepared/data.npz \
  --seed 42
```

The output contains stratified training, validation, and test splits. Preprocessing parameters are fitted on the training split and applied unchanged to the other splits.

## Run NSPrivacy

```bash
python NSPrivacy/dp_pipeline/train.py \
  --data prepared/data.npz \
  --output outputs/nsprivacy_e0114 \
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

For the five manuscript seeds, use [`run_seeds.py`](NSPrivacy/experiments/run_seeds.py). Each run writes the model state, public configuration, evaluation metrics, and privacy audit record to the selected output directory.

## Reported results

The manuscript result records are indexed in [`NSPrivacy/reported_results/RESULTS_INDEX.md`](NSPrivacy/reported_results/RESULTS_INDEX.md). They cover the CIC-IDS2017 and MIMIC-IV comparisons, per-class recall, ablation study, membership-inference evaluation, PUE analysis, hyperparameter sensitivity, training dynamics, and computational cost.

## Data availability

The repository does not redistribute CIC-IDS2017 or MIMIC-IV. Users should obtain each dataset from its official source and follow the applicable access and data-use requirements.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Until an archival publication record is available, cite the manuscript title and this repository URL.

## License

The code is released under the [MIT License](LICENSE). Dataset licenses and access conditions remain governed by their respective providers.

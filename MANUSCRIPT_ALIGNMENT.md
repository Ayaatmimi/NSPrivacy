# Paper-to-Code Map

This page maps the main manuscript components to their public implementation and documentation.

| Manuscript component | Repository location |
| --- | --- |
| Feature encoder, mask network, classification head | [`NSPrivacy/dp_pipeline/nsprivacy_dp.py`](NSPrivacy/dp_pipeline/nsprivacy_dp.py) |
| Sensitivity-aware latent perturbation | [`NSPrivacyNet.forward`](NSPrivacy/dp_pipeline/nsprivacy_dp.py) |
| Complete per-record objective | [`_single_record_loss`](NSPrivacy/dp_pipeline/nsprivacy_dp.py) |
| Per-record gradient calculation and flat clipping | [`_per_record_gradients` and `_set_private_gradients`](NSPrivacy/dp_pipeline/nsprivacy_dp.py) |
| Poisson sampling, Gaussian mechanism, and RDP accounting | [`train_private`](NSPrivacy/dp_pipeline/nsprivacy_dp.py) |
| Projected privacy-budget check | [`train_private`](NSPrivacy/dp_pipeline/nsprivacy_dp.py) |
| Mask adaptation and freezing | [`train_private`](NSPrivacy/dp_pipeline/nsprivacy_dp.py) |
| Restricted technical audit trail | [`train_private` and `save_run`](NSPrivacy/dp_pipeline/nsprivacy_dp.py) |
| Tabular preprocessing | [`NSPrivacy/experiments/prepare_tabular.py`](NSPrivacy/experiments/prepare_tabular.py) |
| Five-seed execution | [`NSPrivacy/experiments/run_seeds.py`](NSPrivacy/experiments/run_seeds.py) |
| MIA, PUE, bootstrap intervals, and paired testing | [`NSPrivacy/experiments/evaluation.py`](NSPrivacy/experiments/evaluation.py) |
| Manuscript tables and analyses | [`NSPrivacy/reported_results/`](NSPrivacy/reported_results/) |
| Automated checks | [`.github/workflows/dp-smoke-test.yml`](.github/workflows/dp-smoke-test.yml) |

## Audit record

The released audit record contains public mechanism settings, privacy-accountant outputs, completed update counts, and policy-module activation status. It excludes raw records, masks, sensitivity vectors, latent representations, per-record gradients, private RNG seeds, and private mini-batch statistics.

## Paper results

The files under [`NSPrivacy/reported_results/`](NSPrivacy/reported_results/) provide a readable record of the values reported in the manuscript. New executions write to user-selected output directories and remain separate from these paper records.

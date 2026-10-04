# NSPrivacy

[![DP pipeline smoke test](https://github.com/Ayaatmimi/NSPrivacy/actions/workflows/dp-smoke-test.yml/badge.svg)](https://github.com/Ayaatmimi/NSPrivacy/actions/workflows/dp-smoke-test.yml)

Official project repository for the NSPrivacy manuscript.

NSPrivacy combines data minimization, individual non-retention, structural privacy design, and differentially private optimization for privacy-preserving classification.

## Repository contents

- `NSPrivacy/dp_pipeline/`: NSPrivacy DP-SGD and RDP-accounting implementation
- `NSPrivacy/experiments/`: preprocessing, repeated-seed, baseline, MIA, PUE,
  bootstrap, and statistical-testing utilities
- `NSPrivacy/reported_results/`: all results reported in the manuscript
- `NSPrivacy/notebooks/`: experiment notebook directory
- `NSPrivacy/CODE_STATUS.md`: implementation and privacy-accounting notes

The private trainer uses Poisson sampling, complete per-record gradient
clipping, Gaussian noise, RDP composition, projected budget checks, and a
restricted technical audit trail.

## Data availability

CIC-IDS2017 and MIMIC-IV are not redistributed by this repository. Users should obtain the datasets from their official sources and follow their respective access requirements.

## Reproducibility

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the verified environment,
data format, five-seed launcher, and evaluation workflow. See
[`MANUSCRIPT_ALIGNMENT.md`](MANUSCRIPT_ALIGNMENT.md) for the two manuscript
sentences that must remain consistent with the released audit boundary and
per-class calculations.

## Citation

Metadata for the manuscript is provided in [`CITATION.cff`](CITATION.cff).
The software is released under the MIT License.

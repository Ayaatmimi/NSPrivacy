# Code and result status

## Preserved artifacts

The Markdown files in `reported_results/` record the manuscript tables. They are fixed reference artifacts.

## Candidate implementation

`dp_pipeline/` implements:

- Poisson-sampled private updates
- complete per-record loss evaluation
- global per-record gradient clipping
- Gaussian noise on the summed clipped gradient
- RDP composition after every update
- projected privacy-budget stopping
- trainable-mask warm-up followed by mask freezing
- separate operating-system-seeded generators for sampling and DP noise

## Current limitation

The public pipeline has not yet completed the full dataset-specific five-seed reproduction study. It must not be described as reproducing every reported number until that study is run and checked.

The original development notebooks used an earlier training path and are not evidence for the final DP guarantee. They are withheld from the main release until their claims, paths, outputs, and data dependencies are reconciled with the final method.

## Privacy boundary

The accountant covers the training updates performed by `dp_pipeline/nsprivacy_dp.py`. Data-dependent preprocessing, private validation, hyperparameter selection, and release of evaluation statistics require their own privacy analysis unless they use public or explicitly unprotected data.

# Implementation notes

The `dp_pipeline/` directory implements the NSPrivacy private-training procedure.

## Training pipeline

The implementation includes:

- Poisson-sampled private updates
- complete per-record loss evaluation
- global per-record gradient clipping
- Gaussian noise on the summed clipped gradient
- RDP composition after every update
- projected privacy-budget stopping
- trainable-mask warm-up followed by mask freezing
- separate operating-system-seeded generators for sampling and DP noise

## Result artifacts

The Markdown files in `reported_results/` contain the values reported in the manuscript.

## Privacy-accounting scope

The RDP accountant records the training updates performed by `dp_pipeline/nsprivacy_dp.py`. Dataset preprocessing, model evaluation, and reporting are maintained as separate stages of the experimental workflow.

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
- fixed expected-batch normalization, including noise-only empty-batch updates
- epoch-level audit records containing only permitted public configuration,
  accountant outputs, and module status

## Result artifacts

The Markdown files in `reported_results/` contain the values reported in the manuscript.

## Privacy-accounting scope

The RDP accountant records every scheduled private update performed by
`dp_pipeline/nsprivacy_dp.py`, including conservatively counted noise-only
updates for empty Poisson batches. Dataset preprocessing, selection across
private runs, model evaluation, and reporting are separate stages of the
experimental workflow and require their own privacy treatment when they use
protected records.

The released audit record excludes private RNG seeds, class-weight values,
raw losses, masks, sensitivities, representations, per-record gradients, and
private mini-batch statistics.

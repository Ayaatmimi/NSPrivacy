# Reported Results on MIMIC-IV

These are the frozen results reported in the IEEE TKDE manuscript. They represent the original experimental records and must not be overwritten by later development or verification runs.

The task is binary in-hospital mortality prediction. Results are means over five seeds. The reported uncertainty values are 95% bootstrap confidence intervals based on 1,000 bootstrap resamples.

| Method            | Accuracy (%) | AUROC |  Epsilon | MIA AUC |
| ----------------- | -----------: | ----: | -------: | ------: |
| Standard MLP      |  91.64 ± 1.2 | 0.923 | Infinity |   0.598 |
| DP-SGD            |  78.55 ± 2.1 | 0.812 |      1.0 |   0.524 |
| DP-SGD (Adaptive) |  79.83 ± 1.9 | 0.821 |      1.0 |   0.522 |
| Auto-S            |  80.14 ± 1.8 | 0.826 |      1.0 |   0.521 |
| DP-RandP          |  81.37 ± 1.7 | 0.834 |      1.0 |   0.520 |
| PATE              |  82.18 ± 1.9 | 0.845 |     <1.0 |   0.519 |
| DPSUR             |  80.92 ± 1.8 | 0.830 |      1.0 |   0.521 |
| NSPrivacy         |  87.27 ± 1.8 | 0.891 |    0.114 |   0.521 |

## Evaluation Setting

* Dataset: MIMIC-IV
* Prediction target: in-hospital mortality
* Number of clinical variables: 29
* Number of runs: 5
* Confidence-interval procedure: 1,000 bootstrap resamples
* MIA AUC values closer to 0.5 indicate attack performance closer to random guessing.
* The NSPrivacy configuration selected on CIC-IDS2017 was applied without additional tuning.

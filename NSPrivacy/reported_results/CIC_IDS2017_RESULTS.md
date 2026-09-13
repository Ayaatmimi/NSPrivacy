# Reported Results on CIC-IDS2017

These are the frozen results reported in the IEEE TKDE manuscript. They represent the original experimental records and must not be overwritten by later development or verification runs.

## Main Results

Results are averaged over five seeds: 42, 123, 456, 789, and 1024. Balanced accuracy is reported as mean ± sample standard deviation. The dagger symbol indicates a statistically significant difference from DP-SGD at the same privacy budget with \(p<0.001\) after Bonferroni correction.

| Category                    | Method            |  Epsilon | log10(Epsilon) | Balanced Accuracy (%) | Macro-F1 | Macro-AUROC | MIA AUC |   PUE |
| --------------------------- | ----------------- | -------: | -------------: | --------------------: | -------: | ----------: | ------: | ----: |
| Non-private                 | Random Forest     | Infinity |            N/A |                 99.86 |    0.971 |       0.999 |   0.651 |   N/A |
| Non-private                 | XGBoost           | Infinity |            N/A |                 94.00 |    0.940 |       0.980 |   0.830 |   N/A |
| Non-private                 | Standard MLP      | Infinity |            N/A |          98.09 ± 0.08 |    0.968 |       0.998 |   0.612 |   N/A |
| DP, epsilon approximately 1 | DP-SGD            |      1.0 |           0.00 |          89.10 ± 0.31 |    0.872 |       0.934 |   0.531 | 0.723 |
| DP, epsilon approximately 1 | DP-SGD (Adaptive) |      1.0 |           0.00 |          90.23 ± 0.28 |    0.885 |       0.942 |   0.529 | 0.741 |
| DP, epsilon approximately 1 | Auto-S            |      1.0 |           0.00 |          90.87 ± 0.26 |    0.891 |       0.945 |   0.528 | 0.748 |
| DP, epsilon approximately 1 | DP-RandP          |      1.0 |           0.00 |          91.54 ± 0.24 |    0.897 |       0.951 |   0.526 | 0.762 |
| DP, epsilon approximately 1 | PATE              |     <1.0 |          <0.00 |          92.45 ± 0.28 |    0.904 |       0.956 |   0.523 | 0.801 |
| DP, epsilon approximately 1 | DPSUR             |      1.0 |           0.00 |          91.18 ± 0.25 |    0.893 |       0.948 |   0.527 | 0.768 |
| DP, epsilon approximately 1 | NSPrivacy         |      1.0 |           0.00 |        97.89 ± 0.09 † |    0.969 |       0.998 |   0.519 | 0.978 |
| DP, epsilon 0.114           | DP-SGD            |    0.114 |          -0.94 |          71.23 ± 0.89 |    0.684 |       0.812 |   0.508 | 0.612 |
| DP, epsilon 0.114           | NSPrivacy         |    0.114 |          -0.94 |        97.32 ± 0.12 † |    0.962 |       0.997 |   0.530 | 0.968 |

## Privacy-Budget Analysis

Balanced accuracy values are percentages averaged over five seeds.

| Method     | Epsilon = 0.05 | Epsilon = 0.114 | Epsilon = 0.5 | Epsilon = 1.0 |
| ---------- | -------------: | --------------: | ------------: | ------------: |
| DP-SGD     |          58.41 |           71.23 |         84.17 |         89.10 |
| NSPrivacy  |          95.18 |           97.32 |         97.71 |         97.89 |
| Difference |         +36.77 |          +26.09 |        +13.54 |         +8.79 |

## Privacy Setting

* Delta: \(10^{-5}\)
* Maximum training epochs: 100
* Number of runs: 5
* Random seeds: 42, 123, 456, 789, and 1024
* Primary metric: balanced accuracy
* MIA AUC values closer to 0.5 indicate attack performance closer to random guessing.

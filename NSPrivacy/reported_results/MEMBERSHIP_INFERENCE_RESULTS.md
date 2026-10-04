# Membership-Inference Results on CIC-IDS2017

The output-based ML-Doctor attack uses a balanced set of training members and
held-out nonmembers. Values closer to 0.5 indicate attack performance closer to
random guessing. `Leakage` is the reported attack advantage.

| Method | MIA AUC | MIA Accuracy | Precision | Leakage |
| --- | ---: | ---: | ---: | ---: |
| Standard MLP | 0.612 | 0.584 | 0.571 | 0.168 |
| DP-SGD (epsilon = 1.0) | 0.531 | 0.519 | 0.512 | 0.038 |
| DP-SGD (Adaptive) | 0.529 | 0.517 | 0.510 | 0.036 |
| Auto-S | 0.528 | 0.516 | 0.509 | 0.035 |
| DP-RandP | 0.526 | 0.514 | 0.508 | 0.033 |
| PATE | 0.523 | 0.512 | 0.508 | 0.026 |
| DPSUR | 0.527 | 0.515 | 0.509 | 0.034 |
| NSPrivacy (epsilon = 0.114) | 0.530 | 0.517 | 0.511 | 0.030 |

This experiment evaluates one output-based attack and complements, but does
not replace, the formal differential-privacy guarantee.

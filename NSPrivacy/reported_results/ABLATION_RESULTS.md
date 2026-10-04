# Ablation Results on CIC-IDS2017

Each row removes one component from NSPrivacy at \(\varepsilon=0.114\).
Representation dispersion (`STD`) is the mean feature-wise standard deviation
of the protected representation.

| Configuration | Balanced Accuracy | MIA AUC | STD |
| --- | ---: | ---: | ---: |
| Full NSPrivacy | 0.973 | 0.530 | 0.042 |
| Without Data Minimization | 0.912 | 0.548 | 0.078 |
| Without Non-Retention | 0.958 | 0.562 | 0.068 |
| Without Structural Design | 0.965 | 0.587 | 0.055 |
| Without Mask Adaptation | 0.941 | 0.535 | 0.048 |

The full framework has the highest balanced accuracy, the MIA AUC closest to
0.5, and the lowest representation dispersion. Component effects are not
interpreted additively because the modules interact during training.

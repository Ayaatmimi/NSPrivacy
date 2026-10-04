# PUE Evaluation at Epsilon 0.114

The reference weights are \((w_U,w_P,w_E)=(0.4,0.4,0.2)\). They are fixed
before test evaluation. `U`, `P`, and `E` denote utility, empirical privacy,
and technical-evidence completeness.

| Method | U | P | E | PUE |
| --- | ---: | ---: | ---: | ---: |
| NSPrivacy | 0.9732 | 0.940 | 1.000 | 0.965 |
| DP-SGD | 0.7123 | 0.984 | 0.000 | 0.679 |

The PUE difference is
\(0.2609w_U-0.044w_P+w_E\). NSPrivacy ranks higher whenever
\(0.2609w_U+w_E>0.044w_P\).

The manuscript evaluates 1,000 positive weight triples drawn from
Dirichlet(1,1,1) and Dirichlet(5,5,2), together with a grid over
\(w_U,w_P\in\{0.2,0.3,0.4,0.5\}\) for valid positive \(w_E\). NSPrivacy ranks
higher in every evaluated configuration. This does not imply superiority under
every possible weighting.

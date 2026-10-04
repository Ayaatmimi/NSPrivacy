# Computational Cost on CIC-IDS2017

Training time and memory are measured per epoch. Inference latency is measured
per batch. All methods use the same hardware, software environment, batch size,
and data split.

| Method | Training (min/epoch) | Memory (GB) | Inference (ms/batch) | Parameters (M) |
| --- | ---: | ---: | ---: | ---: |
| Non-private MLP | 3.2 | 2.1 | 0.8 | 0.060 |
| DP-SGD | 12.4 | 8.3 | 0.8 | 0.060 |
| PATE | 45.6 | 12.5 | 0.8 | 0.663 |
| Auto-S | 12.4 | 8.3 | 0.8 | 0.060 |
| DP-RandP | 18.2 | 10.5 | 0.8 | 0.060 |
| NSPrivacy | 14.1 | 9.1 | 1.1 | 0.064 |

Relative to DP-SGD, NSPrivacy increases training time by 13.7%, memory use by
9.6%, and the parameter count by approximately 0.004 million.

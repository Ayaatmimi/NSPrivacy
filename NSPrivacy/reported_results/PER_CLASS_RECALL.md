# Per-Class Recall on CIC-IDS2017

These are the frozen per-class recall values reported in the IEEE TKDE manuscript. They must not be overwritten by later development or verification runs.

| Class                | NSPrivacy |    DP-SGD | Standard MLP | Class Count |
| -------------------- | --------: | --------: | -----------: | ----------: |
| Benign               |     0.987 |     0.951 |        0.992 |   2,273,097 |
| DoS                  |     0.991 |     0.923 |        0.994 |     252,661 |
| DDoS                 |     0.987 |     0.918 |        0.992 |     128,027 |
| PortScan             |     0.984 |     0.912 |        0.989 |     158,930 |
| Brute Force          |     0.968 |     0.887 |        0.978 |      13,835 |
| Botnet               |     0.952 |     0.856 |        0.969 |       1,966 |
| Web Attack           |     0.941 |     0.845 |        0.962 |       2,180 |
| Infiltration         |     0.934 |     0.823 |        0.956 |          36 |
| **Balanced average** | **0.968** | **0.889** |    **0.979** |         N/A |

NSPrivacy exceeds DP-SGD for every class. The largest recall improvement is 0.111 for Infiltration. The improvements for Botnet and Web Attack are both 0.096.

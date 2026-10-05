# Data Access and Preparation

NSPrivacy uses derived tabular inputs. Raw datasets are not redistributed in this repository.

## CIC-IDS2017

Obtain CIC-IDS2017 from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html).

The manuscript evaluates an eight-class task:

1. Benign
2. DoS
3. DDoS
4. PortScan
5. Brute Force
6. Botnet
7. Web Attack
8. Infiltration

Before running the generic preprocessor, normalize label whitespace and consolidate the original CIC-IDS2017 labels into these eight categories. The eight-class protocol excludes the 11 Heartbleed flows. This yields the class counts reported in the manuscript:

| Class | Count |
| --- | ---: |
| Benign | 2,273,097 |
| DoS | 252,661 |
| DDoS | 128,027 |
| PortScan | 158,930 |
| Brute Force | 13,835 |
| Botnet | 1,966 |
| Web Attack | 2,180 |
| Infiltration | 36 |

The counts total 2,830,732 after excluding Heartbleed. Constant features and exact duplicate rows are removed before the stratified 80/10/10 split. Training-set medians, winsorization thresholds, and standardization parameters are then applied unchanged to validation and test data.

## MIMIC-IV

Obtain MIMIC-IV from [PhysioNet](https://physionet.org/content/mimiciv/) and complete its credentialing and data-use requirements.

The manuscript cohort uses adult patients from their first ICU admission, requires an ICU stay of at least 48 hours, uses measurements from the first 24 hours, and predicts in-hospital mortality from 29 clinical variables. Construct the cohort according to the manuscript protocol before using the generic tabular preprocessor.

## Expected derived CSV

The input to `prepare_tabular.py` must contain:

- one row per example;
- numeric feature columns;
- one label column selected with `--label-column`;
- no identifiers or free-text fields intended only for record linkage.

Example:

```bash
python NSPrivacy/experiments/prepare_tabular.py \
  --csv path/to/derived_table.csv \
  --label-column label \
  --output prepared/data.npz \
  --seed 42
```

The generated NPZ file contains `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, `y_test`, class weights, and preprocessing metadata.

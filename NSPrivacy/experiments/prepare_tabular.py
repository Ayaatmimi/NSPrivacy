"""Leakage-safe preprocessing for tabular classification datasets.

The input is a CSV file with one label column. Duplicate rows and constant
features are removed, then an 80/10/10 stratified split is created. Imputation,
winsorization, and standardization parameters are fitted on the training split
only and applied unchanged to validation and test data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


def _fit_transform(train: np.ndarray, *others: np.ndarray):
    median = np.nanmedian(train, axis=0)
    train = np.where(np.isnan(train), median, train)
    others = tuple(np.where(np.isnan(x), median, x) for x in others)
    lower = np.quantile(train, 0.01, axis=0)
    upper = np.quantile(train, 0.99, axis=0)
    train = np.clip(train, lower, upper)
    others = tuple(np.clip(x, lower, upper) for x in others)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1.0
    transformed = ((train - mean) / scale,) + tuple(
        (x - mean) / scale for x in others
    )
    parameters = {
        "median": median.tolist(),
        "winsor_lower": lower.tolist(),
        "winsor_upper": upper.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }
    return transformed, parameters


def prepare(csv_path: str, label_column: str, output_path: str, seed: int) -> None:
    frame = pd.read_csv(csv_path).drop_duplicates()
    if label_column not in frame:
        raise ValueError(f"label column not found: {label_column}")
    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame.pop(label_column).astype(str))
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    keep = numeric.nunique(dropna=False) > 1
    numeric = numeric.loc[:, keep]
    x = numeric.to_numpy(dtype=np.float64)

    x_train, x_hold, y_train, y_hold = train_test_split(
        x, labels, test_size=0.20, random_state=seed, stratify=labels
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_hold, y_hold, test_size=0.50, random_state=seed, stratify=y_hold
    )
    (x_train, x_val, x_test), parameters = _fit_transform(
        x_train, x_val, x_test
    )
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    metadata = {
        "seed": seed,
        "split": [0.8, 0.1, 0.1],
        "feature_names": numeric.columns.tolist(),
        "label_classes": encoder.classes_.tolist(),
        "class_weights": weights.tolist(),
        "preprocessing": parameters,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        x_train=x_train.astype(np.float32),
        y_train=y_train.astype(np.int64),
        x_val=x_val.astype(np.float32),
        y_val=y_val.astype(np.int64),
        x_test=x_test.astype(np.float32),
        y_test=y_test.astype(np.int64),
        class_weights=weights.astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.csv, args.label_column, args.output, args.seed)


if __name__ == "__main__":
    main()

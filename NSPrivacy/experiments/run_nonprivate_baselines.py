"""Reproducible non-private reference baselines for prepared NPZ data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier


def metrics(model, x_test, y_test) -> dict:
    prediction = model.predict(x_test)
    probability = model.predict_proba(x_test)
    if probability.shape[1] == 2:
        auc = roc_auc_score(y_test, probability[:, 1])
    else:
        auc = roc_auc_score(y_test, probability, multi_class="ovr", average="macro")
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "macro_f1": float(f1_score(y_test, prediction, average="macro")),
        "macro_auroc": float(auc),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    arrays = np.load(args.data, allow_pickle=False)
    x_train, y_train = arrays["x_train"], arrays["y_train"]
    x_test, y_test = arrays["x_test"], arrays["y_test"]
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1,
            random_state=args.seed
        ),
        "gradient_boosting_reference": HistGradientBoostingClassifier(
            random_state=args.seed
        ),
        "standard_mlp_reference": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), max_iter=100,
            random_state=args.seed
        ),
    }
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        results[name] = metrics(model, x_test, y_test)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

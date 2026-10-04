"""Evaluation utilities used by the NSPrivacy experiment protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


def output_membership_attack(
    member_probabilities: np.ndarray,
    nonmember_probabilities: np.ndarray,
    seed: int = 42,
) -> dict:
    """Run a balanced output-based membership classifier.

    The function accepts target-model probability vectors. It uses a logistic
    attack classifier and reports the same metrics as the manuscript. This is
    an executable output-attack protocol, not a claim of byte-level identity
    with the external ML-Doctor package.
    """
    count = min(len(member_probabilities), len(nonmember_probabilities))
    member = np.asarray(member_probabilities[:count], dtype=np.float64)
    nonmember = np.asarray(nonmember_probabilities[:count], dtype=np.float64)
    x = np.concatenate([member, nonmember])
    y = np.concatenate([np.ones(count), np.zeros(count)]).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=seed, stratify=y
    )
    attack = LogisticRegression(max_iter=2000, random_state=seed)
    attack.fit(x_train, y_train)
    score = attack.predict_proba(x_test)[:, 1]
    prediction = (score >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, score))
    accuracy = float(accuracy_score(y_test, prediction))
    precision = float(precision_score(y_test, prediction, zero_division=0))
    return {
        "mia_auc": auc,
        "mia_accuracy": accuracy,
        "mia_precision": precision,
        "leakage": abs(2.0 * accuracy - 1.0),
    }


def pue(utility: float, mia_auc: float, evidence: float, weights=(0.4, 0.4, 0.2)) -> dict:
    privacy = 1.0 - 2.0 * abs(float(mia_auc) - 0.5)
    score = sum(w * v for w, v in zip(weights, (utility, privacy, evidence)))
    return {
        "utility": float(utility),
        "empirical_privacy": float(privacy),
        "evidence": float(evidence),
        "weights": list(weights),
        "pue": float(score),
    }


def bootstrap_interval(values: np.ndarray, seed: int = 42, draws: int = 1000):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.asarray(
        [generator.choice(values, size=len(values), replace=True).mean() for _ in range(draws)]
    )
    return np.quantile(means, [0.025, 0.975]).tolist()


def paired_test(reference: np.ndarray, candidate: np.ndarray, comparisons: int = 1) -> dict:
    statistic, p_value = ttest_rel(candidate, reference)
    return {
        "t_statistic": float(statistic),
        "p_value": float(p_value),
        "bonferroni_p": float(min(1.0, p_value * comparisons)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member-probabilities", required=True)
    parser.add_argument("--nonmember-probabilities", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = output_membership_attack(
        np.load(args.member_probabilities),
        np.load(args.nonmember_probabilities),
        args.seed,
    )
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

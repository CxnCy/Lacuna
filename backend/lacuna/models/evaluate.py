"""Evaluation metrics for Lacuna temporal backtesting."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def precision_at_k(
    y_true: Iterable[int],
    y_score: Iterable[float],
    k: int,
) -> float:
    """Return the fraction of positives among the top-k scored examples."""
    y_true = np.asarray(list(y_true))
    y_score = np.asarray(list(y_score))

    if len(y_true) == 0:
        return 0.0

    k = min(k, len(y_true))

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    top_indices = np.argsort(y_score)[::-1][:k]
    return float(np.mean(y_true[top_indices]))


def evaluate_predictions(
    y_true,
    y_pred,
    y_score,
    precision_ks=(10, 50, 100),
) -> Dict[str, float]:
    """Calculate classification and ranking metrics."""
    metrics = {
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "pr_auc": float(
            average_precision_score(y_true, y_score)
        ),
    }

    # ROC-AUC is undefined when the test set contains only one class.
    if len(set(y_true)) > 1:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, y_score)
        )
    else:
        metrics["roc_auc"] = float("nan")

    for k in precision_ks:
        metrics[f"precision_at_{k}"] = precision_at_k(
            y_true,
            y_score,
            k,
        )

    return metrics

"""Probability calibration diagnostics for Lacuna models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def get_prediction_probabilities(
    model,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return positive-class prediction probabilities."""
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Calibration analysis requires a model "
            "that exposes predict_proba."
        )

    probabilities = model.predict_proba(X)

    if probabilities.ndim != 2:
        raise ValueError(
            "predict_proba must return a 2D array."
        )

    if probabilities.shape[1] != 2:
        raise ValueError(
            "Lacuna calibration currently expects "
            "binary classification probabilities."
        )

    positive_probabilities = probabilities[:, 1]

    if np.any(
        (positive_probabilities < 0)
        | (positive_probabilities > 1)
    ):
        raise ValueError(
            "Predicted probabilities must lie "
            "between 0 and 1."
        )

    return positive_probabilities


def calculate_brier_score(
    y_true,
    y_probability,
) -> float:
    """Calculate the Brier score for binary predictions."""
    y_true = np.asarray(y_true)
    y_probability = np.asarray(
        y_probability
    )

    if len(y_true) != len(
        y_probability
    ):
        raise ValueError(
            "y_true and y_probability must "
            "have equal length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Calibration evaluation requires "
            "at least one observation."
        )

    return float(
        brier_score_loss(
            y_true,
            y_probability,
        )
    )


def build_calibration_table(
    y_true,
    y_probability,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Build a reliability table from calibration bins."""
    if n_bins < 2:
        raise ValueError(
            "n_bins must be at least 2."
        )

    if strategy not in {
        "uniform",
        "quantile",
    }:
        raise ValueError(
            "strategy must be 'uniform' or 'quantile'."
        )

    y_true = np.asarray(
        y_true
    ).astype(int)

    y_probability = np.asarray(
        y_probability,
        dtype=float,
    )

    if len(y_true) != len(
        y_probability
    ):
        raise ValueError(
            "y_true and y_probability must "
            "have equal length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Calibration evaluation requires "
            "at least one observation."
        )

    fraction_positive, mean_predicted = (
        calibration_curve(
            y_true,
            y_probability,
            n_bins=n_bins,
            strategy=strategy,
        )
    )

    rows = []

    if strategy == "quantile":
        try:
            bin_ids = pd.qcut(
                y_probability,
                q=n_bins,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            bin_ids = np.zeros(
                len(y_probability),
                dtype=int,
            )
    else:
        bin_edges = np.linspace(
            0.0,
            1.0,
            n_bins + 1,
        )

        bin_ids = np.digitize(
            y_probability,
            bin_edges[1:-1],
            right=True,
        )

    bin_ids = np.asarray(
        bin_ids
    )

    unique_bins = sorted(
        pd.Series(
            bin_ids
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    for output_index, bin_id in enumerate(
        unique_bins
    ):
        mask = (
            bin_ids == bin_id
        )

        if not mask.any():
            continue

        bin_probabilities = (
            y_probability[
                mask
            ]
        )

        bin_labels = (
            y_true[
                mask
            ]
        )

        rows.append(
            {
                "bin": output_index + 1,
                "count": int(
                    mask.sum()
                ),
                "mean_predicted_probability": float(
                    bin_probabilities.mean()
                ),
                "observed_positive_rate": float(
                    bin_labels.mean()
                ),
                "min_predicted_probability": float(
                    bin_probabilities.min()
                ),
                "max_predicted_probability": float(
                    bin_probabilities.max()
                ),
            }
        )

    table = pd.DataFrame(
        rows
    )

    if table.empty:
        return table

    table[
        "calibration_gap"
    ] = (
        table[
            "mean_predicted_probability"
        ]
        - table[
            "observed_positive_rate"
        ]
    )

    return table


def calculate_expected_calibration_error(
    calibration_table: pd.DataFrame,
) -> float:
    """Calculate weighted expected calibration error."""
    required = {
        "count",
        "mean_predicted_probability",
        "observed_positive_rate",
    }

    missing = required.difference(
        calibration_table.columns
    )

    if missing:
        raise ValueError(
            "Calibration table is missing "
            f"columns: {sorted(missing)}"
        )

    if calibration_table.empty:
        raise ValueError(
            "Calibration table cannot be empty."
        )

    total_count = calibration_table[
        "count"
    ].sum()

    if total_count <= 0:
        raise ValueError(
            "Calibration table must contain "
            "positive observation counts."
        )

    absolute_gaps = (
        calibration_table[
            "mean_predicted_probability"
        ]
        - calibration_table[
            "observed_positive_rate"
        ]
    ).abs()

    weights = (
        calibration_table[
            "count"
        ]
        / total_count
    )

    return float(
        (
            weights
            * absolute_gaps
        ).sum()
    )


def summarize_calibration(
    y_true,
    y_probability,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> dict:
    """Summarize model calibration diagnostics."""
    y_true = np.asarray(
        y_true
    ).astype(int)

    y_probability = np.asarray(
        y_probability,
        dtype=float,
    )

    table = build_calibration_table(
        y_true,
        y_probability,
        n_bins=n_bins,
        strategy=strategy,
    )

    brier_score = calculate_brier_score(
        y_true,
        y_probability,
    )

    ece = (
        calculate_expected_calibration_error(
            table
        )
    )

    observed_prevalence = float(
        y_true.mean()
    )

    mean_probability = float(
        y_probability.mean()
    )

    return {
        "sample_count": int(
            len(y_true)
        ),
        "positive_count": int(
            y_true.sum()
        ),
        "observed_positive_rate": (
            observed_prevalence
        ),
        "mean_predicted_probability": (
            mean_probability
        ),
        "probability_prevalence_gap": (
            mean_probability
            - observed_prevalence
        ),
        "brier_score": (
            brier_score
        ),
        "expected_calibration_error": (
            ece
        ),
        "minimum_predicted_probability": float(
            y_probability.min()
        ),
        "maximum_predicted_probability": float(
            y_probability.max()
        ),
    }
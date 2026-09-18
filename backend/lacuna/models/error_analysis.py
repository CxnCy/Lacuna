"""Error analysis utilities for Lacuna temporal model evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lacuna.models.train import FEATURE_COLUMNS


METADATA_COLUMNS = [
    "topic_a",
    "topic_b",
    "cutoff_year",
    "prediction_window_start",
    "prediction_window_end",
]


def build_error_analysis_table(
    model,
    dataset: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Build row-level prediction and error analysis records.

    Only leakage-safe historical features are passed to the model.
    Metadata and future-derived fields are retained only for analysis.
    """
    required_columns = FEATURE_COLUMNS + ["label"]

    missing_columns = [
        column for column in required_columns if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    X = dataset[FEATURE_COLUMNS]
    y_true = dataset["label"].astype(int)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X)
        scores = 1.0 / (1.0 + np.exp(-raw_scores))
    else:
        raise ValueError(
            "Model must expose predict_proba or decision_function."
        )

    y_pred = (scores >= threshold).astype(int)

    result = dataset[
        [
            column
            for column in METADATA_COLUMNS
            if column in dataset.columns
        ]
        + FEATURE_COLUMNS
    ].copy()

    result["true_label"] = y_true.to_numpy()
    result["predicted_label"] = y_pred
    result["prediction_score"] = scores

    result["error_type"] = np.select(
        [
            (result["true_label"] == 1)
            & (result["predicted_label"] == 1),
            (result["true_label"] == 0)
            & (result["predicted_label"] == 1),
            (result["true_label"] == 1)
            & (result["predicted_label"] == 0),
            (result["true_label"] == 0)
            & (result["predicted_label"] == 0),
        ],
        [
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
        ],
        default="unknown",
    )

    result["rank"] = (
        result["prediction_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    return result.sort_values(
        "prediction_score",
        ascending=False,
    ).reset_index(drop=True)


def get_high_confidence_errors(
    analysis_table: pd.DataFrame,
    *,
    error_type: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return the highest-priority examples for one error category."""
    valid_types = {
        "false_positive",
        "false_negative",
        "true_positive",
        "true_negative",
    }

    if error_type not in valid_types:
        raise ValueError(
            f"error_type must be one of {sorted(valid_types)}"
        )

    subset = analysis_table[
        analysis_table["error_type"] == error_type
    ].copy()

    if error_type == "false_negative":
        subset = subset.sort_values(
            "prediction_score",
            ascending=True,
        )
    else:
        subset = subset.sort_values(
            "prediction_score",
            ascending=False,
        )

    return subset.head(top_n).reset_index(drop=True)


def summarize_feature_profiles(
    analysis_table: pd.DataFrame,
) -> pd.DataFrame:
    """Compare mean and median feature profiles across prediction outcomes."""
    required_columns = {"error_type", *FEATURE_COLUMNS}

    missing_columns = required_columns.difference(
        analysis_table.columns
    )

    if missing_columns:
        raise ValueError(
            f"Analysis table is missing columns: {sorted(missing_columns)}"
        )

    summary = (
        analysis_table.groupby("error_type")[FEATURE_COLUMNS]
        .agg(["mean", "median"])
    )

    summary.columns = [
        f"{feature}_{statistic}"
        for feature, statistic in summary.columns
    ]

    return summary.reset_index()
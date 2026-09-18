"""Model-selection utilities for Lacuna Step 7.6."""

from __future__ import annotations

from typing import Any

import pandas as pd


REQUIRED_METRIC_COLUMNS = {
    "test_year",
    "model_name",
    "pr_auc",
    "roc_auc",
    "precision",
    "recall",
    "f1",
}

REQUIRED_CALIBRATION_COLUMNS = {
    "test_year",
    "model_name",
    "brier_score",
    "expected_calibration_error",
    "observed_positive_rate",
    "mean_predicted_probability",
}


def validate_metrics_table(
    metrics: pd.DataFrame,
) -> None:
    """Validate temporal model evaluation metrics."""
    missing = REQUIRED_METRIC_COLUMNS.difference(
        metrics.columns
    )

    if missing:
        raise ValueError(
            "Metrics table is missing required columns: "
            f"{sorted(missing)}"
        )


def validate_calibration_table(
    calibration: pd.DataFrame,
) -> None:
    """Validate model calibration summary."""
    missing = REQUIRED_CALIBRATION_COLUMNS.difference(
        calibration.columns
    )

    if missing:
        raise ValueError(
            "Calibration table is missing required columns: "
            f"{sorted(missing)}"
        )


def build_model_selection_table(
    metrics: pd.DataFrame,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    """Combine temporal performance and calibration evidence."""
    validate_metrics_table(
        metrics
    )

    validate_calibration_table(
        calibration
    )

    selection = metrics.merge(
        calibration[
            [
                "test_year",
                "model_name",
                "brier_score",
                "expected_calibration_error",
                "observed_positive_rate",
                "mean_predicted_probability",
                "probability_prevalence_gap",
            ]
        ],
        on=[
            "test_year",
            "model_name",
        ],
        how="left",
        validate="one_to_one",
    )

    if selection[
        "brier_score"
    ].isna().any():
        raise ValueError(
            "Calibration results are missing for "
            "one or more evaluated model/backtest pairs."
        )

    selection[
        "absolute_prevalence_gap"
    ] = selection[
        "probability_prevalence_gap"
    ].abs()

    return selection.sort_values(
        [
            "test_year",
            "model_name",
        ]
    ).reset_index(
        drop=True
    )


def summarize_temporal_stability(
    selection_table: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize model performance across temporal backtests."""
    rows: list[dict[str, Any]] = []

    for model_name, group in selection_table.groupby(
        "model_name"
    ):
        group = group.sort_values(
            "test_year"
        )

        row = {
            "model_name": model_name,
            "backtest_count": len(group),
            "mean_pr_auc": float(
                group["pr_auc"].mean()
            ),
            "min_pr_auc": float(
                group["pr_auc"].min()
            ),
            "mean_roc_auc": float(
                group["roc_auc"].mean()
            ),
            "min_roc_auc": float(
                group["roc_auc"].min()
            ),
            "mean_precision": float(
                group["precision"].mean()
            ),
            "mean_recall": float(
                group["recall"].mean()
            ),
            "mean_f1": float(
                group["f1"].mean()
            ),
            "mean_brier_score": float(
                group["brier_score"].mean()
            ),
            "mean_ece": float(
                group[
                    "expected_calibration_error"
                ].mean()
            ),
            "mean_absolute_prevalence_gap": float(
                group[
                    "absolute_prevalence_gap"
                ].mean()
            ),
            "pr_auc_change": (
                float(
                    group.iloc[-1]["pr_auc"]
                    - group.iloc[0]["pr_auc"]
                )
                if len(group) > 1
                else 0.0
            ),
            "roc_auc_change": (
                float(
                    group.iloc[-1]["roc_auc"]
                    - group.iloc[0]["roc_auc"]
                )
                if len(group) > 1
                else 0.0
            ),
        }

        for k in [
            10,
            50,
            100,
        ]:
            column = f"precision_at_{k}"

            if column in group.columns:
                row[
                    f"mean_precision_at_{k}"
                ] = float(
                    group[column].mean()
                )

                row[
                    f"max_precision_at_{k}"
                ] = float(
                    group[column].max()
                )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "model_name"
    ).reset_index(
        drop=True
    )


def build_model_assessment(
    stability: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build evidence-based model assessments.

    This does not automatically declare a production model.
    """
    rows = []

    for _, row in stability.iterrows():
        model_name = row[
            "model_name"
        ]

        strengths = []
        limitations = []

        if row[
            "mean_roc_auc"
        ] >= 0.6:
            strengths.append(
                "above-random mean ROC-AUC"
            )

        if row[
            "mean_recall"
        ] >= 0.25:
            strengths.append(
                "comparatively strong recall"
            )

        if row[
            "mean_ece"
        ] <= 0.10:
            strengths.append(
                "comparatively lower calibration error"
            )

        if row[
            "mean_pr_auc"
        ] < 0.05:
            limitations.append(
                "very low PR-AUC under severe class imbalance"
            )

        if row[
            "mean_precision"
        ] < 0.05:
            limitations.append(
                "very low classification precision"
            )

        if row[
            "mean_ece"
        ] > 0.20:
            limitations.append(
                "severe probability miscalibration"
            )

        if row[
            "pr_auc_change"
        ] < 0:
            limitations.append(
                "PR-AUC deteriorates in the later backtest"
            )

        if row[
            "roc_auc_change"
        ] < 0:
            limitations.append(
                "ROC-AUC deteriorates in the later backtest"
            )

        rows.append(
            {
                "model_name": model_name,
                "strengths": (
                    "; ".join(
                        strengths
                    )
                    if strengths
                    else "none identified by automatic criteria"
                ),
                "limitations": (
                    "; ".join(
                        limitations
                    )
                    if limitations
                    else "none identified by automatic criteria"
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def recommend_model_role(
    stability: pd.DataFrame,
) -> dict[str, Any]:
    """
    Determine the most defensible current model role.

    Lacuna prioritizes ranking candidate connections rather than
    treating raw probabilities as literal future-event probabilities.
    """
    if stability.empty:
        raise ValueError(
            "Temporal stability table cannot be empty."
        )

    ranked = stability.sort_values(
        [
            "mean_pr_auc",
            "mean_roc_auc",
        ],
        ascending=[
            False,
            False,
        ],
    )

    best = ranked.iloc[
        0
    ]

    model_name = best[
        "model_name"
    ]

    severe_limitations = []

    if best[
        "mean_pr_auc"
    ] < 0.05:
        severe_limitations.append(
            "low temporal PR-AUC"
        )

    if best[
        "mean_precision"
    ] < 0.05:
        severe_limitations.append(
            "low classification precision"
        )

    if best[
        "mean_ece"
    ] > 0.20:
        severe_limitations.append(
            "poor probability calibration"
        )

    if severe_limitations:
        decision = (
            "ranking_research_baseline"
        )

        probability_interpretation = (
            "not_calibrated_probability"
        )

        atlas_guidance = (
            "Use model output only as a relative ranking score "
            "for exploratory candidate lacunae. Do not present "
            "scores as literal probabilities of future emergence "
            "or as claims of scientific discovery."
        )
    else:
        decision = (
            "candidate_ranking_model"
        )

        probability_interpretation = (
            "requires_context"
        )

        atlas_guidance = (
            "Use for ranked candidate exploration with explicit "
            "model uncertainty and temporal-validation context."
        )

    return {
        "selected_model": model_name,
        "selected_role": decision,
        "probability_interpretation": (
            probability_interpretation
        ),
        "severe_limitations": (
            "; ".join(
                severe_limitations
            )
            if severe_limitations
            else "none"
        ),
        "atlas_guidance": (
            atlas_guidance
        ),
    }
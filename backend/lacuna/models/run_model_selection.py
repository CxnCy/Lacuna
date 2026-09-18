"""Run Lacuna Step 7.6 model selection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lacuna.models.model_selection import (
    build_model_assessment,
    build_model_selection_table,
    recommend_model_role,
    summarize_temporal_stability,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]

CALIBRATION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "analysis"
    / "calibration"
    / "calibration_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "analysis"
    / "model_selection"
)


BACKTEST_METRICS = [
    {
        "test_year": 2015,
        "model_name": (
            "logistic_regression"
        ),
        "pr_auc": 0.0307,
        "roc_auc": 0.7238,
        "precision": 0.0208,
        "recall": 0.6977,
        "f1": 0.0403,
        "precision_at_10": 0.0000,
        "precision_at_50": 0.0600,
        "precision_at_100": 0.0600,
    },
    {
        "test_year": 2015,
        "model_name": (
            "random_forest"
        ),
        "pr_auc": 0.0165,
        "roc_auc": 0.6259,
        "precision": 0.0209,
        "recall": 0.1163,
        "f1": 0.0355,
        "precision_at_10": 0.0000,
        "precision_at_50": 0.0000,
        "precision_at_100": 0.0000,
    },
    {
        "test_year": 2015,
        "model_name": (
            "hist_gradient_boosting"
        ),
        "pr_auc": 0.0206,
        "roc_auc": 0.5607,
        "precision": 0.0196,
        "recall": 0.1163,
        "f1": 0.0336,
        "precision_at_10": 0.0000,
        "precision_at_50": 0.0000,
        "precision_at_100": 0.0000,
    },
    {
        "test_year": 2020,
        "model_name": (
            "logistic_regression"
        ),
        "pr_auc": 0.0069,
        "roc_auc": 0.6443,
        "precision": 0.0041,
        "recall": 0.4615,
        "f1": 0.0081,
        "precision_at_10": 0.0000,
        "precision_at_50": 0.0000,
        "precision_at_100": 0.0200,
    },
    {
        "test_year": 2020,
        "model_name": (
            "random_forest"
        ),
        "pr_auc": 0.0039,
        "roc_auc": 0.6411,
        "precision": 0.0000,
        "recall": 0.0000,
        "f1": 0.0000,
        "precision_at_10": 0.0000,
        "precision_at_50": 0.0000,
        "precision_at_100": 0.0000,
    },
    {
        "test_year": 2020,
        "model_name": (
            "hist_gradient_boosting"
        ),
        "pr_auc": 0.0040,
        "roc_auc": 0.4798,
        "precision": 0.0092,
        "recall": 0.0769,
        "f1": 0.0164,
        "precision_at_10": 0.0000,
        "precision_at_50": 0.0000,
        "precision_at_100": 0.0000,
    },
]


def save_dataframe(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    """Save model-selection artifact."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_DIR
        / filename,
        index=False,
    )


def run_model_selection() -> None:
    """Execute Step 7.6 model-selection analysis."""
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(
            "Calibration summary not found: "
            f"{CALIBRATION_PATH}"
        )

    calibration = pd.read_csv(
        CALIBRATION_PATH
    )

    metrics = pd.DataFrame(
        BACKTEST_METRICS
    )

    selection = (
        build_model_selection_table(
            metrics,
            calibration,
        )
    )

    stability = (
        summarize_temporal_stability(
            selection
        )
    )

    assessment = (
        build_model_assessment(
            stability
        )
    )

    recommendation = (
        recommend_model_role(
            stability
        )
    )

    recommendation_df = (
        pd.DataFrame(
            [
                recommendation
            ]
        )
    )

    save_dataframe(
        selection,
        "temporal_model_comparison.csv",
    )

    save_dataframe(
        stability,
        "temporal_stability_summary.csv",
    )

    save_dataframe(
        assessment,
        "model_assessment.csv",
    )

    save_dataframe(
        recommendation_df,
        "model_recommendation.csv",
    )

    print(
        "=" * 72
    )
    print(
        "LACUNA — STEP 7.6 MODEL SELECTION"
    )
    print(
        "=" * 72
    )

    print()
    print(
        "TEMPORAL MODEL SUMMARY"
    )
    print(
        "-" * 72
    )

    for _, row in stability.iterrows():
        print(
            f"{row['model_name']}: "
            f"mean PR-AUC={row['mean_pr_auc']:.4f} | "
            f"mean ROC-AUC={row['mean_roc_auc']:.4f} | "
            f"mean precision={row['mean_precision']:.4f} | "
            f"mean recall={row['mean_recall']:.4f} | "
            f"mean ECE={row['mean_ece']:.4f}"
        )

    print()
    print(
        "CURRENT DECISION"
    )
    print(
        "-" * 72
    )

    print(
        "Selected model: "
        f"{recommendation['selected_model']}"
    )

    print(
        "Role: "
        f"{recommendation['selected_role']}"
    )

    print(
        "Probability interpretation: "
        f"{recommendation['probability_interpretation']}"
    )

    print(
        "Limitations: "
        f"{recommendation['severe_limitations']}"
    )

    print()
    print(
        "Atlas guidance:"
    )

    print(
        recommendation[
            "atlas_guidance"
        ]
    )

    print()
    print(
        "=" * 72
    )
    print(
        "STEP 7.6 MODEL SELECTION COMPLETE"
    )
    print(
        f"Artifacts: {OUTPUT_DIR}"
    )
    print(
        "=" * 72
    )


if __name__ == "__main__":
    run_model_selection()
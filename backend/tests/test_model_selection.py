import pandas as pd
import pytest

from lacuna.models.model_selection import (
    build_model_assessment,
    build_model_selection_table,
    recommend_model_role,
    summarize_temporal_stability,
)


def make_metrics():
    return pd.DataFrame(
        [
            {
                "test_year": 2015,
                "model_name": "model_a",
                "pr_auc": 0.10,
                "roc_auc": 0.70,
                "precision": 0.05,
                "recall": 0.50,
                "f1": 0.09,
                "precision_at_10": 0.10,
            },
            {
                "test_year": 2020,
                "model_name": "model_a",
                "pr_auc": 0.05,
                "roc_auc": 0.65,
                "precision": 0.02,
                "recall": 0.40,
                "f1": 0.04,
                "precision_at_10": 0.00,
            },
            {
                "test_year": 2015,
                "model_name": "model_b",
                "pr_auc": 0.04,
                "roc_auc": 0.60,
                "precision": 0.03,
                "recall": 0.10,
                "f1": 0.05,
                "precision_at_10": 0.00,
            },
            {
                "test_year": 2020,
                "model_name": "model_b",
                "pr_auc": 0.02,
                "roc_auc": 0.55,
                "precision": 0.01,
                "recall": 0.05,
                "f1": 0.02,
                "precision_at_10": 0.00,
            },
        ]
    )


def make_calibration():
    return pd.DataFrame(
        [
            {
                "test_year": 2015,
                "model_name": "model_a",
                "brier_score": 0.20,
                "expected_calibration_error": 0.30,
                "observed_positive_rate": 0.01,
                "mean_predicted_probability": 0.40,
                "probability_prevalence_gap": 0.39,
            },
            {
                "test_year": 2020,
                "model_name": "model_a",
                "brier_score": 0.18,
                "expected_calibration_error": 0.25,
                "observed_positive_rate": 0.005,
                "mean_predicted_probability": 0.35,
                "probability_prevalence_gap": 0.345,
            },
            {
                "test_year": 2015,
                "model_name": "model_b",
                "brier_score": 0.05,
                "expected_calibration_error": 0.08,
                "observed_positive_rate": 0.01,
                "mean_predicted_probability": 0.07,
                "probability_prevalence_gap": 0.06,
            },
            {
                "test_year": 2020,
                "model_name": "model_b",
                "brier_score": 0.03,
                "expected_calibration_error": 0.05,
                "observed_positive_rate": 0.005,
                "mean_predicted_probability": 0.04,
                "probability_prevalence_gap": 0.035,
            },
        ]
    )


def test_build_model_selection_table():
    result = build_model_selection_table(
        make_metrics(),
        make_calibration(),
    )

    assert len(result) == 4

    assert (
        "brier_score"
        in result.columns
    )

    assert (
        "absolute_prevalence_gap"
        in result.columns
    )


def test_temporal_stability_summary():
    selection = build_model_selection_table(
        make_metrics(),
        make_calibration(),
    )

    result = summarize_temporal_stability(
        selection
    )

    assert len(result) == 2

    model_a = result.loc[
        result["model_name"]
        == "model_a"
    ].iloc[0]

    assert (
        model_a["backtest_count"]
        == 2
    )

    assert (
        model_a["pr_auc_change"]
        < 0
    )


def test_model_assessment():
    selection = build_model_selection_table(
        make_metrics(),
        make_calibration(),
    )

    stability = summarize_temporal_stability(
        selection
    )

    result = build_model_assessment(
        stability
    )

    assert len(result) == 2

    assert (
        "strengths"
        in result.columns
    )

    assert (
        "limitations"
        in result.columns
    )


def test_recommend_model_role():
    selection = build_model_selection_table(
        make_metrics(),
        make_calibration(),
    )

    stability = summarize_temporal_stability(
        selection
    )

    result = recommend_model_role(
        stability
    )

    assert (
        result[
            "selected_model"
        ]
        == "model_a"
    )

    assert (
        result[
            "selected_role"
        ]
        == "ranking_research_baseline"
    )

    assert (
        result[
            "probability_interpretation"
        ]
        == "not_calibrated_probability"
    )

    assert (
        "poor probability calibration"
        in result[
            "severe_limitations"
        ]
    )


def test_weak_model_is_research_baseline():
    stability = pd.DataFrame(
        [
            {
                "model_name": "weak_model",
                "mean_pr_auc": 0.01,
                "mean_roc_auc": 0.65,
                "mean_precision": 0.01,
                "mean_recall": 0.50,
                "mean_f1": 0.02,
                "mean_brier_score": 0.20,
                "mean_ece": 0.40,
                "mean_absolute_prevalence_gap": 0.30,
                "pr_auc_change": -0.005,
                "roc_auc_change": -0.05,
                "backtest_count": 2,
            }
        ]
    )

    result = recommend_model_role(
        stability
    )

    assert (
        result[
            "selected_role"
        ]
        == "ranking_research_baseline"
    )

    assert (
        result[
            "probability_interpretation"
        ]
        == "not_calibrated_probability"
    )


def test_missing_metrics_rejected():
    metrics = make_metrics().drop(
        columns=[
            "pr_auc"
        ]
    )

    with pytest.raises(
        ValueError
    ):
        build_model_selection_table(
            metrics,
            make_calibration(),
        )
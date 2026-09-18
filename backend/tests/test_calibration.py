import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import (
    LogisticRegression,
)

from lacuna.models.calibration import (
    build_calibration_table,
    calculate_brier_score,
    calculate_expected_calibration_error,
    get_prediction_probabilities,
    summarize_calibration,
)


def make_dataset():
    X = pd.DataFrame(
        {
            "x1": [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
            ],
            "x2": [
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
            ],
        }
    )

    y = pd.Series(
        [
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            1,
            1,
            1,
        ]
    )

    return X, y


def test_get_prediction_probabilities():
    X, y = make_dataset()

    model = LogisticRegression()
    model.fit(
        X,
        y,
    )

    probabilities = (
        get_prediction_probabilities(
            model,
            X,
        )
    )

    assert len(
        probabilities
    ) == len(
        X
    )

    assert np.all(
        probabilities >= 0
    )

    assert np.all(
        probabilities <= 1
    )


def test_calculate_brier_score():
    y_true = np.array(
        [
            0,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.1,
            0.9,
        ]
    )

    score = calculate_brier_score(
        y_true,
        probabilities,
    )

    assert score == pytest.approx(
        0.01
    )


def test_build_calibration_table():
    y_true = np.array(
        [
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
        ]
    )

    probabilities = np.array(
        [
            0.05,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
        ]
    )

    result = build_calibration_table(
        y_true,
        probabilities,
        n_bins=5,
        strategy="quantile",
    )

    assert not result.empty

    assert (
        "mean_predicted_probability"
        in result.columns
    )

    assert (
        "observed_positive_rate"
        in result.columns
    )

    assert (
        "calibration_gap"
        in result.columns
    )

    assert (
        result["count"].sum()
        == len(y_true)
    )


def test_expected_calibration_error():
    table = pd.DataFrame(
        {
            "count": [
                50,
                50,
            ],
            "mean_predicted_probability": [
                0.1,
                0.8,
            ],
            "observed_positive_rate": [
                0.2,
                0.6,
            ],
        }
    )

    result = (
        calculate_expected_calibration_error(
            table
        )
    )

    assert result == pytest.approx(
        0.15
    )


def test_summarize_calibration():
    y_true = np.array(
        [
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
        ]
    )

    probabilities = np.array(
        [
            0.05,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
        ]
    )

    result = summarize_calibration(
        y_true,
        probabilities,
        n_bins=5,
    )

    assert (
        result["sample_count"]
        == 10
    )

    assert (
        result["positive_count"]
        == 5
    )

    assert (
        "brier_score"
        in result
    )

    assert (
        "expected_calibration_error"
        in result
    )

    assert (
        "probability_prevalence_gap"
        in result
    )


def test_invalid_calibration_strategy():
    y_true = np.array(
        [
            0,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.2,
            0.8,
        ]
    )

    with pytest.raises(
        ValueError
    ):
        build_calibration_table(
            y_true,
            probabilities,
            strategy="invalid",
        )
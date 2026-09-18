import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from lacuna.models.error_analysis import (
    build_error_analysis_table,
    get_high_confidence_errors,
    summarize_feature_profiles,
)
from lacuna.models.train import FEATURE_COLUMNS


def make_dataset():
    rows = []

    for i in range(10):
        row = {
            "topic_a": f"topic_a_{i}",
            "topic_b": f"topic_b_{i}",
            "cutoff_year": 2015,
            "prediction_window_start": 2016,
            "prediction_window_end": 2020,
            "label": i % 2,
        }

        for j, feature in enumerate(FEATURE_COLUMNS):
            row[feature] = float(i + j)

        rows.append(row)

    return pd.DataFrame(rows)


def test_build_error_analysis_table():
    dataset = make_dataset()

    X = dataset[FEATURE_COLUMNS]
    y = dataset["label"]

    model = LogisticRegression()
    model.fit(X, y)

    result = build_error_analysis_table(
        model,
        dataset,
    )

    assert len(result) == len(dataset)
    assert "prediction_score" in result.columns
    assert "predicted_label" in result.columns
    assert "true_label" in result.columns
    assert "error_type" in result.columns
    assert "rank" in result.columns

    assert set(result["error_type"]).issubset(
        {
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
        }
    )

    assert sorted(result["rank"].tolist()) == list(
        range(1, len(dataset) + 1)
    )


def test_error_analysis_preserves_metadata():
    dataset = make_dataset()

    model = LogisticRegression()
    model.fit(
        dataset[FEATURE_COLUMNS],
        dataset["label"],
    )

    result = build_error_analysis_table(
        model,
        dataset,
    )

    assert "topic_a" in result.columns
    assert "topic_b" in result.columns
    assert "cutoff_year" in result.columns
    assert "prediction_window_start" in result.columns
    assert "prediction_window_end" in result.columns


def test_future_information_is_not_exposed_as_feature():
    dataset = make_dataset()
    dataset["future_co_publication_count"] = 99

    model = LogisticRegression()
    model.fit(
        dataset[FEATURE_COLUMNS],
        dataset["label"],
    )

    result = build_error_analysis_table(
        model,
        dataset,
    )

    assert "future_co_publication_count" not in result.columns


def test_get_high_confidence_errors():
    dataset = make_dataset()

    model = LogisticRegression()
    model.fit(
        dataset[FEATURE_COLUMNS],
        dataset["label"],
    )

    analysis = build_error_analysis_table(
        model,
        dataset,
    )

    result = get_high_confidence_errors(
        analysis,
        error_type="true_positive",
        top_n=3,
    )

    assert len(result) <= 3

    if not result.empty:
        assert (
            result["error_type"] == "true_positive"
        ).all()


def test_invalid_error_type_rejected():
    dataset = make_dataset()

    model = LogisticRegression()
    model.fit(
        dataset[FEATURE_COLUMNS],
        dataset["label"],
    )

    analysis = build_error_analysis_table(
        model,
        dataset,
    )

    with pytest.raises(ValueError):
        get_high_confidence_errors(
            analysis,
            error_type="invalid",
        )


def test_summarize_feature_profiles():
    dataset = make_dataset()

    model = LogisticRegression()
    model.fit(
        dataset[FEATURE_COLUMNS],
        dataset["label"],
    )

    analysis = build_error_analysis_table(
        model,
        dataset,
    )

    result = summarize_feature_profiles(analysis)

    assert "error_type" in result.columns

    for feature in FEATURE_COLUMNS:
        assert f"{feature}_mean" in result.columns
        assert f"{feature}_median" in result.columns
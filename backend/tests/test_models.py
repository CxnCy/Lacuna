import pickle

import pandas as pd
import pytest

from lacuna.models.backtest import (
    load_dataset,
    prepare_xy,
    run_temporal_backtest,
    validate_temporal_split,
)
from lacuna.models.evaluate import evaluate_predictions, precision_at_k
from lacuna.models.io import load_model_artifact, save_model_artifact
from lacuna.models.train import FEATURE_COLUMNS, build_models


def make_dataset(cutoff_year, rows=20):
    records = []

    for index in range(rows):
        label = 1 if index % 5 == 0 else 0

        records.append(
            {
                "topic_a": f"T{index}",
                "topic_b": f"T{index + 1}",
                "cutoff_year": cutoff_year,
                "prediction_window_start": cutoff_year + 1,
                "prediction_window_end": cutoff_year + 5,
                "future_co_publication_count": 10 if label else 0,
                "label": label,
                "adamic_adar_score": float(index + 1),
                "common_neighbour_count": index % 7,
                "degree_a": index + 2,
                "degree_b": index + 3,
                "jaccard_coefficient": index / max(rows, 1),
                "preferential_attachment": (index + 2) * (index + 3),
                "weighted_degree_a": float(index + 4),
                "weighted_degree_b": float(index + 5),
            }
        )

    return pd.DataFrame(records)


def test_feature_columns_exclude_future_information():
    assert "future_co_publication_count" not in FEATURE_COLUMNS
    assert "prediction_window_start" not in FEATURE_COLUMNS
    assert "prediction_window_end" not in FEATURE_COLUMNS
    assert "label" not in FEATURE_COLUMNS


def test_prepare_xy_uses_only_declared_features():
    df = make_dataset(2010)
    X, y = prepare_xy(df)

    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y)
    assert "future_co_publication_count" not in X.columns


def test_validate_temporal_split_accepts_past_to_future():
    validate_temporal_split([2010], 2015)
    validate_temporal_split([2010, 2015], 2020)


def test_validate_temporal_split_rejects_leakage():
    with pytest.raises(ValueError):
        validate_temporal_split([2010, 2020], 2020)


def test_precision_at_k():
    y_true = [1, 0, 1, 0]
    y_score = [0.9, 0.8, 0.7, 0.1]

    assert precision_at_k(y_true, y_score, 2) == 0.5


def test_evaluate_predictions_returns_required_metrics():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    y_score = [0.1, 0.9, 0.2, 0.8]

    metrics = evaluate_predictions(y_true, y_pred, y_score)

    expected = {
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "precision_at_10",
        "precision_at_50",
        "precision_at_100",
    }

    assert expected.issubset(metrics.keys())


def test_build_models_returns_expected_models():
    models = build_models()

    assert set(models.keys()) == {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    }


def test_load_dataset_reads_list_of_dicts(tmp_path):
    records = make_dataset(2010).to_dict("records")
    path = tmp_path / "dataset.pkl"

    with path.open("wb") as file:
        pickle.dump(records, file)

    loaded = load_dataset(path)

    assert isinstance(loaded, pd.DataFrame)
    assert len(loaded) == len(records)


def test_temporal_backtest_runs_all_models():
    datasets = {
        2010: make_dataset(2010, rows=30),
        2015: make_dataset(2015, rows=30),
    }

    results = run_temporal_backtest(
        datasets,
        train_cutoffs=[2010],
        test_cutoff=2015,
    )

    assert len(results) == 3

    for result in results:
        assert result["test_cutoff"] == 2015
        assert "pr_auc" in result["metrics"]
        assert "model" in result


def test_model_artifact_round_trip(tmp_path):
    model = build_models()["random_forest"]

    metadata = {
        "model_name": "random_forest",
        "test_cutoff": 2015,
    }

    path = tmp_path / "model.joblib"

    save_model_artifact(model, path, metadata)

    artifact = load_model_artifact(path)

    assert artifact["metadata"] == metadata
    assert "model" in artifact

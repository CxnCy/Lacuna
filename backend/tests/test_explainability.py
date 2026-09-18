import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lacuna.models.explainability import (
    get_logistic_regression_coefficients,
    get_permutation_importance,
    get_random_forest_feature_importance,
)
from lacuna.models.train import FEATURE_COLUMNS


def make_dataset():
    X = pd.DataFrame(
        {
            feature: [float(i + j) for i in range(10)]
            for j, feature in enumerate(FEATURE_COLUMNS)
        }
    )
    y = pd.Series([0, 1] * 5)
    return X, y


def test_logistic_regression_coefficients():
    X, y = make_dataset()

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression()),
        ]
    )
    model.fit(X, y)

    result = get_logistic_regression_coefficients(model)

    assert set(result["feature"]) == set(FEATURE_COLUMNS)
    assert len(result) == len(FEATURE_COLUMNS)
    assert "coefficient" in result.columns
    assert "absolute_coefficient" in result.columns
    assert "direction" in result.columns


def test_random_forest_feature_importance():
    X, y = make_dataset()

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )
    model.fit(X, y)

    result = get_random_forest_feature_importance(model)

    assert set(result["feature"]) == set(FEATURE_COLUMNS)
    assert len(result) == len(FEATURE_COLUMNS)
    assert "importance" in result.columns
    assert result["importance"].sum() == pytest.approx(1.0)


def test_permutation_importance():
    X, y = make_dataset()

    model = LogisticRegression()
    model.fit(X, y)

    result = get_permutation_importance(
        model,
        X,
        y,
        n_repeats=2,
    )

    assert set(result["feature"]) == set(FEATURE_COLUMNS)
    assert len(result) == len(FEATURE_COLUMNS)
    assert "importance_mean" in result.columns
    assert "importance_std" in result.columns


def test_permutation_importance_rejects_missing_features():
    X, y = make_dataset()

    model = LogisticRegression()
    model.fit(X, y)

    incomplete_X = X.drop(
        columns=[FEATURE_COLUMNS[0]]
    )

    with pytest.raises(ValueError):
        get_permutation_importance(
            model,
            incomplete_X,
            y,
            n_repeats=2,
        )
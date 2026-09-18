"""Model construction and training utilities for Lacuna."""

from __future__ import annotations

from typing import Dict

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "adamic_adar_score",
    "common_neighbour_count",
    "degree_a",
    "degree_b",
    "jaccard_coefficient",
    "preferential_attachment",
    "weighted_degree_a",
    "weighted_degree_b",
]

TARGET_COLUMN = "label"

# These columns are deliberately excluded from model features.
# In particular, future_co_publication_count is future information and
# must never be included in X.
NON_FEATURE_COLUMNS = [
    "topic_a",
    "topic_b",
    "cutoff_year",
    "prediction_window_start",
    "prediction_window_end",
    "future_co_publication_count",
    "label",
]


def build_models(random_state: int = 42) -> Dict[str, ClassifierMixin]:
    """Return the initial Lacuna model suite.

    Logistic regression and random forest use balanced class weighting
    because positive emergence labels are rare.

    HistGradientBoosting does not expose class_weight consistently across
    all scikit-learn versions, so sample weights are handled during fitting.
    """
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=200,
            random_state=random_state,
        ),
    }


def fit_model(
    model_name: str,
    model: ClassifierMixin,
    X,
    y,
) -> ClassifierMixin:
    """Fit one Lacuna model with appropriate imbalance handling."""
    if model_name == "hist_gradient_boosting":
        negative_count = int((y == 0).sum())
        positive_count = int((y == 1).sum())

        if positive_count == 0:
            raise ValueError(
                "Cannot train HistGradientBoosting without positive examples."
            )

        positive_weight = negative_count / positive_count
        sample_weight = y.map(
            lambda value: positive_weight if value == 1 else 1.0
        )

        model.fit(X, y, sample_weight=sample_weight)
    else:
        model.fit(X, y)

    return model

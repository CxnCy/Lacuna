"""Feature explainability utilities for Lacuna models."""

from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance

from lacuna.models.train import FEATURE_COLUMNS


def get_logistic_regression_coefficients(model) -> pd.DataFrame:
    """Extract feature coefficients from the fitted logistic regression pipeline."""
    if not hasattr(model, "named_steps"):
        raise ValueError(
            "Expected a fitted scikit-learn Pipeline for logistic regression."
        )

    classifier = model.named_steps.get("classifier")

    if classifier is None or not hasattr(classifier, "coef_"):
        raise ValueError(
            "Logistic regression classifier does not expose fitted coefficients."
        )

    coefficients = classifier.coef_[0]

    if len(coefficients) != len(FEATURE_COLUMNS):
        raise ValueError(
            "Coefficient count does not match Lacuna feature count."
        )

    result = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "coefficient": coefficients,
        }
    )

    result["absolute_coefficient"] = result["coefficient"].abs()
    result["direction"] = result["coefficient"].apply(
        lambda value: "positive" if value > 0 else "negative" if value < 0 else "neutral"
    )

    return result.sort_values(
        "absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)


def get_random_forest_feature_importance(model) -> pd.DataFrame:
    """Extract impurity-based feature importance from a fitted random forest."""
    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "Model does not expose feature_importances_."
        )

    importances = model.feature_importances_

    if len(importances) != len(FEATURE_COLUMNS):
        raise ValueError(
            "Feature importance count does not match Lacuna feature count."
        )

    result = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": importances,
        }
    )

    return result.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)


def get_permutation_importance(
    model,
    X,
    y,
    *,
    scoring: str = "average_precision",
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calculate permutation importance on an evaluation dataset.

    This should normally be run on the future temporal test split rather
    than the training data so that importance reflects out-of-time
    predictive contribution.
    """
    missing_features = [
        feature for feature in FEATURE_COLUMNS if feature not in X.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required Lacuna features: {missing_features}"
        )

    X_features = X[FEATURE_COLUMNS]

    result = permutation_importance(
        model,
        X_features,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    return importance_df.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)
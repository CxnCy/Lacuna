"""Expanding-window temporal backtesting for Lacuna."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from lacuna.models.evaluate import evaluate_predictions
from lacuna.models.train import FEATURE_COLUMNS, build_models, fit_model


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a Step 5 list-of-dictionaries ML dataset as a DataFrame."""
    path = Path(path)

    with path.open("rb") as file:
        records = pickle.load(file)

    if not isinstance(records, list):
        raise TypeError(
            f"Expected list dataset at {path}, "
            f"received {type(records).__name__}."
        )

    dataframe = pd.DataFrame(records)

    required = set(FEATURE_COLUMNS) | {
        "label",
        "cutoff_year",
        "future_co_publication_count",
    }

    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"Dataset {path} is missing required columns: "
            f"{sorted(missing)}"
        )

    return dataframe


def validate_temporal_split(
    train_cutoffs: Sequence[int],
    test_cutoff: int,
) -> None:
    """Ensure every training cutoff strictly precedes the test cutoff."""
    if not train_cutoffs:
        raise ValueError("At least one training cutoff is required.")

    if max(train_cutoffs) >= test_cutoff:
        raise ValueError(
            "Temporal leakage detected: training cutoffs must "
            "strictly precede the test cutoff."
        )


def prepare_xy(dataframe: pd.DataFrame):
    """Extract leakage-safe feature matrix X and target y."""
    X = dataframe[FEATURE_COLUMNS].copy()
    y = dataframe["label"].astype(int).copy()
    return X, y


def run_temporal_backtest(
    datasets: Dict[int, pd.DataFrame],
    train_cutoffs: Iterable[int],
    test_cutoff: int,
    random_state: int = 42,
) -> List[dict]:
    """Train all models on historical cutoffs and evaluate on a future cutoff."""
    train_cutoffs = sorted(train_cutoffs)
    validate_temporal_split(train_cutoffs, test_cutoff)

    missing_cutoffs = [
        cutoff
        for cutoff in [*train_cutoffs, test_cutoff]
        if cutoff not in datasets
    ]

    if missing_cutoffs:
        raise KeyError(
            f"Missing datasets for cutoffs: {missing_cutoffs}"
        )

    train_df = pd.concat(
        [datasets[cutoff] for cutoff in train_cutoffs],
        ignore_index=True,
    )
    test_df = datasets[test_cutoff].copy()

    X_train, y_train = prepare_xy(train_df)
    X_test, y_test = prepare_xy(test_df)

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data must contain both positive and negative labels."
        )

    models = build_models(random_state=random_state)
    results = []

    for model_name, model in models.items():
        fitted_model = fit_model(
            model_name,
            model,
            X_train,
            y_train,
        )

        y_pred = fitted_model.predict(X_test)

        if not hasattr(fitted_model, "predict_proba"):
            raise TypeError(
                f"{model_name} does not support probability prediction."
            )

        y_score = fitted_model.predict_proba(X_test)[:, 1]

        metrics = evaluate_predictions(
            y_test,
            y_pred,
            y_score,
        )

        results.append(
            {
                "model_name": model_name,
                "train_cutoffs": tuple(train_cutoffs),
                "test_cutoff": test_cutoff,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_positive_rate": float(y_train.mean()),
                "test_positive_rate": float(y_test.mean()),
                "metrics": metrics,
                "model": fitted_model,
            }
        )

    return results

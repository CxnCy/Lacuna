"""Run Step 7.5 probability calibration diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lacuna.models.calibration import (
    build_calibration_table,
    get_prediction_probabilities,
    summarize_calibration,
)
from lacuna.models.io import (
    load_model_artifact,
)
from lacuna.models.train import (
    FEATURE_COLUMNS,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]

MODEL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
)

ML_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "analysis"
    / "calibration"
)


BACKTESTS = [
    {
        "train_label": "2010",
        "test_year": 2015,
    },
    {
        "train_label": "2010_2015",
        "test_year": 2020,
    },
]


MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
]


def load_dataset(
    year: int,
) -> pd.DataFrame:
    """Load one temporal ML test dataset."""
    path = (
        ML_DIR
        / f"lacuna_ml_dataset_{year}.pkl"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"ML dataset not found: {path}"
        )

    dataset = pd.read_pickle(
        path
    )

    if isinstance(
        dataset,
        list,
    ):
        dataset = pd.DataFrame(
            dataset
        )

    if not isinstance(
        dataset,
        pd.DataFrame,
    ):
        raise TypeError(
            "ML dataset must be a pandas "
            "DataFrame or list of dictionaries."
        )

    return dataset


def load_estimator(
    model_name: str,
    train_label: str,
    test_year: int,
):
    """Load and validate a saved Step 6 estimator."""
    path = (
        MODEL_DIR
        / (
            f"{model_name}"
            f"_train_{train_label}"
            f"_test_{test_year}.joblib"
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}"
        )

    artifact = load_model_artifact(
        path
    )

    if not isinstance(
        artifact,
        dict,
    ):
        raise TypeError(
            "Expected model artifact "
            "to be a dictionary."
        )

    if "model" not in artifact:
        raise ValueError(
            "Model artifact does not "
            "contain a 'model' key."
        )

    model = artifact[
        "model"
    ]

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise ValueError(
            f"{model_name} does not expose "
            "predict_proba and cannot be "
            "evaluated as a probability model."
        )

    return model


def save_dataframe(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    """Save calibration output."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_DIR
        / filename,
        index=False,
    )


def run_calibration() -> None:
    """Execute calibration diagnostics for all backtests."""
    summary_rows = []

    print(
        "=" * 72
    )
    print(
        "LACUNA — STEP 7.5 CALIBRATION"
    )
    print(
        "=" * 72
    )

    for backtest in BACKTESTS:
        train_label = backtest[
            "train_label"
        ]

        test_year = backtest[
            "test_year"
        ]

        dataset = load_dataset(
            test_year
        )

        missing_features = [
            feature
            for feature in FEATURE_COLUMNS
            if feature not in dataset.columns
        ]

        if missing_features:
            raise ValueError(
                "Dataset is missing required "
                f"features: {missing_features}"
            )

        if "label" not in dataset.columns:
            raise ValueError(
                "Dataset is missing label column."
            )

        X_test = dataset[
            FEATURE_COLUMNS
        ]

        y_test = dataset[
            "label"
        ].astype(int)

        print()
        print(
            f"TRAIN {train_label} "
            f"→ TEST {test_year}"
        )
        print(
            "-" * 72
        )

        for model_name in MODEL_NAMES:
            model = load_estimator(
                model_name,
                train_label,
                test_year,
            )

            probabilities = (
                get_prediction_probabilities(
                    model,
                    X_test,
                )
            )

            summary = (
                summarize_calibration(
                    y_test,
                    probabilities,
                    n_bins=10,
                    strategy="quantile",
                )
            )

            summary_row = {
                "train_period": (
                    train_label
                ),
                "test_year": (
                    test_year
                ),
                "model_name": (
                    model_name
                ),
                **summary,
            }

            summary_rows.append(
                summary_row
            )

            calibration_table = (
                build_calibration_table(
                    y_test,
                    probabilities,
                    n_bins=10,
                    strategy="quantile",
                )
            )

            calibration_table.insert(
                0,
                "model_name",
                model_name,
            )

            calibration_table.insert(
                0,
                "test_year",
                test_year,
            )

            save_dataframe(
                calibration_table,
                (
                    f"{test_year}_"
                    f"{model_name}_"
                    "reliability.csv"
                ),
            )

            print(
                f"{model_name}: "
                f"Brier={summary['brier_score']:.6f} | "
                f"ECE={summary['expected_calibration_error']:.6f} | "
                f"Observed="
                f"{summary['observed_positive_rate']:.4%} | "
                f"Mean predicted="
                f"{summary['mean_predicted_probability']:.4%}"
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    save_dataframe(
        summary_df,
        "calibration_summary.csv",
    )

    print()
    print(
        "=" * 72
    )
    print(
        "STEP 7.5 CALIBRATION COMPLETE"
    )
    print(
        f"Artifacts: {OUTPUT_DIR}"
    )
    print(
        "=" * 72
    )


if __name__ == "__main__":
    run_calibration()
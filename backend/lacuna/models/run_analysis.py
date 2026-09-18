"""Run Step 7 feature explainability and error analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lacuna.models.error_analysis import (
    build_error_analysis_table,
    summarize_feature_profiles,
)
from lacuna.models.explainability import (
    get_logistic_regression_coefficients,
    get_permutation_importance,
    get_random_forest_feature_importance,
)
from lacuna.models.io import load_model_artifact
from lacuna.models.train import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
ANALYSIS_DIR = PROJECT_ROOT / "artifacts" / "analysis"

EXPLAINABILITY_DIR = (
    ANALYSIS_DIR / "feature_explainability"
)

ERROR_ANALYSIS_DIR = (
    ANALYSIS_DIR / "error_analysis"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml"
)


BACKTESTS = [
    {
        "test_year": 2015,
        "train_label": "2010",
    },
    {
        "test_year": 2020,
        "train_label": "2010_2015",
    },
]


MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
]


def find_dataset(year: int) -> Path:
    """Locate the ML dataset for a cutoff year."""
    path = (
        DATA_DIR
        / f"lacuna_ml_dataset_{year}.pkl"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Could not find ML dataset for "
            f"cutoff year {year}: {path}"
        )

    return path


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a Lacuna ML dataset."""
    if path.suffix != ".pkl":
        raise ValueError(
            f"Unsupported dataset format: {path.suffix}"
        )

    dataset = pd.read_pickle(path)

    if isinstance(dataset, list):
        dataset = pd.DataFrame(dataset)

    if not isinstance(dataset, pd.DataFrame):
        raise TypeError(
            "Loaded ML dataset must be a pandas "
            "DataFrame or list of dictionaries."
        )

    return dataset


def find_model_artifact(
    model_name: str,
    train_label: str,
    test_year: int,
) -> Path:
    """Locate a saved Step 6 model artifact."""
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

    return path


def extract_model_and_metadata(
    artifact,
):
    """Extract fitted estimator and metadata from a model artifact."""
    if not isinstance(artifact, dict):
        raise TypeError(
            "Expected model artifact to be a dictionary."
        )

    if "model" not in artifact:
        raise ValueError(
            "Model artifact does not contain a 'model' key."
        )

    if "metadata" not in artifact:
        raise ValueError(
            "Model artifact does not contain a 'metadata' key."
        )

    model = artifact["model"]
    metadata = artifact["metadata"]

    if not (
        hasattr(model, "predict_proba")
        or hasattr(model, "decision_function")
    ):
        raise ValueError(
            "Saved estimator must expose predict_proba "
            "or decision_function."
        )

    if not isinstance(metadata, dict):
        raise TypeError(
            "Model artifact metadata must be a dictionary."
        )

    return model, metadata


def validate_artifact_metadata(
    metadata: dict,
    model_name: str,
    test_year: int,
) -> None:
    """Perform basic validation of saved model metadata."""
    metadata_model_name = metadata.get(
        "model_name"
    )

    if (
        metadata_model_name is not None
        and metadata_model_name != model_name
    ):
        raise ValueError(
            "Artifact model-name mismatch: "
            f"expected {model_name}, "
            f"found {metadata_model_name}"
        )

    metadata_test_year = metadata.get(
        "test_year"
    )

    if (
        metadata_test_year is not None
        and int(metadata_test_year) != test_year
    ):
        raise ValueError(
            "Artifact test-year mismatch: "
            f"expected {test_year}, "
            f"found {metadata_test_year}"
        )


def save_dataframe(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    """Save an analysis dataframe as CSV."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        path,
        index=False,
    )


def run_analysis() -> None:
    """Run Step 7.1 and Step 7.2 analyses."""
    EXPLAINABILITY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ERROR_ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for backtest in BACKTESTS:
        test_year = backtest["test_year"]
        train_label = backtest["train_label"]

        print()
        print("=" * 72)
        print(
            "STEP 7 ANALYSIS — "
            f"TRAIN {train_label} "
            f"→ TEST {test_year}"
        )
        print("=" * 72)

        dataset_path = find_dataset(
            test_year
        )

        dataset = load_dataset(
            dataset_path
        )

        print(
            f"Dataset: {dataset_path}"
        )
        print(
            f"Rows: {len(dataset)}"
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

        for model_name in MODEL_NAMES:
            print()
            print(
                f"Analysing: {model_name}"
            )

            artifact_path = (
                find_model_artifact(
                    model_name,
                    train_label,
                    test_year,
                )
            )

            print(
                f"Model: {artifact_path}"
            )

            artifact = load_model_artifact(
                artifact_path
            )

            model, metadata = (
                extract_model_and_metadata(
                    artifact
                )
            )

            validate_artifact_metadata(
                metadata,
                model_name,
                test_year,
            )

            error_table = (
                build_error_analysis_table(
                    model,
                    dataset,
                )
            )

            error_path = (
                ERROR_ANALYSIS_DIR
                / (
                    f"{test_year}_"
                    f"{model_name}_errors.csv"
                )
            )

            save_dataframe(
                error_table,
                error_path,
            )

            feature_profiles = (
                summarize_feature_profiles(
                    error_table
                )
            )

            profile_path = (
                ERROR_ANALYSIS_DIR
                / (
                    f"{test_year}_"
                    f"{model_name}_"
                    "feature_profiles.csv"
                )
            )

            save_dataframe(
                feature_profiles,
                profile_path,
            )

            permutation = (
                get_permutation_importance(
                    model,
                    X_test,
                    y_test,
                )
            )

            permutation_path = (
                EXPLAINABILITY_DIR
                / (
                    f"{test_year}_"
                    f"{model_name}_"
                    "permutation_importance.csv"
                )
            )

            save_dataframe(
                permutation,
                permutation_path,
            )

            if (
                model_name
                == "logistic_regression"
            ):
                coefficients = (
                    get_logistic_regression_coefficients(
                        model
                    )
                )

                coefficient_path = (
                    EXPLAINABILITY_DIR
                    / (
                        f"{test_year}_"
                        "logistic_regression_"
                        "coefficients.csv"
                    )
                )

                save_dataframe(
                    coefficients,
                    coefficient_path,
                )

            if (
                model_name
                == "random_forest"
            ):
                importance = (
                    get_random_forest_feature_importance(
                        model
                    )
                )

                importance_path = (
                    EXPLAINABILITY_DIR
                    / (
                        f"{test_year}_"
                        "random_forest_"
                        "feature_importance.csv"
                    )
                )

                save_dataframe(
                    importance,
                    importance_path,
                )

            counts = (
                error_table[
                    "error_type"
                ]
                .value_counts()
                .to_dict()
            )

            print(
                f"Outcomes: {counts}"
            )

            print(
                "Saved error analysis: "
                f"{error_path}"
            )

            print(
                "Saved feature profiles: "
                f"{profile_path}"
            )

            print(
                "Saved permutation importance: "
                f"{permutation_path}"
            )

    print()
    print("=" * 72)
    print(
        "STEP 7.1 + 7.2 ANALYSIS COMPLETE"
    )
    print(
        f"Artifacts: {ANALYSIS_DIR}"
    )
    print("=" * 72)


if __name__ == "__main__":
    run_analysis()
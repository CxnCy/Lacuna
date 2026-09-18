"""Generate ranked Lacuna candidate research connections."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lacuna.prediction.engine import (
    build_prediction_manifest,
    load_prediction_model,
    rank_candidate_connections,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]

MODEL_SELECTION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "analysis"
    / "model_selection"
    / "model_recommendation.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "predictions"
)

PREDICTION_CUTOFF_YEAR = 2025

TRAIN_PERIOD = (
    "2010_2015"
)

VALIDATED_TEST_YEAR = 2020

TOP_K = 100


def load_dataframe(
    path: Path,
) -> pd.DataFrame:
    """Load a DataFrame-compatible pickle artifact."""
    if not path.exists():
        raise FileNotFoundError(
            f"Data artifact not found: {path}"
        )

    data = pd.read_pickle(
        path
    )

    if isinstance(
        data,
        list,
    ):
        data = pd.DataFrame(
            data
        )

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            f"Expected DataFrame or list at {path}, "
            f"found {type(data)}"
        )

    return data


def load_model_recommendation() -> dict:
    """Load the Step 7.6 model-selection decision."""
    if not MODEL_SELECTION_PATH.exists():
        raise FileNotFoundError(
            "Model recommendation artifact not found: "
            f"{MODEL_SELECTION_PATH}"
        )

    recommendation = pd.read_csv(
        MODEL_SELECTION_PATH
    )

    if recommendation.empty:
        raise ValueError(
            "Model recommendation artifact is empty."
        )

    required = {
        "selected_model",
        "selected_role",
        "probability_interpretation",
    }

    missing = required.difference(
        recommendation.columns
    )

    if missing:
        raise ValueError(
            "Model recommendation is missing required fields: "
            f"{sorted(missing)}"
        )

    return recommendation.iloc[
        0
    ].to_dict()


def resolve_model_artifact(
    model_name: str,
) -> Path:
    """
    Resolve the validated artifact used by the prediction engine.

    The latest temporally validated artifact is used. This remains a
    research-baseline ranking model rather than a production forecaster.
    """
    artifact_path = (
        MODEL_DIR
        / (
            f"{model_name}"
            f"_train_{TRAIN_PERIOD}"
            f"_test_{VALIDATED_TEST_YEAR}.joblib"
        )
    )

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Selected model artifact not found: {artifact_path}"
        )

    return artifact_path


def run_predictions() -> None:
    """Generate ranked candidate lacunae for Atlas integration."""
    recommendation = (
        load_model_recommendation()
    )

    model_name = str(
        recommendation[
            "selected_model"
        ]
    )

    model_role = str(
        recommendation[
            "selected_role"
        ]
    )

    if model_role not in {
        "ranking_research_baseline",
        "candidate_ranking_model",
    }:
        raise ValueError(
            "Selected model does not have an approved "
            f"prediction-engine role: {model_role}"
        )

    artifact_path = (
        resolve_model_artifact(
            model_name
        )
    )

    feature_path = (
        FEATURE_DIR
        / (
            f"lacuna_features_"
            f"{PREDICTION_CUTOFF_YEAR}.pkl"
        )
    )

    candidates = load_dataframe(
        feature_path
    )

    model = load_prediction_model(
        artifact_path
    )

    ranked = (
        rank_candidate_connections(
            model,
            candidates,
            top_k=TOP_K,
            model_name=model_name,
            model_role=model_role,
        )
    )

    manifest = (
        build_prediction_manifest(
            model_name=model_name,
            model_role=model_role,
            artifact_path=artifact_path,
            candidate_count=len(
                candidates
            ),
            output_count=len(
                ranked
            ),
            cutoff_year=PREDICTION_CUTOFF_YEAR,
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranking_path = (
        OUTPUT_DIR
        / (
            f"ranked_lacunae_"
            f"{PREDICTION_CUTOFF_YEAR}.csv"
        )
    )

    manifest_path = (
        OUTPUT_DIR
        / (
            f"ranked_lacunae_"
            f"{PREDICTION_CUTOFF_YEAR}"
            "_manifest.json"
        )
    )

    ranked.to_csv(
        ranking_path,
        index=False,
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print("=" * 72)
    print(
        "LACUNA — STEP 7.7 PREDICTION ENGINE"
    )
    print("=" * 72)

    print(
        f"Model: {model_name}"
    )

    print(
        f"Role: {model_role}"
    )

    print(
        f"Model artifact: {artifact_path}"
    )

    print(
        f"Prediction cutoff: "
        f"{PREDICTION_CUTOFF_YEAR}"
    )

    print(
        f"Candidate pairs scored: "
        f"{len(candidates)}"
    )

    print(
        f"Ranked candidates exported: "
        f"{len(ranked)}"
    )

    print()
    print(
        "IMPORTANT: ranking_score is a relative "
        "model ranking signal, not a calibrated "
        "probability of future emergence."
    )

    print()
    print(
        f"Ranking: {ranking_path}"
    )

    print(
        f"Manifest: {manifest_path}"
    )

    print()
    print("=" * 72)
    print(
        "STEP 7.7 PREDICTION ENGINE COMPLETE"
    )
    print("=" * 72)


if __name__ == "__main__":
    run_predictions()
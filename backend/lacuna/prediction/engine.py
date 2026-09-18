"""Prediction and ranking engine for Lacuna candidate research connections."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from lacuna.models.io import load_model_artifact
from lacuna.models.train import FEATURE_COLUMNS


FORBIDDEN_MODEL_FEATURES = {
    "topic_a",
    "topic_b",
    "cutoff_year",
    "prediction_window_start",
    "prediction_window_end",
    "future_co_publication_count",
    "label",
}

DEFAULT_METADATA_COLUMNS = [
    "topic_a",
    "topic_b",
    "cutoff_year",
    "prediction_window_start",
    "prediction_window_end",
]


def validate_prediction_features(
    candidates: pd.DataFrame,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> None:
    """Validate that prediction inputs contain leakage-safe model features."""
    forbidden = set(
        feature_columns
    ).intersection(
        FORBIDDEN_MODEL_FEATURES
    )

    if forbidden:
        raise ValueError(
            "Model feature configuration contains forbidden "
            f"metadata or future-derived fields: {sorted(forbidden)}"
        )

    missing = [
        feature
        for feature in feature_columns
        if feature not in candidates.columns
    ]

    if missing:
        raise ValueError(
            "Candidate dataset is missing required model features: "
            f"{missing}"
        )


def load_prediction_model(
    artifact_path: str | Path,
):
    """Load the estimator from a validated Lacuna model artifact."""
    artifact_path = Path(
        artifact_path
    )

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {artifact_path}"
        )

    artifact = load_model_artifact(
        artifact_path
    )

    if not isinstance(
        artifact,
        dict,
    ):
        raise TypeError(
            "Expected model artifact to be a dictionary."
        )

    if "model" not in artifact:
        raise ValueError(
            "Model artifact does not contain a 'model' key."
        )

    model = artifact[
        "model"
    ]

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise ValueError(
            "Prediction engine requires an estimator "
            "that exposes predict_proba."
        )

    return model


def calculate_ranking_scores(
    model,
    candidates: pd.DataFrame,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> np.ndarray:
    """
    Calculate relative ranking scores.

    Scores are derived from model positive-class outputs but must not
    be interpreted as calibrated probabilities of future emergence.
    """
    validate_prediction_features(
        candidates,
        feature_columns,
    )

    X = candidates[
        list(
            feature_columns
        )
    ]

    probabilities = model.predict_proba(
        X
    )

    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != 2
    ):
        raise ValueError(
            "Prediction engine currently expects "
            "binary classification outputs."
        )

    scores = np.asarray(
        probabilities[
            :,
            1,
        ],
        dtype=float,
    )

    if np.any(
        ~np.isfinite(
            scores
        )
    ):
        raise ValueError(
            "Model produced non-finite ranking scores."
        )

    return scores


def rank_candidate_connections(
    model,
    candidates: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
    top_k: int | None = None,
    model_name: str = "unknown",
    model_role: str = "ranking_research_baseline",
) -> pd.DataFrame:
    """
    Score and rank candidate topic pairs for exploratory research use.

    The returned ranking_score is not a calibrated probability.
    """
    if top_k is not None and top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    scores = calculate_ranking_scores(
        model,
        candidates,
        feature_columns,
    )

    preserved_metadata = [
        column
        for column in metadata_columns
        if column in candidates.columns
    ]

    required_identity = [
        "topic_a",
        "topic_b",
    ]

    missing_identity = [
        column
        for column in required_identity
        if column not in candidates.columns
    ]

    if missing_identity:
        raise ValueError(
            "Candidate dataset must preserve topic identity columns: "
            f"{missing_identity}"
        )

    for column in required_identity:
        if column not in preserved_metadata:
            preserved_metadata.append(
                column
            )

    result = candidates[
        preserved_metadata
    ].copy()

    result[
        "ranking_score"
    ] = scores

    result[
        "model_name"
    ] = model_name

    result[
        "model_role"
    ] = model_role

    result[
        "score_interpretation"
    ] = (
        "relative_ranking_score_not_calibrated_probability"
    )

    result[
        "prediction_type"
    ] = (
        "candidate_emerging_research_connection"
    )

    result = result.sort_values(
        "ranking_score",
        ascending=False,
        kind="stable",
    ).reset_index(
        drop=True
    )

    result[
        "rank"
    ] = np.arange(
        1,
        len(
            result
        ) + 1,
    )

    ordered_columns = [
        "rank",
        "topic_a",
        "topic_b",
    ]

    for column in preserved_metadata:
        if column not in ordered_columns:
            ordered_columns.append(
                column
            )

    ordered_columns.extend(
        [
            "ranking_score",
            "model_name",
            "model_role",
            "score_interpretation",
            "prediction_type",
        ]
    )

    result = result[
        ordered_columns
    ]

    if top_k is not None:
        result = result.head(
            top_k
        ).copy()

    return result


def build_prediction_manifest(
    *,
    model_name: str,
    model_role: str,
    artifact_path: str | Path,
    candidate_count: int,
    output_count: int,
    cutoff_year: int | None = None,
) -> dict[str, Any]:
    """Build metadata describing a generated prediction ranking."""
    return {
        "model_name": model_name,
        "model_role": model_role,
        "model_artifact": str(
            artifact_path
        ),
        "cutoff_year": cutoff_year,
        "candidate_count": int(
            candidate_count
        ),
        "output_count": int(
            output_count
        ),
        "score_interpretation": (
            "relative_ranking_score_not_calibrated_probability"
        ),
        "prediction_type": (
            "candidate_emerging_research_connection"
        ),
        "scientific_claim": (
            "exploratory_model_ranking_not_prediction_of_discovery"
        ),
    }

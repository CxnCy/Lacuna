import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import (
    LogisticRegression,
)

from lacuna.models.train import (
    FEATURE_COLUMNS,
)
from lacuna.prediction.engine import (
    FORBIDDEN_MODEL_FEATURES,
    build_prediction_manifest,
    calculate_ranking_scores,
    rank_candidate_connections,
    validate_prediction_features,
)


def make_candidates():
    rows = []

    for index in range(
        20
    ):
        row = {
            "topic_a": (
                f"topic_{index}"
            ),
            "topic_b": (
                f"topic_{index + 100}"
            ),
            "cutoff_year": 2025,
        }

        for feature_index, feature in enumerate(
            FEATURE_COLUMNS
        ):
            row[
                feature
            ] = float(
                index
                + feature_index
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def make_model(
    candidates,
):
    X = candidates[
        FEATURE_COLUMNS
    ]

    y = np.array(
        [
            0,
            1,
        ]
        * 10
    )

    model = (
        LogisticRegression(
            max_iter=1000
        )
    )

    model.fit(
        X,
        y,
    )

    return model


def test_feature_configuration_is_leakage_safe():
    assert not set(
        FEATURE_COLUMNS
    ).intersection(
        FORBIDDEN_MODEL_FEATURES
    )


def test_validate_prediction_features():
    candidates = (
        make_candidates()
    )

    validate_prediction_features(
        candidates
    )


def test_missing_feature_rejected():
    candidates = (
        make_candidates()
        .drop(
            columns=[
                FEATURE_COLUMNS[
                    0
                ]
            ]
        )
    )

    with pytest.raises(
        ValueError
    ):
        validate_prediction_features(
            candidates
        )


def test_forbidden_feature_configuration_rejected():
    candidates = (
        make_candidates()
    )

    unsafe_features = list(
        FEATURE_COLUMNS
    ) + [
        "label"
    ]

    candidates[
        "label"
    ] = 0

    with pytest.raises(
        ValueError
    ):
        validate_prediction_features(
            candidates,
            unsafe_features,
        )


def test_calculate_ranking_scores():
    candidates = (
        make_candidates()
    )

    model = make_model(
        candidates
    )

    scores = (
        calculate_ranking_scores(
            model,
            candidates,
        )
    )

    assert len(
        scores
    ) == len(
        candidates
    )

    assert np.all(
        np.isfinite(
            scores
        )
    )


def test_rank_candidate_connections():
    candidates = (
        make_candidates()
    )

    model = make_model(
        candidates
    )

    result = (
        rank_candidate_connections(
            model,
            candidates,
            top_k=5,
            model_name=(
                "logistic_regression"
            ),
            model_role=(
                "ranking_research_baseline"
            ),
        )
    )

    assert len(
        result
    ) == 5

    assert result[
        "rank"
    ].tolist() == [
        1,
        2,
        3,
        4,
        5,
    ]

    assert result[
        "ranking_score"
    ].is_monotonic_decreasing

    assert (
        "topic_a"
        in result.columns
    )

    assert (
        "topic_b"
        in result.columns
    )

    assert (
        result[
            "score_interpretation"
        ]
        .eq(
            "relative_ranking_score_not_calibrated_probability"
        )
        .all()
    )


def test_ranking_does_not_expose_future_information():
    candidates = (
        make_candidates()
    )

    candidates[
        "future_co_publication_count"
    ] = 999

    candidates[
        "label"
    ] = 1

    model = make_model(
        candidates
    )

    result = (
        rank_candidate_connections(
            model,
            candidates,
            top_k=5,
        )
    )

    assert (
        "future_co_publication_count"
        not in result.columns
    )

    assert (
        "label"
        not in result.columns
    )


def test_invalid_top_k_rejected():
    candidates = (
        make_candidates()
    )

    model = make_model(
        candidates
    )

    with pytest.raises(
        ValueError
    ):
        rank_candidate_connections(
            model,
            candidates,
            top_k=0,
        )


def test_prediction_manifest():
    result = (
        build_prediction_manifest(
            model_name=(
                "logistic_regression"
            ),
            model_role=(
                "ranking_research_baseline"
            ),
            artifact_path=(
                "model.joblib"
            ),
            candidate_count=100,
            output_count=10,
            cutoff_year=2025,
        )
    )

    assert (
        result[
            "model_name"
        ]
        == "logistic_regression"
    )

    assert (
        result[
            "candidate_count"
        ]
        == 100
    )

    assert (
        result[
            "output_count"
        ]
        == 10
    )

    assert (
        result[
            "scientific_claim"
        ]
        == (
            "exploratory_model_ranking_not_prediction_of_discovery"
        )
    )
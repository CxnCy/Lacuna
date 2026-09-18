import pandas as pd

from lacuna.models.methodology_audit import (
    audit_candidate_dataset,
    audit_emergence_labels,
    audit_prediction_windows,
    build_methodology_flags,
    build_temporal_candidate_summary,
)


def test_audit_emergence_labels():
    dataset = pd.DataFrame(
        {
            "label": [
                1,
                1,
                1,
                0,
            ],
            "future_co_publication_count": [
                1,
                1,
                3,
                0,
            ],
        }
    )

    result = audit_emergence_labels(
        dataset,
        2015,
    )

    assert (
        result["candidate_count"]
        == 4
    )

    assert (
        result["positive_count"]
        == 3
    )

    assert (
        result[
            "single_future_co_publication_count"
        ]
        == 2
    )

    assert (
        result[
            "positive_pairs_future_count_ge_2"
        ]
        == 1
    )


def test_audit_prediction_windows():
    dataset = pd.DataFrame(
        {
            "prediction_window_start": [
                2016,
                2016,
            ],
            "prediction_window_end": [
                2020,
                2020,
            ],
        }
    )

    result = audit_prediction_windows(
        dataset,
        2015,
    )

    assert (
        result[
            "start_consistent_with_cutoff"
        ]
        is True
    )

    assert (
        result[
            "single_prediction_window"
        ]
        is True
    )


def test_candidate_integrity():
    candidates = pd.DataFrame(
        {
            "topic_a": [
                "A",
                "B",
                "A",
                "C",
            ],
            "topic_b": [
                "B",
                "A",
                "A",
                "D",
            ],
        }
    )

    result = audit_candidate_dataset(
        candidates,
        2015,
    )

    assert (
        result["self_pair_count"]
        == 1
    )

    assert (
        result["duplicate_pair_count"]
        == 1
    )


def test_temporal_candidate_summary():
    labels = [
        {
            "cutoff_year": 2010,
            "candidate_count": 100,
            "positive_count": 10,
            "negative_count": 90,
            "positive_rate": 0.10,
            "single_future_co_publication_share": 0.6,
        },
        {
            "cutoff_year": 2015,
            "candidate_count": 200,
            "positive_count": 5,
            "negative_count": 195,
            "positive_rate": 0.025,
            "single_future_co_publication_share": 0.8,
        },
    ]

    candidates = [
        {
            "cutoff_year": 2010,
            "unique_candidate_topics": 50,
            "self_pair_count": 0,
            "duplicate_pair_count": 0,
        },
        {
            "cutoff_year": 2015,
            "unique_candidate_topics": 80,
            "self_pair_count": 0,
            "duplicate_pair_count": 0,
        },
    ]

    graphs = [
        {
            "cutoff_year": 2010,
            "graph_node_count": 60,
            "graph_edge_count": 100,
        },
        {
            "cutoff_year": 2015,
            "graph_node_count": 90,
            "graph_edge_count": 150,
        },
    ]

    result = (
        build_temporal_candidate_summary(
            labels,
            candidates,
            graphs,
        )
    )

    assert len(
        result
    ) == 2

    assert (
        "candidates_per_graph_node"
        in result.columns
    )

    assert (
        result.iloc[1][
            "candidate_count_change"
        ]
        == 100
    )


def test_methodology_flags():
    summary = pd.DataFrame(
        {
            "cutoff_year": [
                2010,
                2015,
            ],
            "candidate_count": [
                100,
                200,
            ],
            "positive_count": [
                10,
                5,
            ],
            "positive_rate": [
                0.10,
                0.005,
            ],
            "single_future_co_publication_share": [
                0.6,
                0.8,
            ],
            "self_pair_count": [
                0,
                0,
            ],
            "duplicate_pair_count": [
                0,
                0,
            ],
        }
    )

    result = (
        build_methodology_flags(
            summary
        )
    )

    assert not result.empty

    assert (
        "emergence_label"
        in result[
            "audit_area"
        ].values
    )

    assert (
        "candidate_label_drift"
        in result[
            "audit_area"
        ].values
    )
import math

import networkx as nx
import pytest

from lacuna.candidates.generator import LacunaCandidate
from lacuna.features.dataset import (
    build_feature_rows,
    load_feature_rows,
    save_feature_rows,
)
from lacuna.features.pair_features import compute_pair_features


def build_test_graph() -> nx.Graph:
    graph = nx.Graph()

    graph.add_edge("A", "C", weight=2)
    graph.add_edge("B", "C", weight=3)
    graph.add_edge("A", "D", weight=4)
    graph.add_edge("C", "D", weight=1)

    return graph


def test_compute_pair_features_returns_expected_values():
    graph = build_test_graph()

    features = compute_pair_features(graph, "A", "B")

    assert features["common_neighbour_count"] == 1
    assert features["jaccard_coefficient"] == pytest.approx(0.5)

    expected_adamic_adar = 1 / math.log(3)
    assert features["adamic_adar_score"] == pytest.approx(
        expected_adamic_adar
    )

    assert features["preferential_attachment"] == 2
    assert features["degree_a"] == 2
    assert features["degree_b"] == 1
    assert features["weighted_degree_a"] == pytest.approx(6.0)
    assert features["weighted_degree_b"] == pytest.approx(3.0)


def test_compute_pair_features_rejects_existing_edge():
    graph = build_test_graph()

    with pytest.raises(
        ValueError,
        match="already has a direct edge",
    ):
        compute_pair_features(graph, "A", "C")


def test_compute_pair_features_rejects_missing_topic():
    graph = build_test_graph()

    with pytest.raises(
        ValueError,
        match="Topic B is not present",
    ):
        compute_pair_features(graph, "A", "UNKNOWN")


def test_compute_pair_features_rejects_self_pair():
    graph = build_test_graph()

    with pytest.raises(
        ValueError,
        match="two distinct topics",
    ):
        compute_pair_features(graph, "A", "A")


def test_build_feature_rows_creates_one_row_per_candidate():
    graph = build_test_graph()

    candidates = [
        LacunaCandidate(
            topic_a="A",
            topic_b="B",
            common_neighbor_count=1,
        )
    ]

    rows = build_feature_rows(
        graph=graph,
        candidates=candidates,
        cutoff_year=2015,
    )

    assert len(rows) == 1

    row = rows[0]

    assert row["cutoff_year"] == 2015
    assert row["topic_a"] == "A"
    assert row["topic_b"] == "B"
    assert row["common_neighbour_count"] == 1
    assert row["jaccard_coefficient"] == pytest.approx(0.5)
    assert row["adamic_adar_score"] == pytest.approx(
        1 / math.log(3)
    )
    assert row["preferential_attachment"] == 2
    assert row["degree_a"] == 2
    assert row["degree_b"] == 1
    assert row["weighted_degree_a"] == pytest.approx(6.0)
    assert row["weighted_degree_b"] == pytest.approx(3.0)


def test_build_feature_rows_rejects_invalid_candidate_type():
    graph = build_test_graph()

    candidates = [
        {
            "topic_a": "A",
            "topic_b": "B",
        }
    ]

    with pytest.raises(
        TypeError,
        match="must be a LacunaCandidate instance",
    ):
        build_feature_rows(
            graph=graph,
            candidates=candidates,
            cutoff_year=2015,
        )


def test_feature_rows_save_load_round_trip(tmp_path):
    graph = build_test_graph()

    candidates = [
        LacunaCandidate(
            topic_a="A",
            topic_b="B",
            common_neighbor_count=1,
        )
    ]

    rows = build_feature_rows(
        graph=graph,
        candidates=candidates,
        cutoff_year=2015,
    )

    output_path = tmp_path / "features.pkl"

    save_feature_rows(
        rows=rows,
        path=output_path,
    )

    loaded_rows = load_feature_rows(output_path)

    assert loaded_rows == rows
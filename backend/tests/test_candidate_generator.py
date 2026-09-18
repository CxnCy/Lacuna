import networkx as nx
import pytest

from lacuna.candidates.generator import (
    LacunaCandidate,
    generate_distance_2_candidates,
    load_candidates,
    save_candidates,
)


def test_generates_missing_distance_2_candidate():
    graph = nx.Graph()

    graph.add_edge("topic_a", "shared_topic")
    graph.add_edge("shared_topic", "topic_b")

    candidates = generate_distance_2_candidates(graph)

    assert candidates == [
        LacunaCandidate(
            topic_a="topic_a",
            topic_b="topic_b",
            common_neighbor_count=1,
        )
    ]


def test_excludes_existing_direct_edge():
    graph = nx.Graph()

    graph.add_edge("topic_a", "shared_topic")
    graph.add_edge("shared_topic", "topic_b")
    graph.add_edge("topic_a", "topic_b")

    candidates = generate_distance_2_candidates(graph)

    assert candidates == []


def test_counts_multiple_common_neighbors():
    graph = nx.Graph()

    graph.add_edge("topic_a", "shared_1")
    graph.add_edge("shared_1", "topic_b")

    graph.add_edge("topic_a", "shared_2")
    graph.add_edge("shared_2", "topic_b")

    candidates = generate_distance_2_candidates(graph)

    matching = [
        candidate
        for candidate in candidates
        if candidate.topic_a == "topic_a"
        and candidate.topic_b == "topic_b"
    ]

    assert len(matching) == 1
    assert matching[0].common_neighbor_count == 2


def test_returns_each_undirected_pair_once():
    graph = nx.Graph()

    graph.add_edge("topic_a", "shared_1")
    graph.add_edge("shared_1", "topic_b")

    graph.add_edge("topic_a", "shared_2")
    graph.add_edge("shared_2", "topic_b")

    candidates = generate_distance_2_candidates(graph)

    pairs = [
        (candidate.topic_a, candidate.topic_b)
        for candidate in candidates
    ]

    assert pairs.count(("topic_a", "topic_b")) == 1


def test_candidates_are_sorted_by_common_neighbor_count():
    graph = nx.Graph()

    # topic_a and topic_b share two neighbours.
    graph.add_edge("topic_a", "shared_1")
    graph.add_edge("shared_1", "topic_b")
    graph.add_edge("topic_a", "shared_2")
    graph.add_edge("shared_2", "topic_b")

    # topic_c and topic_d share one neighbour.
    graph.add_edge("topic_c", "shared_3")
    graph.add_edge("shared_3", "topic_d")

    candidates = generate_distance_2_candidates(graph)

    assert (
        candidates[0].common_neighbor_count
        >= candidates[-1].common_neighbor_count
    )


def test_rejects_directed_graph():
    graph = nx.DiGraph()

    graph.add_edge("topic_a", "shared_topic")
    graph.add_edge("shared_topic", "topic_b")

    with pytest.raises(
        ValueError,
        match="undirected topic graph",
    ):
        generate_distance_2_candidates(graph)


def test_save_and_load_candidates(tmp_path):
    candidates = [
        LacunaCandidate(
            topic_a="topic_a",
            topic_b="topic_b",
            common_neighbor_count=3,
        ),
        LacunaCandidate(
            topic_a="topic_c",
            topic_b="topic_d",
            common_neighbor_count=1,
        ),
    ]

    output_path = (
        tmp_path
        / "candidates"
        / "lacuna_candidates_2015.pkl"
    )

    saved_path = save_candidates(
        candidates,
        output_path,
    )

    loaded_candidates = load_candidates(saved_path)

    assert saved_path.exists()
    assert loaded_candidates == candidates


def test_save_candidates_creates_parent_directory(tmp_path):
    candidates = [
        LacunaCandidate(
            topic_a="topic_a",
            topic_b="topic_b",
            common_neighbor_count=1,
        )
    ]

    output_path = (
        tmp_path
        / "nested"
        / "candidate"
        / "directory"
        / "candidates.pkl"
    )

    save_candidates(
        candidates,
        output_path,
    )

    assert output_path.exists()
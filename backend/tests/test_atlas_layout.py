import networkx as nx
import pytest

from lacuna.atlas.layout import compute_layout
from lacuna.atlas.temporal_alignment import align_layout


def make_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=2)
    graph.add_edge("B", "C", weight=1)
    graph.add_edge("C", "D", weight=3)
    return graph


def test_layout_is_deterministic() -> None:
    graph = make_graph()

    first = compute_layout(
        graph,
        seed=42,
        iterations=50,
    )
    second = compute_layout(
        graph,
        seed=42,
        iterations=50,
    )

    assert first.keys() == second.keys()

    for node in first:
        assert first[node] == pytest.approx(second[node])


def test_layout_contains_every_node() -> None:
    graph = make_graph()
    positions = compute_layout(graph)

    assert set(positions) == set(graph.nodes())


def test_alignment_preserves_target_nodes() -> None:
    reference = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
    }

    target = {
        "A": (10.0, 10.0),
        "B": (10.0, 11.0),
        "C": (11.0, 11.0),
    }

    aligned = align_layout(reference, target)

    assert set(aligned) == {"A", "B", "C"}
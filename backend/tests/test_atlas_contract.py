import networkx as nx
import pytest

from lacuna.atlas.contract import build_atlas_state


def make_graph() -> nx.Graph:
    graph = nx.Graph(cutoff_year=2015)
    graph.add_node("T1", name="Topic One", hierarchy=None)
    graph.add_node("T2", name="Topic Two", hierarchy=None)
    graph.add_edge("T1", "T2", weight=3)
    return graph


def test_build_atlas_state_serialises_graph() -> None:
    graph = make_graph()

    state = build_atlas_state(
        graph,
        cutoff_year=2015,
        positions={
            "T1": (0.0, 0.0),
            "T2": (1.0, 1.0),
        },
        communities={
            "T1": 0,
            "T2": 0,
        },
    )

    assert state["cutoff_year"] == 2015
    assert state["metadata"]["node_count"] == 2
    assert state["metadata"]["edge_count"] == 1
    assert state["nodes"][0]["name"] == "Topic One"
    assert state["edges"][0]["weight"] == 3.0


def test_build_atlas_state_rejects_cutoff_mismatch() -> None:
    graph = make_graph()

    with pytest.raises(ValueError, match="Graph cutoff"):
        build_atlas_state(
            graph,
            cutoff_year=2020,
            positions={
                "T1": (0.0, 0.0),
                "T2": (1.0, 1.0),
            },
            communities={
                "T1": 0,
                "T2": 0,
            },
        )


def test_build_atlas_state_requires_all_positions() -> None:
    graph = make_graph()

    with pytest.raises(ValueError, match="positions missing"):
        build_atlas_state(
            graph,
            cutoff_year=2015,
            positions={"T1": (0.0, 0.0)},
            communities={
                "T1": 0,
                "T2": 0,
            },
        )
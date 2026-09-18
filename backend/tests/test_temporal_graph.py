import networkx as nx

from lacuna.network.temporal_graph import (
    build_temporal_snapshots,
    build_topic_graph,
    load_temporal_snapshot,
    save_temporal_snapshots,
)


def test_build_topic_graph_creates_weighted_edges():
    works = [
        {
            "publication_year": 2010,
            "topics": [
                {
                    "id": "T1",
                    "name": "Topic One",
                    "hierarchy": {"domain": "Science"},
                },
                {
                    "id": "T2",
                    "name": "Topic Two",
                    "hierarchy": {"domain": "Science"},
                },
            ],
        },
        {
            "publication_year": 2011,
            "topics": [
                {
                    "id": "T1",
                    "name": "Topic One",
                    "hierarchy": {"domain": "Science"},
                },
                {
                    "id": "T2",
                    "name": "Topic Two",
                    "hierarchy": {"domain": "Science"},
                },
            ],
        },
    ]

    graph = build_topic_graph(works, cutoff_year=2015)

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert graph["T1"]["T2"]["weight"] == 2


def test_build_topic_graph_excludes_future_works():
    works = [
        {
            "publication_year": 2010,
            "topics": [
                {
                    "id": "T1",
                    "name": "Past Topic",
                    "hierarchy": {},
                },
                {
                    "id": "T2",
                    "name": "Existing Topic",
                    "hierarchy": {},
                },
            ],
        },
        {
            "publication_year": 2020,
            "topics": [
                {
                    "id": "T1",
                    "name": "Past Topic",
                    "hierarchy": {},
                },
                {
                    "id": "T3",
                    "name": "Future Topic",
                    "hierarchy": {},
                },
            ],
        },
    ]

    graph = build_topic_graph(works, cutoff_year=2015)

    assert "T1" in graph
    assert "T2" in graph

    # T3 exists only in a future publication and must not leak
    # into the historical 2015 graph.
    assert "T3" not in graph

    assert graph.has_edge("T1", "T2")
    assert not graph.has_edge("T1", "T3")


def test_graph_preserves_topic_metadata():
    works = [
        {
            "publication_year": 2010,
            "topics": [
                {
                    "id": "T1",
                    "name": "Machine Learning",
                    "hierarchy": {
                        "domain": "Computer Science",
                        "field": "Artificial Intelligence",
                    },
                }
            ],
        }
    ]

    graph = build_topic_graph(
        works,
        cutoff_year=2015,
    )

    assert graph.nodes["T1"]["name"] == "Machine Learning"

    assert graph.nodes["T1"]["hierarchy"] == {
        "domain": "Computer Science",
        "field": "Artificial Intelligence",
    }


def test_graph_records_cutoff_year():
    graph = build_topic_graph(
        [],
        cutoff_year=2015,
    )

    assert graph.graph["cutoff_year"] == 2015


def test_temporal_snapshots_respect_each_cutoff():
    works = [
        {
            "publication_year": 2010,
            "topics": [
                {
                    "id": "T1",
                    "name": "Topic One",
                    "hierarchy": {},
                },
                {
                    "id": "T2",
                    "name": "Topic Two",
                    "hierarchy": {},
                },
            ],
        },
        {
            "publication_year": 2020,
            "topics": [
                {
                    "id": "T2",
                    "name": "Topic Two",
                    "hierarchy": {},
                },
                {
                    "id": "T3",
                    "name": "Topic Three",
                    "hierarchy": {},
                },
            ],
        },
    ]

    snapshots = build_temporal_snapshots(
        works,
        cutoff_years=[2010, 2020],
    )

    graph_2010 = snapshots[2010]
    graph_2020 = snapshots[2020]

    # Future knowledge must not appear in the earlier snapshot.
    assert "T3" not in graph_2010
    assert not graph_2010.has_edge("T2", "T3")

    # Later snapshots may contain newly observed knowledge.
    assert "T3" in graph_2020
    assert graph_2020.has_edge("T2", "T3")


def test_temporal_snapshots_are_cumulative():
    works = [
        {
            "publication_year": 2010,
            "topics": [
                {
                    "id": "T1",
                    "name": "Topic One",
                    "hierarchy": {},
                },
                {
                    "id": "T2",
                    "name": "Topic Two",
                    "hierarchy": {},
                },
            ],
        },
        {
            "publication_year": 2020,
            "topics": [
                {
                    "id": "T2",
                    "name": "Topic Two",
                    "hierarchy": {},
                },
                {
                    "id": "T3",
                    "name": "Topic Three",
                    "hierarchy": {},
                },
            ],
        },
    ]

    snapshots = build_temporal_snapshots(
        works,
        cutoff_years=[2010, 2020],
    )

    graph_2010 = snapshots[2010]
    graph_2020 = snapshots[2020]

    assert set(graph_2010.nodes).issubset(
        graph_2020.nodes
    )

    assert set(graph_2010.edges).issubset(
        graph_2020.edges
    )


def test_snapshot_persistence_round_trip(tmp_path):
    works = [
        {
            "publication_year": 2010,
            "topics": [
                {
                    "id": "T1",
                    "name": "Topic One",
                    "hierarchy": {
                        "domain": "Science",
                        "field": "Test Field",
                    },
                },
                {
                    "id": "T2",
                    "name": "Topic Two",
                    "hierarchy": None,
                },
            ],
        }
    ]

    snapshots = build_temporal_snapshots(
        works,
        cutoff_years=[2010],
    )

    save_temporal_snapshots(
        snapshots,
        tmp_path,
    )

    loaded_graph = load_temporal_snapshot(
        tmp_path / "topic_graph_2010.pkl"
    )

    assert loaded_graph.graph["cutoff_year"] == 2010

    assert loaded_graph.number_of_nodes() == 2
    assert loaded_graph.number_of_edges() == 1

    assert loaded_graph["T1"]["T2"]["weight"] == 1

    assert loaded_graph.nodes["T1"]["name"] == "Topic One"

    assert loaded_graph.nodes["T1"]["hierarchy"] == {
        "domain": "Science",
        "field": "Test Field",
    }

    assert loaded_graph.nodes["T2"]["hierarchy"] is None
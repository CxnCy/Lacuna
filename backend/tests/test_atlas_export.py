import json

import networkx as nx

from lacuna.atlas.export import (
    build_atlas_states,
    export_atlas_states,
)


def make_graph(
    year: int,
    include_third_node: bool = False,
) -> nx.Graph:
    graph = nx.Graph(cutoff_year=year)

    graph.add_node("T1", name="Topic One")
    graph.add_node("T2", name="Topic Two")
    graph.add_edge("T1", "T2", weight=2)

    if include_third_node:
        graph.add_node("T3", name="Topic Three")
        graph.add_edge("T2", "T3", weight=1)

    return graph


def test_build_atlas_states_preserves_temporal_nodes() -> None:
    states = build_atlas_states(
        {
            2010: make_graph(2010),
            2015: make_graph(
                2015,
                include_third_node=True,
            ),
        },
        layout_iterations=20,
    )

    nodes_2010 = {
        node["id"]
        for node in states[2010]["nodes"]
    }
    nodes_2015 = {
        node["id"]
        for node in states[2015]["nodes"]
    }

    assert nodes_2010 == {"T1", "T2"}
    assert nodes_2015 == {"T1", "T2", "T3"}


def test_export_atlas_states_writes_json(tmp_path) -> None:
    manifest = export_atlas_states(
        {
            2010: make_graph(2010),
            2015: make_graph(
                2015,
                include_third_node=True,
            ),
        },
        tmp_path,
        layout_iterations=20,
    )

    assert (tmp_path / "atlas_2010.json").exists()
    assert (tmp_path / "atlas_2015.json").exists()
    assert (tmp_path / "atlas_manifest.json").exists()

    saved_manifest = json.loads(
        (tmp_path / "atlas_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert saved_manifest == manifest
    assert manifest["prediction_overlay_included"] is False
    assert len(manifest["states"]) == 2
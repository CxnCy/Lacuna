"""Frontend-facing Atlas data contract."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import networkx as nx

from .layout import Position


ATLAS_SCHEMA_VERSION = "1.0"


def _serialise_identifier(value: Hashable) -> str:
    return str(value)


def build_atlas_state(
    graph: nx.Graph,
    *,
    cutoff_year: int,
    positions: Mapping[Hashable, Position],
    communities: Mapping[Hashable, int],
) -> dict[str, Any]:
    """
    Convert one historical topic graph into frontend-ready Atlas data.

    The contract intentionally describes network communities as communities,
    not as fabricated scientific domains or fields.
    """
    graph_cutoff = graph.graph.get("cutoff_year")

    if graph_cutoff is not None and int(graph_cutoff) != cutoff_year:
        raise ValueError(
            "Graph cutoff does not match requested Atlas cutoff: "
            f"{graph_cutoff} != {cutoff_year}"
        )

    missing_positions = set(graph.nodes()) - set(positions)

    if missing_positions:
        raise ValueError(
            "Atlas positions missing for "
            f"{len(missing_positions)} graph node(s)."
        )

    missing_communities = set(graph.nodes()) - set(communities)

    if missing_communities:
        raise ValueError(
            "Atlas communities missing for "
            f"{len(missing_communities)} graph node(s)."
        )

    nodes: list[dict[str, Any]] = []

    for node_id in sorted(graph.nodes(), key=str):
        metadata = graph.nodes[node_id]
        x_position, y_position = positions[node_id]

        nodes.append(
            {
                "id": _serialise_identifier(node_id),
                "name": str(
                    metadata.get("name")
                    or metadata.get("display_name")
                    or node_id
                ),
                "community_id": int(communities[node_id]),
                "x": float(x_position),
                "y": float(y_position),
                "degree": int(graph.degree(node_id)),
                "weighted_degree": float(
                    graph.degree(node_id, weight="weight")
                ),
                "hierarchy": metadata.get("hierarchy"),
            }
        )

    edges: list[dict[str, Any]] = []

    for source, target, metadata in graph.edges(data=True):
        source_id, target_id = sorted(
            (
                _serialise_identifier(source),
                _serialise_identifier(target),
            )
        )

        edges.append(
            {
                "source": source_id,
                "target": target_id,
                "weight": float(metadata.get("weight", 1.0)),
            }
        )

    edges.sort(
        key=lambda edge: (
            edge["source"],
            edge["target"],
        )
    )

    return {
        "schema_version": ATLAS_SCHEMA_VERSION,
        "cutoff_year": cutoff_year,
        "metadata": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "coordinate_system": "graph_derived_2d",
            "community_semantics": "data_derived_network_community",
            "score_semantics": None,
        },
        "nodes": nodes,
        "edges": edges,
    }
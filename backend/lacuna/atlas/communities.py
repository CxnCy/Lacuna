"""Deterministic network-community detection for Atlas states."""

from __future__ import annotations

from collections.abc import Hashable

import networkx as nx


def detect_communities(
    graph: nx.Graph,
    *,
    seed: int = 42,
) -> dict[Hashable, int]:
    """
    Assign every node to a data-derived network community.

    Communities are structural groupings in the topic co-occurrence graph.
    They must not be interpreted as OpenAlex domains, fields, or subfields.
    """
    if graph.number_of_nodes() == 0:
        return {}

    if graph.number_of_edges() == 0:
        ordered_nodes = sorted(graph.nodes(), key=str)
        return {node: index for index, node in enumerate(ordered_nodes)}

    communities = nx.community.louvain_communities(
        graph,
        weight="weight",
        seed=seed,
    )

    # Louvain community numbering itself has no scientific meaning.
    # Sort communities by their lexicographically smallest node ID so the
    # exported integer identifiers remain reproducible.
    ordered = sorted(
        (set(community) for community in communities),
        key=lambda community: min(str(node) for node in community),
    )

    assignments: dict[Hashable, int] = {}

    for community_id, community in enumerate(ordered):
        for node in community:
            assignments[node] = community_id

    return assignments
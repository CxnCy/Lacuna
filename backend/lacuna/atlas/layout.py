"""Reproducible graph-derived spatial layouts for the Atlas."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

import networkx as nx
import numpy as np


Position = tuple[float, float]


def _normalise_positions(
    positions: Mapping[Hashable, np.ndarray],
) -> dict[Hashable, Position]:
    """Scale positions into a stable [-1, 1] coordinate range."""
    if not positions:
        return {}

    nodes = list(positions)
    coordinates = np.asarray(
        [positions[node] for node in nodes],
        dtype=float,
    )

    coordinates -= coordinates.mean(axis=0)

    max_abs = float(np.abs(coordinates).max())

    if max_abs > 0:
        coordinates /= max_abs

    return {
        node: (float(x), float(y))
        for node, (x, y) in zip(nodes, coordinates, strict=True)
    }


def compute_layout(
    graph: nx.Graph,
    *,
    seed: int = 42,
    iterations: int = 100,
) -> dict[Hashable, Position]:
    """
    Compute a deterministic spatial layout from one historical graph.

    Only topology present in the supplied graph contributes to the layout.
    No future graph state is consulted.
    """
    if graph.number_of_nodes() == 0:
        return {}

    ordered_nodes = sorted(graph.nodes(), key=str)

    if graph.number_of_nodes() == 1:
        return {ordered_nodes[0]: (0.0, 0.0)}

    # Rebuild in deterministic insertion order before applying spring layout.
    ordered_graph = nx.Graph()
    ordered_graph.add_nodes_from(
        (node, dict(graph.nodes[node]))
        for node in ordered_nodes
    )

    ordered_edges = sorted(
        graph.edges(data=True),
        key=lambda edge: tuple(
            sorted((str(edge[0]), str(edge[1])))
        ),
    )
    ordered_graph.add_edges_from(ordered_edges)

    positions = nx.spring_layout(
        ordered_graph,
        seed=seed,
        weight="weight",
        iterations=iterations,
        dim=2,
    )

    return _normalise_positions(positions)
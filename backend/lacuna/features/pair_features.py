from __future__ import annotations

from typing import Any

import networkx as nx


def _weighted_degree(graph: nx.Graph, node: Any) -> float:
    """Return the sum of incident edge weights for a node."""
    return float(graph.degree(node, weight="weight"))


def compute_pair_features(
    graph: nx.Graph,
    topic_a: Any,
    topic_b: Any,
) -> dict[str, float | int]:
    """
    Compute structural features for one candidate topic pair.

    The supplied graph must already represent the historical snapshot at
    cutoff T. This function uses only information contained in that graph.

    Candidate pairs are expected to be missing edges at cutoff T.
    """

    if topic_a not in graph:
        raise ValueError(f"Topic A is not present in graph: {topic_a}")

    if topic_b not in graph:
        raise ValueError(f"Topic B is not present in graph: {topic_b}")

    if topic_a == topic_b:
        raise ValueError("Candidate pair must contain two distinct topics.")

    if graph.has_edge(topic_a, topic_b):
        raise ValueError(
            "Candidate pair already has a direct edge in the cutoff graph."
        )

    common_neighbours = list(nx.common_neighbors(graph, topic_a, topic_b))
    common_neighbour_count = len(common_neighbours)

    jaccard_score = next(
        nx.jaccard_coefficient(graph, [(topic_a, topic_b)])
    )[2]

    adamic_adar_score = next(
        nx.adamic_adar_index(graph, [(topic_a, topic_b)])
    )[2]

    preferential_attachment_score = next(
        nx.preferential_attachment(graph, [(topic_a, topic_b)])
    )[2]

    degree_a = graph.degree(topic_a)
    degree_b = graph.degree(topic_b)

    weighted_degree_a = _weighted_degree(graph, topic_a)
    weighted_degree_b = _weighted_degree(graph, topic_b)

    return {
        "common_neighbour_count": common_neighbour_count,
        "jaccard_coefficient": float(jaccard_score),
        "adamic_adar_score": float(adamic_adar_score),
        "preferential_attachment": int(preferential_attachment_score),
        "degree_a": int(degree_a),
        "degree_b": int(degree_b),
        "weighted_degree_a": weighted_degree_a,
        "weighted_degree_b": weighted_degree_b,
    }
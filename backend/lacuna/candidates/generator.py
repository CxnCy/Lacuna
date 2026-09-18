import pickle
from dataclasses import dataclass
from pathlib import Path

import networkx as nx


@dataclass(frozen=True)
class LacunaCandidate:
    """
    A structurally plausible candidate relationship between
    two topics in a historical knowledge graph.

    Candidate generation does not imply that the pair is a
    confirmed lacuna or that it will emerge in the future.
    """

    topic_a: str
    topic_b: str
    common_neighbor_count: int


def generate_distance_2_candidates(
    graph: nx.Graph,
) -> list[LacunaCandidate]:
    """
    Generate candidate topic pairs that:

    1. Are not directly connected.
    2. Share at least one common neighbour.

    The supplied graph is assumed to represent a historical
    snapshot at a specific cutoff. This function uses only
    information contained in that graph.

    Each undirected topic pair is returned exactly once.
    """

    if graph.is_directed():
        raise ValueError(
            "Candidate generation requires an undirected topic graph."
        )

    candidate_pairs: dict[
        tuple[str, str],
        set[str],
    ] = {}

    for common_neighbor in graph.nodes:
        neighbors = sorted(graph.neighbors(common_neighbor))

        for index, topic_a in enumerate(neighbors):
            for topic_b in neighbors[index + 1 :]:
                if graph.has_edge(topic_a, topic_b):
                    continue

                pair = tuple(sorted((topic_a, topic_b)))

                if pair not in candidate_pairs:
                    candidate_pairs[pair] = set()

                candidate_pairs[pair].add(common_neighbor)

    candidates = [
        LacunaCandidate(
            topic_a=topic_a,
            topic_b=topic_b,
            common_neighbor_count=len(common_neighbors),
        )
        for (
            topic_a,
            topic_b,
        ), common_neighbors in candidate_pairs.items()
    ]

    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.common_neighbor_count,
            candidate.topic_a,
            candidate.topic_b,
        ),
    )


def save_candidates(
    candidates: list[LacunaCandidate],
    path: str | Path,
) -> Path:
    """
    Persist a generated candidate set to disk.

    Parent directories are created automatically.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as file:
        pickle.dump(candidates, file)

    return output_path


def load_candidates(
    path: str | Path,
) -> list[LacunaCandidate]:
    """
    Load a previously persisted candidate set.
    """

    input_path = Path(path)

    with input_path.open("rb") as file:
        candidates = pickle.load(file)

    return candidates
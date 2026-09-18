from itertools import combinations
from pathlib import Path
import pickle

import networkx as nx


def build_topic_graph(
    works: list[dict],
    cutoff_year: int,
) -> nx.Graph:
    """
    Build a weighted topic co-publication graph using only works
    published at or before the cutoff year.

    Nodes represent OpenAlex topics.
    Edges represent topic co-occurrence within works.
    Edge weight counts the number of co-publications.
    """
    graph = nx.Graph(cutoff_year=cutoff_year)

    for work in works:
        publication_year = work.get("publication_year")

        if publication_year is None or publication_year > cutoff_year:
            continue

        topics = work.get("topics", [])

        # Deduplicate topics within a single work by topic ID.
        unique_topics = {
            topic["id"]: topic
            for topic in topics
            if topic.get("id")
        }

        # Add topic nodes and preserve available metadata.
        for topic_id, topic in unique_topics.items():
            graph.add_node(
                topic_id,
                name=topic.get("name"),
                hierarchy=topic.get("hierarchy"),
            )

        # Every pair of topics in the same work contributes
        # one unit to their co-publication relationship.
        for topic_a, topic_b in combinations(sorted(unique_topics), 2):
            if graph.has_edge(topic_a, topic_b):
                graph[topic_a][topic_b]["weight"] += 1
            else:
                graph.add_edge(
                    topic_a,
                    topic_b,
                    weight=1,
                )

    return graph


def build_temporal_snapshots(
    works: list[dict],
    cutoff_years: list[int],
) -> dict[int, nx.Graph]:
    """
    Build cumulative topic graph snapshots for multiple
    historical cutoff years.

    Each snapshot contains only information from works
    published at or before its cutoff year.
    """
    snapshots = {}

    for cutoff_year in sorted(set(cutoff_years)):
        snapshots[cutoff_year] = build_topic_graph(
            works=works,
            cutoff_year=cutoff_year,
        )

    return snapshots


def save_temporal_snapshots(
    snapshots: dict[int, nx.Graph],
    output_dir: str | Path,
) -> None:
    """
    Save temporal graph snapshots as pickle files.

    Pickle preserves NetworkX graph structure and Python metadata,
    including nested topic hierarchy dictionaries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for cutoff_year, graph in snapshots.items():
        file_path = output_path / f"topic_graph_{cutoff_year}.pkl"

        with file_path.open("wb") as file:
            pickle.dump(graph, file)


def load_temporal_snapshot(
    file_path: str | Path,
) -> nx.Graph:
    """
    Load a previously saved temporal graph snapshot.
    """
    path = Path(file_path)

    with path.open("rb") as file:
        graph = pickle.load(file)

    return graph
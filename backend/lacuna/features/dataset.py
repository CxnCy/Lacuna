from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from lacuna.candidates.generator import LacunaCandidate
from lacuna.features.pair_features import compute_pair_features


def build_feature_rows(
    graph: nx.Graph,
    candidates: Iterable[LacunaCandidate],
    cutoff_year: int,
) -> list[dict[str, Any]]:
    """
    Build ML feature rows for candidate topic pairs at a historical cutoff.

    The graph must represent only information available at or before
    cutoff_year.

    This function constructs historical features only. It does not inspect
    future data and does not construct outcome labels.
    """
    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        if not isinstance(candidate, LacunaCandidate):
            raise TypeError(
                "Each candidate must be a LacunaCandidate instance."
            )

        features = compute_pair_features(
            graph=graph,
            topic_a=candidate.topic_a,
            topic_b=candidate.topic_b,
        )

        row = {
            "cutoff_year": cutoff_year,
            "topic_a": candidate.topic_a,
            "topic_b": candidate.topic_b,
            **features,
        }

        rows.append(row)

    return rows


def save_feature_rows(
    rows: list[dict[str, Any]],
    path: str | Path,
) -> None:
    """Persist historical feature rows to a pickle artifact."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as file:
        pickle.dump(rows, file)


def load_feature_rows(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Load historical feature rows from a pickle artifact."""
    input_path = Path(path)

    with input_path.open("rb") as file:
        rows = pickle.load(file)

    return rows
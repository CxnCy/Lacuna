from __future__ import annotations

from collections import Counter
from typing import Any

from lacuna.candidates.generator import LacunaCandidate


def build_future_labels(
    candidates: list[LacunaCandidate],
    works: list[dict[str, Any]],
    cutoff_year: int,
    prediction_window_years: int = 5,
) -> list[dict[str, Any]]:
    """
    Construct future outcome labels for historical candidate pairs.

    Features and candidates are defined at cutoff_year.
    Only works strictly after cutoff_year and within the prediction
    window are used to construct labels.

    MVP operational definition:
        label = 1 if the candidate pair co-occurs in at least one work
        during the future prediction window.
        label = 0 otherwise.

    This definition is provisional and intended for development-pipeline
    validation on the small development corpus.
    """
    start_year = cutoff_year + 1
    end_year = cutoff_year + prediction_window_years

    candidate_pairs = {
        tuple(sorted((candidate.topic_a, candidate.topic_b)))
        for candidate in candidates
    }

    future_counts: Counter[tuple[str, str]] = Counter()

    for work in works:
        year = work.get("publication_year")

        if year is None or not (start_year <= year <= end_year):
            continue

        topics = work.get("topics", [])

        topic_ids = sorted({
            topic["id"]
            for topic in topics
            if isinstance(topic, dict) and topic.get("id")
        })

        for index, topic_a in enumerate(topic_ids):
            for topic_b in topic_ids[index + 1:]:
                pair = (topic_a, topic_b)

                if pair in candidate_pairs:
                    future_counts[pair] += 1

    labels = []

    for candidate in candidates:
        pair = tuple(sorted(
            (candidate.topic_a, candidate.topic_b)
        ))

        future_co_publications = future_counts.get(pair, 0)

        labels.append({
            "cutoff_year": cutoff_year,
            "topic_a": candidate.topic_a,
            "topic_b": candidate.topic_b,
            "prediction_window_start": start_year,
            "prediction_window_end": end_year,
            "future_co_publication_count": future_co_publications,
            "label": int(future_co_publications >= 1),
        })

    return labels
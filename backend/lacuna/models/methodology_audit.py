"""Methodology audits for Lacuna emergence labels and candidate generation."""

from __future__ import annotations

from typing import Any

import pandas as pd


REQUIRED_LABEL_COLUMNS = [
    "label",
    "future_co_publication_count",
]

REQUIRED_CANDIDATE_COLUMNS = [
    "topic_a",
    "topic_b",
]


def audit_emergence_labels(
    dataset: pd.DataFrame,
    cutoff_year: int,
) -> dict[str, Any]:
    """Audit the current emergence-label distribution without modifying it."""
    missing = [
        column
        for column in REQUIRED_LABEL_COLUMNS
        if column not in dataset.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing required label-audit columns: {missing}"
        )

    labels = dataset["label"].astype(int)

    positives = dataset.loc[
        labels == 1
    ].copy()

    total_candidates = len(dataset)
    positive_count = len(positives)
    negative_count = total_candidates - positive_count

    positive_rate = (
        positive_count / total_candidates
        if total_candidates
        else 0.0
    )

    if positive_count:
        future_counts = positives[
            "future_co_publication_count"
        ]

        single_future_count = int(
            (future_counts == 1).sum()
        )

        single_future_share = (
            single_future_count
            / positive_count
        )

        future_median = float(
            future_counts.median()
        )

        future_mean = float(
            future_counts.mean()
        )

        future_min = float(
            future_counts.min()
        )

        future_max = float(
            future_counts.max()
        )

        at_least_two = int(
            (future_counts >= 2).sum()
        )

        at_least_three = int(
            (future_counts >= 3).sum()
        )

        at_least_five = int(
            (future_counts >= 5).sum()
        )

    else:
        single_future_count = 0
        single_future_share = 0.0
        future_median = 0.0
        future_mean = 0.0
        future_min = 0.0
        future_max = 0.0
        at_least_two = 0
        at_least_three = 0
        at_least_five = 0

    return {
        "cutoff_year": cutoff_year,
        "candidate_count": total_candidates,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": positive_rate,
        "single_future_co_publication_count": single_future_count,
        "single_future_co_publication_share": single_future_share,
        "future_co_publication_median_positive": future_median,
        "future_co_publication_mean_positive": future_mean,
        "future_co_publication_min_positive": future_min,
        "future_co_publication_max_positive": future_max,
        "positive_pairs_future_count_ge_2": at_least_two,
        "positive_pairs_future_count_ge_3": at_least_three,
        "positive_pairs_future_count_ge_5": at_least_five,
    }


def audit_prediction_windows(
    dataset: pd.DataFrame,
    cutoff_year: int,
) -> dict[str, Any]:
    """Audit temporal prediction-window consistency."""
    required = [
        "prediction_window_start",
        "prediction_window_end",
    ]

    missing = [
        column
        for column in required
        if column not in dataset.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing prediction-window columns: {missing}"
        )

    starts = sorted(
        dataset[
            "prediction_window_start"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    ends = sorted(
        dataset[
            "prediction_window_end"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    expected_start = cutoff_year + 1

    start_consistent = (
        len(starts) == 1
        and int(starts[0]) == expected_start
    )

    single_window = (
        len(starts) == 1
        and len(ends) == 1
    )

    return {
        "cutoff_year": cutoff_year,
        "prediction_window_starts": str(starts),
        "prediction_window_ends": str(ends),
        "expected_window_start": expected_start,
        "start_consistent_with_cutoff": start_consistent,
        "single_prediction_window": single_window,
    }


def audit_candidate_dataset(
    candidates: pd.DataFrame,
    cutoff_year: int,
) -> dict[str, Any]:
    """Audit candidate-pair integrity without changing candidate generation."""
    missing = [
        column
        for column in REQUIRED_CANDIDATE_COLUMNS
        if column not in candidates.columns
    ]

    if missing:
        raise ValueError(
            f"Candidate dataset is missing required columns: {missing}"
        )

    pair_frame = candidates[
        ["topic_a", "topic_b"]
    ].copy()

    self_pairs = int(
        (
            pair_frame["topic_a"]
            == pair_frame["topic_b"]
        ).sum()
    )

    canonical_pairs = pair_frame.apply(
        lambda row: tuple(
            sorted(
                (
                    str(row["topic_a"]),
                    str(row["topic_b"]),
                )
            )
        ),
        axis=1,
    )

    duplicate_pairs = int(
        canonical_pairs.duplicated().sum()
    )

    unique_topics = set(
        pair_frame["topic_a"]
    ).union(
        set(pair_frame["topic_b"])
    )

    return {
        "cutoff_year": cutoff_year,
        "candidate_count": len(candidates),
        "unique_candidate_topics": len(unique_topics),
        "self_pair_count": self_pairs,
        "duplicate_pair_count": duplicate_pairs,
    }


def build_temporal_candidate_summary(
    label_audits: list[dict[str, Any]],
    candidate_audits: list[dict[str, Any]],
    graph_stats: list[dict[str, Any]],
) -> pd.DataFrame:
    """Combine temporal candidate, label, and graph statistics."""
    label_df = pd.DataFrame(
        label_audits
    )

    candidate_df = pd.DataFrame(
        candidate_audits
    )

    graph_df = pd.DataFrame(
        graph_stats
    )

    summary = label_df.merge(
        candidate_df[
            [
                "cutoff_year",
                "unique_candidate_topics",
                "self_pair_count",
                "duplicate_pair_count",
            ]
        ],
        on="cutoff_year",
        how="left",
    )

    summary = summary.merge(
        graph_df,
        on="cutoff_year",
        how="left",
    )

    if "graph_node_count" in summary.columns:
        summary[
            "candidates_per_graph_node"
        ] = (
            summary["candidate_count"]
            / summary["graph_node_count"]
        )

    if "graph_edge_count" in summary.columns:
        summary[
            "candidates_per_graph_edge"
        ] = (
            summary["candidate_count"]
            / summary["graph_edge_count"]
        )

    summary = summary.sort_values(
        "cutoff_year"
    ).reset_index(
        drop=True
    )

    summary[
        "candidate_count_change"
    ] = summary[
        "candidate_count"
    ].diff()

    summary[
        "positive_count_change"
    ] = summary[
        "positive_count"
    ].diff()

    summary[
        "positive_rate_change"
    ] = summary[
        "positive_rate"
    ].diff()

    return summary


def build_methodology_flags(
    temporal_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create diagnostic flags requiring methodological investigation."""
    flags: list[dict[str, Any]] = []

    for _, row in temporal_summary.iterrows():
        cutoff_year = int(
            row["cutoff_year"]
        )

        if (
            row[
                "single_future_co_publication_share"
            ]
            >= 0.5
        ):
            flags.append(
                {
                    "cutoff_year": cutoff_year,
                    "audit_area": "emergence_label",
                    "severity": "high",
                    "flag": (
                        "Most positive labels are supported by "
                        "exactly one future co-publication."
                    ),
                }
            )

        if (
            row["positive_rate"]
            < 0.01
        ):
            flags.append(
                {
                    "cutoff_year": cutoff_year,
                    "audit_area": "class_balance",
                    "severity": "high",
                    "flag": (
                        "Positive prevalence is below 1%."
                    ),
                }
            )

        if (
            row.get(
                "self_pair_count",
                0,
            )
            > 0
        ):
            flags.append(
                {
                    "cutoff_year": cutoff_year,
                    "audit_area": "candidate_integrity",
                    "severity": "high",
                    "flag": (
                        "Candidate generation contains self-pairs."
                    ),
                }
            )

        if (
            row.get(
                "duplicate_pair_count",
                0,
            )
            > 0
        ):
            flags.append(
                {
                    "cutoff_year": cutoff_year,
                    "audit_area": "candidate_integrity",
                    "severity": "high",
                    "flag": (
                        "Candidate generation contains duplicate "
                        "unordered topic pairs."
                    ),
                }
            )

    if len(
        temporal_summary
    ) > 1:
        ordered = temporal_summary.sort_values(
            "cutoff_year"
        )

        candidate_increasing = (
            ordered[
                "candidate_count"
            ]
            .diff()
            .dropna()
            > 0
        ).all()

        positive_decreasing = (
            ordered[
                "positive_count"
            ]
            .diff()
            .dropna()
            < 0
        ).all()

        if (
            candidate_increasing
            and positive_decreasing
        ):
            flags.append(
                {
                    "cutoff_year": "temporal",
                    "audit_area": "candidate_label_drift",
                    "severity": "high",
                    "flag": (
                        "Candidate counts increase over time while "
                        "positive counts decrease."
                    ),
                }
            )

    return pd.DataFrame(
        flags,
        columns=[
            "cutoff_year",
            "audit_area",
            "severity",
            "flag",
        ],
    )
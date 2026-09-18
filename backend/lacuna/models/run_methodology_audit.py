"""Run Lacuna Step 7.3 and 7.4 methodology audits."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

from lacuna.models.methodology_audit import (
    audit_candidate_dataset,
    audit_emergence_labels,
    audit_prediction_windows,
    build_methodology_flags,
    build_temporal_candidate_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

ML_DIR = (
    PROCESSED_DIR
    / "ml"
)

CANDIDATE_DIR = (
    PROCESSED_DIR
    / "candidates"
)

GRAPH_DIR = (
    PROCESSED_DIR
    / "graphs"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "analysis"
    / "methodology_audit"
)

CUTOFF_YEARS = [
    2010,
    2015,
    2020,
]


def load_pickle_dataframe(
    path: Path,
) -> pd.DataFrame:
    """Load a pickle artifact as a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}"
        )

    data = pd.read_pickle(path)

    if isinstance(data, list):
        data = pd.DataFrame(data)

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"Expected DataFrame or list at {path}, "
            f"found {type(data)}"
        )

    return data


def load_graph(
    path: Path,
) -> nx.Graph:
    """Load a NetworkX graph pickle."""
    if not path.exists():
        raise FileNotFoundError(
            f"Graph artifact not found: {path}"
        )

    graph = pd.read_pickle(path)

    if not isinstance(graph, nx.Graph):
        raise TypeError(
            f"Expected NetworkX graph at {path}, "
            f"found {type(graph)}"
        )

    return graph


def save_dataframe(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    """Save an audit table as CSV."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / filename
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )


def run_methodology_audit() -> None:
    """Execute Step 7.3 and Step 7.4 audits."""
    label_audits = []
    candidate_audits = []
    window_audits = []
    graph_stats = []

    print("=" * 72)
    print(
        "LACUNA — STEP 7.3 + 7.4 METHODOLOGY AUDIT"
    )
    print("=" * 72)

    for year in CUTOFF_YEARS:
        print()
        print(
            f"Auditing cutoff: {year}"
        )
        print("-" * 72)

        ml_path = (
            ML_DIR
            / f"lacuna_ml_dataset_{year}.pkl"
        )

        candidate_path = (
            CANDIDATE_DIR
            / f"lacuna_candidates_{year}.pkl"
        )

        graph_path = (
            GRAPH_DIR
            / f"topic_graph_{year}.pkl"
        )

        dataset = load_pickle_dataframe(
            ml_path
        )

        candidates = load_pickle_dataframe(
            candidate_path
        )

        graph = load_graph(
            graph_path
        )

        label_audit = (
            audit_emergence_labels(
                dataset,
                year,
            )
        )

        candidate_audit = (
            audit_candidate_dataset(
                candidates,
                year,
            )
        )

        window_audit = (
            audit_prediction_windows(
                dataset,
                year,
            )
        )

        graph_stat = {
            "cutoff_year": year,
            "graph_node_count": (
                graph.number_of_nodes()
            ),
            "graph_edge_count": (
                graph.number_of_edges()
            ),
        }

        label_audits.append(
            label_audit
        )

        candidate_audits.append(
            candidate_audit
        )

        window_audits.append(
            window_audit
        )

        graph_stats.append(
            graph_stat
        )

        print(
            "Candidates: "
            f"{label_audit['candidate_count']}"
        )

        print(
            "Positives: "
            f"{label_audit['positive_count']}"
        )

        print(
            "Positive rate: "
            f"{label_audit['positive_rate']:.4%}"
        )

        print(
            "Single-publication positive share: "
            f"{label_audit['single_future_co_publication_share']:.4%}"
        )

        print(
            "Future co-publications among positives: "
            f"median="
            f"{label_audit['future_co_publication_median_positive']:.2f} | "
            f"mean="
            f"{label_audit['future_co_publication_mean_positive']:.2f}"
        )

        print(
            "Graph: "
            f"{graph_stat['graph_node_count']} nodes | "
            f"{graph_stat['graph_edge_count']} edges"
        )

        print(
            "Candidate integrity: "
            f"{candidate_audit['self_pair_count']} self-pairs | "
            f"{candidate_audit['duplicate_pair_count']} duplicates"
        )

        print(
            "Prediction window consistent: "
            f"{window_audit['start_consistent_with_cutoff']}"
        )

    label_df = pd.DataFrame(
        label_audits
    )

    candidate_df = pd.DataFrame(
        candidate_audits
    )

    window_df = pd.DataFrame(
        window_audits
    )

    graph_df = pd.DataFrame(
        graph_stats
    )

    temporal_summary = (
        build_temporal_candidate_summary(
            label_audits,
            candidate_audits,
            graph_stats,
        )
    )

    flags = build_methodology_flags(
        temporal_summary
    )

    save_dataframe(
        label_df,
        "emergence_label_audit.csv",
    )

    save_dataframe(
        candidate_df,
        "candidate_generation_audit.csv",
    )

    save_dataframe(
        window_df,
        "prediction_window_audit.csv",
    )

    save_dataframe(
        graph_df,
        "graph_growth_audit.csv",
    )

    save_dataframe(
        temporal_summary,
        "temporal_methodology_summary.csv",
    )

    save_dataframe(
        flags,
        "methodology_flags.csv",
    )

    print()
    print("=" * 72)
    print("METHODOLOGY FLAGS")
    print("=" * 72)

    if flags.empty:
        print(
            "No automatic diagnostic flags raised."
        )
    else:
        for _, row in flags.iterrows():
            print(
                f"[{row['severity'].upper()}] "
                f"{row['cutoff_year']} | "
                f"{row['audit_area']} | "
                f"{row['flag']}"
            )

    print()
    print("=" * 72)
    print(
        "STEP 7.3 + 7.4 AUDIT COMPLETE"
    )
    print(
        f"Artifacts: {OUTPUT_DIR}"
    )
    print("=" * 72)


if __name__ == "__main__":
    run_methodology_audit()
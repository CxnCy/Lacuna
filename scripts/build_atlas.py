"""Build frontend-ready temporal Atlas artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from lacuna.atlas.export import export_atlas_states
from lacuna.network.temporal_graph import load_temporal_snapshot


DEFAULT_CUTOFFS = (2010, 2015, 2020, 2025)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Lacuna temporal Atlas artifacts."
    )

    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path("data/processed/graphs"),
        help="Directory containing topic_graph_<year>.pkl snapshots.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/atlas"),
        help="Directory for frontend-ready Atlas JSON artifacts.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible Atlas layouts.",
    )

    parser.add_argument(
        "--layout-iterations",
        type=int,
        default=100,
        help="Number of iterations used by the graph layout algorithm.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    graphs = {}

    for year in DEFAULT_CUTOFFS:
        graph_path = args.graph_dir / f"topic_graph_{year}.pkl"

        if not graph_path.exists():
            raise FileNotFoundError(
                f"Missing temporal graph snapshot: {graph_path}"
            )

        graphs[year] = load_temporal_snapshot(graph_path)

    manifest = export_atlas_states(
        graphs,
        args.output_dir,
        seed=args.seed,
        layout_iterations=args.layout_iterations,
    )

    print()
    print("LACUNA — ATLAS BUILD")
    print("=" * 60)

    for state in manifest["states"]:
        print(
            f"{state['cutoff_year']}: "
            f"{state['node_count']} topics | "
            f"{state['edge_count']} relationships"
        )

    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print("Atlas build complete.")


if __name__ == "__main__":
    main()
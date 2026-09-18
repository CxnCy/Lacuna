"""Atlas artifact generation and JSON export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from .communities import detect_communities
from .contract import ATLAS_SCHEMA_VERSION, build_atlas_state
from .layout import compute_layout
from .temporal_alignment import align_temporal_layouts


def build_atlas_states(
    graphs: dict[int, nx.Graph],
    *,
    seed: int = 42,
    layout_iterations: int = 100,
) -> dict[int, dict[str, Any]]:
    """Build frontend-ready Atlas states from temporal graphs."""
    if not graphs:
        return {}

    layouts = {
        year: compute_layout(
            graph,
            seed=seed,
            iterations=layout_iterations,
        )
        for year, graph in sorted(graphs.items())
    }

    aligned_layouts = align_temporal_layouts(layouts)

    states: dict[int, dict[str, Any]] = {}

    for year, graph in sorted(graphs.items()):
        communities = detect_communities(
            graph,
            seed=seed,
        )

        states[year] = build_atlas_state(
            graph,
            cutoff_year=year,
            positions=aligned_layouts[year],
            communities=communities,
        )

    return states


def export_atlas_states(
    graphs: dict[int, nx.Graph],
    output_dir: str | Path,
    *,
    seed: int = 42,
    layout_iterations: int = 100,
) -> dict[str, Any]:
    """Build and export temporal Atlas JSON artifacts plus a manifest."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    states = build_atlas_states(
        graphs,
        seed=seed,
        layout_iterations=layout_iterations,
    )

    manifest_states: list[dict[str, Any]] = []

    for year, state in sorted(states.items()):
        filename = f"atlas_{year}.json"
        output_path = destination / filename

        output_path.write_text(
            json.dumps(
                state,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        manifest_states.append(
            {
                "cutoff_year": year,
                "file": filename,
                "node_count": state["metadata"]["node_count"],
                "edge_count": state["metadata"]["edge_count"],
            }
        )

    manifest = {
        "schema_version": ATLAS_SCHEMA_VERSION,
        "atlas_type": "temporal_scientific_knowledge_network",
        "coordinate_system": "graph_derived_2d",
        "layout_policy": (
            "cutoff_safe_independent_layouts_sequentially_aligned"
        ),
        "community_semantics": "data_derived_network_community",
        "prediction_overlay_included": False,
        "states": manifest_states,
    }

    (destination / "atlas_manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return manifest
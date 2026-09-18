"""Temporal alignment of independently computed Atlas layouts."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

import numpy as np

from .layout import Position


def align_layout(
    reference: Mapping[Hashable, Position],
    target: Mapping[Hashable, Position],
) -> dict[Hashable, Position]:
    """
    Align a target layout to a previous historical layout using shared nodes.

    The target layout must already have been computed independently from its
    own cutoff-safe graph. Alignment applies only a rigid Procrustes-style
    rotation/reflection plus translation; it does not introduce future edges
    or future graph topology into the historical state.
    """
    if not target:
        return {}

    shared_nodes = sorted(
        set(reference).intersection(target),
        key=str,
    )

    if not shared_nodes:
        return dict(target)

    reference_points = np.asarray(
        [reference[node] for node in shared_nodes],
        dtype=float,
    )
    target_points = np.asarray(
        [target[node] for node in shared_nodes],
        dtype=float,
    )

    reference_centroid = reference_points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)

    if len(shared_nodes) == 1:
        offset = reference_centroid - target_centroid

        return {
            node: tuple(
                float(value)
                for value in np.asarray(position, dtype=float) + offset
            )
            for node, position in target.items()
        }

    reference_centered = reference_points - reference_centroid
    target_centered = target_points - target_centroid

    covariance = target_centered.T @ reference_centered
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    rotation = u_matrix @ vt_matrix

    aligned: dict[Hashable, Position] = {}

    for node, position in target.items():
        point = np.asarray(position, dtype=float)
        transformed = (
            (point - target_centroid) @ rotation
            + reference_centroid
        )
        aligned[node] = (
            float(transformed[0]),
            float(transformed[1]),
        )

    return aligned


def align_temporal_layouts(
    layouts: Mapping[int, Mapping[Hashable, Position]],
) -> dict[int, dict[Hashable, Position]]:
    """
    Sequentially align temporal layouts.

    Each layout is assumed to have been independently computed from its own
    historical graph before this function is called.
    """
    if not layouts:
        return {}

    years = sorted(layouts)

    aligned: dict[int, dict[Hashable, Position]] = {
        years[0]: dict(layouts[years[0]])
    }

    for previous_year, current_year in zip(
        years,
        years[1:],
        strict=False,
    ):
        aligned[current_year] = align_layout(
            aligned[previous_year],
            layouts[current_year],
        )

    return aligned
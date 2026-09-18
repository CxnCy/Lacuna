"""Persistence utilities for Lacuna model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib


def save_model_artifact(
    model,
    output_path: str | Path,
    metadata: Dict[str, Any],
) -> Path:
    """Persist a fitted model together with reproducibility metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "metadata": metadata,
    }

    joblib.dump(artifact, output_path)
    return output_path


def load_model_artifact(path: str | Path):
    """Load a previously persisted Lacuna model artifact."""
    return joblib.load(Path(path))

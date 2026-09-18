from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Generated artifact directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
GRAPHS_DIR = ARTIFACTS_DIR / "graphs"
DATASETS_DIR = ARTIFACTS_DIR / "datasets"
MODELS_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ARTIFACTS_DIR / "results"
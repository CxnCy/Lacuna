from lacuna.core.config import (
    PROJECT_ROOT,
    DATA_DIR,
    ARTIFACTS_DIR,
)
from lacuna.core.logging import get_logger

def test_project_directories():
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert ARTIFACTS_DIR == PROJECT_ROOT / "artifacts"
    from lacuna.core.logging import get_logger


def test_logger():
    logger = get_logger("lacuna.test")
    assert logger.name == "lacuna.test"
"""Project-wide path helpers for NeuroScan AI."""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLIT_DATA_DIR = DATA_DIR / "splits"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
SAMPLES_DIR = ARTIFACTS_DIR / "samples"
SEGMENTATION_DIR = ARTIFACTS_DIR / "segmentation"
MODELS_DIR = ROOT_DIR / "models"
MODEL_EXPORT_DIR = MODELS_DIR / "exports"
MODEL_CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
LOGS_DIR = ROOT_DIR / "logs"
TENSORBOARD_DIR = ROOT_DIR / "tensorboard"


def ensure_project_dirs() -> None:
    """Create known project directories if they do not already exist."""
    for path in [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SPLIT_DATA_DIR,
        ARTIFACTS_DIR,
        PLOTS_DIR,
        REPORTS_DIR,
        SAMPLES_DIR,
        SEGMENTATION_DIR,
        MODELS_DIR,
        MODEL_EXPORT_DIR,
        MODEL_CHECKPOINT_DIR,
        LOGS_DIR,
        TENSORBOARD_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

"""CLI training entrypoint for NeuroScan AI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database import register_model_version
from models import TrainingConfig, train_all_models
from utils import MODEL_EXPORT_DIR, configure_logging, ensure_project_dirs


LOGGER = configure_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NeuroScan AI brain tumour models")
    parser.add_argument("--dataset_zip", type=str, default=None, help="Path to MRI dataset zip archive (optional if data/raw/archive exists)")
    parser.add_argument("--stage1_epochs", type=int, default=5)
    parser.add_argument("--stage2_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    summary = train_all_models(
        TrainingConfig(
            dataset_zip=args.dataset_zip,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            resume=args.resume,
        )
    )

    best_model = summary["best_model"]
    metadata = {
        "model_name": best_model["model_name"],
        "model_version": "1.0.0",
        "model_path": best_model["model_path"],
        "accuracy": best_model["accuracy"],
        "precision": best_model["precision"],
        "recall": best_model["recall"],
        "f1_score": best_model["f1_score"],
    }
    Path(MODEL_EXPORT_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    register_model_version(
        best_model["model_name"],
        "1.0.0",
        best_model["model_path"],
        best_model["accuracy"],
        best_model["precision"],
        best_model["recall"],
        best_model["f1_score"],
    )
    LOGGER.info("Training complete. Best model: %s", best_model)


if __name__ == "__main__":
    main()

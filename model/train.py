"""CLI training entrypoint for NeuroScan AI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from database import register_model_version
from models import TrainingConfig, train_all_models
from utils import MODEL_EXPORT_DIR, configure_logging, ensure_project_dirs


LOGGER = configure_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NeuroScan AI brain tumour models")
    parser.add_argument("--dataset_zip", required=True, help="Path to MRI dataset zip archive")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    summary = train_all_models(
        TrainingConfig(
            dataset_zip=args.dataset_zip,
            epochs=args.epochs,
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

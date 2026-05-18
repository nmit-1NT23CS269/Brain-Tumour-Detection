"""Training pipeline for NeuroScan AI multi-model experiments."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from preprocessing import prepare_dataset_from_zip
from utils import (
    MODEL_CHECKPOINT_DIR,
    MODEL_EXPORT_DIR,
    PLOTS_DIR,
    TENSORBOARD_DIR,
    configure_logging,
    ensure_project_dirs,
)

from .evaluation import evaluate_predictions, persist_metrics, save_confusion_matrix, save_training_history
from .model_registry import CLASS_NAMES, INPUT_SHAPE, MODEL_SPECS, build_transfer_model


LOGGER = configure_logging(__name__)


@dataclass
class TrainingConfig:
    dataset_zip: str
    epochs: int = 12
    batch_size: int = 16
    patience: int = 4
    learning_rate: float = 1e-4
    resume: bool = False


def _build_generators(batch_size: int):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_dir = Path("data/splits/train")
    val_dir = Path("data/splits/val")
    test_dir = Path("data/splits/test")

    datagen_args = dict(
        target_size=INPUT_SHAPE[:2],
        class_mode="categorical",
        batch_size=batch_size,
    )
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.08,
        zoom_range=0.12,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(train_dir, shuffle=True, **datagen_args)
    val_gen = eval_datagen.flow_from_directory(val_dir, shuffle=False, **datagen_args)
    test_gen = eval_datagen.flow_from_directory(test_dir, shuffle=False, **datagen_args)

    # Enable mixed precision only when supported.
    try:
        from tensorflow.keras import mixed_precision

        gpu_devices = tf.config.list_physical_devices("GPU")
        if gpu_devices:
            mixed_precision.set_global_policy("mixed_float16")
            LOGGER.info("Mixed precision enabled.")
    except Exception as exc:
        LOGGER.warning("Mixed precision not enabled: %s", exc)

    return train_gen, val_gen, test_gen


def _build_callbacks(model_name: str, patience: int):
    import tensorflow as tf

    checkpoint_path = MODEL_CHECKPOINT_DIR / f"{model_name}.keras"
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(Path("logs") / f"{model_name}_training.csv")),
        tf.keras.callbacks.TensorBoard(log_dir=str(TENSORBOARD_DIR / model_name)),
    ]


def train_all_models(config: TrainingConfig) -> dict:
    """Prepare dataset, train all supported models, and choose the best result."""
    ensure_project_dirs()
    dataset_summary = prepare_dataset_from_zip(config.dataset_zip)
    train_gen, val_gen, test_gen = _build_generators(config.batch_size)

    leaderboard: list[dict] = []
    best_run: dict | None = None

    for model_name in MODEL_SPECS:
        LOGGER.info("Training model: %s", model_name)
        model = build_transfer_model(model_name)
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config.epochs,
            callbacks=_build_callbacks(model_name, config.patience),
            verbose=1,
        )

        probabilities = model.predict(test_gen, verbose=0)
        y_pred = probabilities.argmax(axis=1)
        y_true = test_gen.classes
        metrics = evaluate_predictions(y_true, y_pred, CLASS_NAMES)
        metrics["model_name"] = model_name
        metrics["history"] = history.history

        model_export_path = MODEL_EXPORT_DIR / f"{model_name}.keras"
        model.save(model_export_path)
        metrics["model_path"] = str(model_export_path)
        metrics["plots"] = save_training_history(history.history, model_name.lower())
        metrics["confusion_matrix_path"] = save_confusion_matrix(
            metrics, CLASS_NAMES, PLOTS_DIR / f"{model_name.lower()}_confusion_matrix.png"
        )
        metrics_path = Path("logs") / f"{model_name.lower()}_metrics.json"
        persist_metrics(metrics, metrics_path)
        leaderboard.append(metrics)

        if best_run is None or metrics["f1_score"] > best_run["f1_score"]:
            best_run = metrics

    assert best_run is not None
    best_alias_path = MODEL_EXPORT_DIR / "best_model.keras"
    shutil.copy2(best_run["model_path"], best_alias_path)
    best_run["model_path"] = str(best_alias_path)

    summary = {
        "dataset_summary": asdict(dataset_summary),
        "leaderboard": sorted(
            [
                {
                    "model_name": run["model_name"],
                    "accuracy": run["accuracy"],
                    "precision": run["precision"],
                    "recall": run["recall"],
                    "f1_score": run["f1_score"],
                    "model_path": run["model_path"],
                    "confusion_matrix_path": run["confusion_matrix_path"],
                }
                for run in leaderboard
            ],
            key=lambda item: item["f1_score"],
            reverse=True,
        ),
        "best_model": {
            "model_name": best_run["model_name"],
            "accuracy": best_run["accuracy"],
            "precision": best_run["precision"],
            "recall": best_run["recall"],
            "f1_score": best_run["f1_score"],
            "model_path": best_run["model_path"],
        },
    }
    Path("logs/training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

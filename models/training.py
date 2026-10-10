"""Two-stage training pipeline for NeuroScan AI transfer-learning models."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from preprocessing.dataset_manager import (
    CLASS_ALIAS_MAP,
    VALID_EXTENSIONS,
    validate_dataset_images,
)
from utils import (
    MODEL_CHECKPOINT_DIR,
    MODEL_EXPORT_DIR,
    PLOTS_DIR,
    TENSORBOARD_DIR,
    configure_logging,
    ensure_project_dirs,
)

from .evaluation import evaluate_predictions, persist_metrics, save_confusion_matrix, save_training_history
from .model_registry import (
    CLASS_NAMES,
    INPUT_SHAPE,
    MODEL_SPECS,
    build_transfer_model,
    get_backbone_and_preprocess,
    save_class_names_json,
    unfreeze_stage2_model,
)

LOGGER = configure_logging(__name__)


@dataclass
class TrainingConfig:
    dataset_zip: str | None = None
    stage1_epochs: int = 6
    stage2_epochs: int = 10
    batch_size: int = 32
    patience: int = 4
    learning_rate: float = 1e-4
    resume: bool = False


def _build_generators(model_name: str, batch_size: int):
    """BuildImageDataGenerators sharing the backbone's exact preprocess_input function."""
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_dir = Path("data/splits/train")
    val_dir = Path("data/splits/val")
    test_dir = Path("data/splits/test")

    _, preprocess_fn = get_backbone_and_preprocess(model_name)

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.05,
        zoom_range=0.10,
        brightness_range=[0.88, 1.12],
        horizontal_flip=True,
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    datagen_args = dict(
        target_size=INPUT_SHAPE[:2],
        class_mode="categorical",
        batch_size=batch_size,
    )

    train_gen = train_datagen.flow_from_directory(train_dir, shuffle=True, **datagen_args)
    val_gen = eval_datagen.flow_from_directory(val_dir, shuffle=False, **datagen_args)
    test_gen = eval_datagen.flow_from_directory(test_dir, shuffle=False, **datagen_args)

    return train_gen, val_gen, test_gen


def _prepare_splits_from_archive() -> None:
    """Create 80/20 train/val splits from Training folder and keep Testing intact."""
    from sklearn.model_selection import train_test_split

    raw_training_dir = Path("data/raw/archive/Training")
    raw_testing_dir = Path("data/raw/archive/Testing")

    if not raw_training_dir.exists() or not raw_testing_dir.exists():
        LOGGER.warning("Dataset raw directories missing at data/raw/archive.")
        return

    split_train = Path("data/splits/train")
    split_val = Path("data/splits/val")
    split_test = Path("data/splits/test")

    valid_splits = True
    for s in [split_train, split_val, split_test]:
        if not s.exists():
            valid_splits = False
            break
        subdirs = [d.name for d in s.iterdir() if d.is_dir()]
        if set(subdirs) != set(CLASS_NAMES):
            valid_splits = False
            break

    if valid_splits:
        LOGGER.info("Dataset splits with all 4 classes already present at data/splits.")
        return

    for s in [split_train, split_val, split_test]:
        if s.exists():
            shutil.rmtree(s)
        s.mkdir(parents=True, exist_ok=True)

    # Copy Testing directory images to data/splits/test
    for class_folder in raw_testing_dir.iterdir():
        if class_folder.is_dir():
            target_class = CLASS_ALIAS_MAP.get(class_folder.name.lower(), class_folder.name)
            target_dir = split_test / target_class
            target_dir.mkdir(parents=True, exist_ok=True)
            for img in class_folder.glob("*.*"):
                if img.suffix.lower() in VALID_EXTENSIONS:
                    shutil.copy2(img, target_dir / img.name)

    # Split Training directory into 80% train and 20% val (stratified)
    for class_folder in raw_training_dir.iterdir():
        if class_folder.is_dir():
            target_class = CLASS_ALIAS_MAP.get(class_folder.name.lower(), class_folder.name)
            img_files = [f for f in class_folder.glob("*.*") if f.suffix.lower() in VALID_EXTENSIONS]
            if not img_files:
                continue

            tr_files, val_files = train_test_split(
                img_files, test_size=0.20, random_state=42, shuffle=True
            )

            for dest_split, files in [(split_train, tr_files), (split_val, val_files)]:
                t_dir = dest_split / target_class
                t_dir.mkdir(parents=True, exist_ok=True)
                for f in files:
                    shutil.copy2(f, t_dir / f.name)

    LOGGER.info("Dataset splits prepared successfully.")


def _build_callbacks(model_name: str, patience: int):
    import tensorflow as tf

    checkpoint_path = MODEL_CHECKPOINT_DIR / f"{model_name}.keras"
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
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
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(Path("logs") / f"{model_name}_training.csv")),
    ]


def train_all_models(config: TrainingConfig) -> dict:
    """Train all supported backbones using 2-stage transfer learning and select the best model."""
    ensure_project_dirs()
    save_class_names_json(MODEL_EXPORT_DIR)
    _prepare_splits_from_archive()

    leaderboard: list[dict] = []
    best_run: dict | None = None

    for model_name in MODEL_SPECS:
        LOGGER.info("=========================================")
        LOGGER.info("Starting Two-Stage Training for %s", model_name)
        LOGGER.info("=========================================")

        train_gen, val_gen, test_gen = _build_generators(model_name, config.batch_size)

        # Compute balanced class weights
        class_weights_arr = compute_class_weight(
            "balanced", classes=np.unique(train_gen.classes), y=train_gen.classes
        )
        class_weight_dict = dict(enumerate(class_weights_arr))

        # --- STAGE 1: Train Classification Head ---
        LOGGER.info("Stage 1: Training Classification Head (Base Frozen)...")
        model = build_transfer_model(model_name, stage="stage1", learning_rate=1e-3)
        history_s1 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config.stage1_epochs,
            class_weight=class_weight_dict,
            verbose=1,
        )

        # --- STAGE 2: Fine-Tune Top Blocks ---
        LOGGER.info("Stage 2: Fine-Tuning Top Blocks (Low LR 1e-4, Frozen BN)...")
        model = unfreeze_stage2_model(model, learning_rate=config.learning_rate)
        history_s2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config.stage2_epochs,
            class_weight=class_weight_dict,
            callbacks=_build_callbacks(model_name, config.patience),
            verbose=1,
        )

        # Merge history
        combined_history = {}
        for key in history_s1.history:
            combined_history[key] = history_s1.history[key] + history_s2.history.get(key, [])

        # Evaluate on Testing set
        probabilities = model.predict(test_gen, verbose=0)
        y_pred = probabilities.argmax(axis=1)
        y_true = test_gen.classes
        metrics = evaluate_predictions(y_true, y_pred, CLASS_NAMES)
        metrics["model_name"] = model_name
        metrics["history"] = combined_history

        model_export_path = MODEL_EXPORT_DIR / f"{model_name}.keras"
        model.save(model_export_path)
        metrics["model_path"] = str(model_export_path)
        metrics["plots"] = save_training_history(combined_history, model_name.lower())
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

    metadata = {
        "model_name": best_run["model_name"],
        "model_version": "2.0.0",
        "model_path": str(best_alias_path),
        "accuracy": best_run["accuracy"],
        "precision": best_run["precision"],
        "recall": best_run["recall"],
        "f1_score": best_run["f1_score"],
    }
    Path(MODEL_EXPORT_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary = {
        "leaderboard": sorted(
            [
                {
                    "model_name": run["model_name"],
                    "accuracy": run["accuracy"],
                    "precision": run["precision"],
                    "recall": run["recall"],
                    "f1_score": run["f1_score"],
                    "model_path": run["model_path"],
                }
                for run in leaderboard
            ],
            key=lambda item: item["f1_score"],
            reverse=True,
        ),
        "best_model": metadata,
    }
    Path("logs/training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

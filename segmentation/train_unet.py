"""U-Net training script for binary brain tumour segmentation.

Loads paired MRI image-mask datasets (e.g. Figshare / TCGA-LGG), applies data augmentation,
trains with combined Dice+BCE loss, evaluates by Dice coefficient and IoU metrics,
and exports the trained model to models/exports/unet_brain_mri.keras.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from segmentation.unet_model import build_unet_model, dice_bce_loss, dice_coef, iou_coef
from utils import MODEL_EXPORT_DIR, configure_logging, ensure_project_dirs

LOGGER = configure_logging(__name__)
UNET_MODEL_PATH = MODEL_EXPORT_DIR / "unet_brain_mri.keras"


def load_segmentation_dataset(dataset_dir: Path, target_size: tuple[int, int] = (224, 224)) -> tuple[np.ndarray, np.ndarray]:
    """Load paired MRI images and binary masks from dataset_dir/images and dataset_dir/masks."""
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        # Check subfolder structure if flattened
        images_dir = dataset_dir
        masks_dir = dataset_dir

    image_paths = sorted([p for p in images_dir.glob("*.*") if p.suffix.lower() in {".jpg", ".png", ".tif", ".jpeg"} and "_mask" not in p.stem])
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for img_path in image_paths:
        # Match mask filename
        mask_candidates = [
            masks_dir / f"{img_path.stem}_mask{img_path.suffix}",
            masks_dir / f"{img_path.stem}_mask.png",
            masks_dir / f"{img_path.stem}.png",
            masks_dir / img_path.name,
        ]
        mask_path = next((p for p in mask_candidates if p.exists()), None)
        if mask_path is None:
            continue

        try:
            img = Image.open(img_path).convert("RGB").resize(target_size)
            mask = Image.open(mask_path).convert("L").resize(target_size)

            images.append(np.array(img, dtype=np.float32) / 255.0)
            mask_arr = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
            masks.append(np.expand_dims(mask_arr, axis=-1))
        except Exception as err:
            LOGGER.warning("Error reading paired sample %s: %s", img_path.name, err)

    if not images:
        raise ValueError(f"No valid image-mask pairs found in {dataset_dir}")

    LOGGER.info("Loaded %d paired MRI image-mask samples.", len(images))
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)


def create_augmented_dataset(X: np.ndarray, Y: np.ndarray, batch_size: int = 16):
    """Create tf.data.Dataset with random augmentation for image and mask pairs."""
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices((X, Y))

    def _augment(img, mask):
        # Random horizontal flip
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)
        # Random vertical flip
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_up_down(img)
            mask = tf.image.flip_up_down(mask)
        return img, mask

    dataset = dataset.shuffle(buffer_size=len(X)).map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_unet(
    dataset_dir: str | Path,
    epochs: int = 25,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    output_model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train U-Net model on segmentation dataset and evaluate Dice coefficient and IoU."""
    import tensorflow as tf

    ensure_project_dirs()
    if output_model_path is None:
        output_model_path = UNET_MODEL_PATH
    output_model_path = Path(output_model_path)

    dataset_path = Path(dataset_dir)
    X, Y = load_segmentation_dataset(dataset_path)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    train_ds = create_augmented_dataset(X_train, Y_train, batch_size=batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(batch_size)

    model = build_unet_model(input_shape=(224, 224, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=dice_bce_loss,
        metrics=[dice_coef, iou_coef],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_model_path),
            monitor="val_dice_coef",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_dice_coef", mode="max", patience=8, verbose=1),
    ]

    LOGGER.info("Starting U-Net training for %d epochs...", epochs)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Evaluate best model
    if output_model_path.exists():
        model = tf.keras.models.load_model(
            output_model_path,
            custom_objects={"dice_bce_loss": dice_bce_loss, "dice_coef": dice_coef, "iou_coef": iou_coef},
        )

    val_loss, val_dice, val_iou = model.evaluate(val_ds, verbose=0)
    LOGGER.info("Validation Evaluation -> Loss: %.4f | Dice Coef: %.4f | IoU: %.4f", val_loss, val_dice, val_iou)

    return {
        "val_loss": float(val_loss),
        "val_dice": float(val_dice),
        "val_iou": float(val_iou),
        "model_path": str(output_model_path),
    }


def main() -> None:
    """CLI entrypoint for U-Net training."""
    parser = argparse.ArgumentParser(description="Train U-Net model for brain tumour segmentation.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to paired segmentation dataset (with images/ and masks/)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    args = parser.parse_args()

    print("=======================================================")
    print("      NEUROSCAN AI U-NET SEGMENTATION TRAINING        ")
    print("=======================================================\n")

    res = train_unet(dataset_dir=args.dataset_dir, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)

    print(f"Model Exported To : {res['model_path']}")
    print(f"Validation Loss   : {res['val_loss']:.4f}")
    print(f"Validation Dice   : {res['val_dice'] * 100:.2f}%")
    print(f"Validation IoU    : {res['val_iou'] * 100:.2f}%\n")
    print("=======================================================")


if __name__ == "__main__":
    main()

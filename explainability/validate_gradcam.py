"""Grad-CAM validation script for evaluating heatmap localization and border/background metrics."""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from explainability.gradcam import find_target_conv_layer, generate_grad_cam
from model.cnn_model import load_model
from models.model_registry import MODEL_SPECS, build_transfer_model, get_backbone_and_preprocess
from utils import configure_logging

LOGGER = configure_logging(__name__)
SEED = 42
SAMPLES_PER_CLASS = 20
BORDER_MARGIN = 0.15
FLAG_THRESHOLD = 0.40

TEST_DIR = Path("data/splits/test")
PLOTS_DIR = Path("data/artifacts/plots/gradcam_validation")
LOGS_DIR = Path("logs")
CLASS_NAMES = ["Glioma", "Meningioma", "Normal", "Pituitary"]


def compute_heatmap_metrics(
    heatmap: np.ndarray,
    raw_rgb: np.ndarray,
    border_margin: float = BORDER_MARGIN,
) -> dict[str, float]:
    """Compute border intensity share and near-black background intensity share.

    Args:
        heatmap: Normalized 2D float array (0.0 to 1.0) of shape (H, W).
        raw_rgb: Un-preprocessed input image RGB uint8 array (H, W, 3).
        border_margin: Fractional margin for border frame (default 0.15).

    Returns:
        dict[str, float]: Border share, background share, and combined share metrics.
    """
    height, width = heatmap.shape[:2]
    total_intensity = float(np.sum(heatmap)) + 1e-8

    # Build 15% outer border mask
    top = int(height * border_margin)
    bottom = int(height * (1.0 - border_margin))
    left = int(width * border_margin)
    right = int(width * (1.0 - border_margin))

    border_mask = np.ones((height, width), dtype=bool)
    border_mask[top:bottom, left:right] = False
    border_intensity = float(np.sum(heatmap[border_mask]))
    border_share = border_intensity / total_intensity

    # Build near-black background mask (grayscale pixel intensity < 20)
    gray = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY)
    bg_mask = gray < 20
    bg_intensity = float(np.sum(heatmap[bg_mask]))
    bg_share = bg_intensity / total_intensity

    combined_share = border_share + bg_share
    return {
        "border_share": round(border_share, 4),
        "background_share": round(bg_share, 4),
        "combined_share": round(combined_share, 4),
    }


def extract_raw_heatmap(
    model: Any,
    img_array: np.ndarray,
    class_idx: int,
    last_conv_layer: str | None = None,
) -> np.ndarray | None:
    """Extract raw 2D normalized float heatmap (0 to 1) from Keras model."""
    import tensorflow as tf

    inp = img_array.copy()
    if inp.ndim == 3:
        inp = np.expand_dims(inp, axis=0)

    target_layer = find_target_conv_layer(model, last_conv_layer)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inp)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]

    heatmap = tf.squeeze(conv_out @ pooled_grads[..., tf.newaxis]).numpy()
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    height, width = inp.shape[1:3]
    resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized


def audit_backbone_conv_layers() -> dict[str, dict[str, Any]]:
    """Audit layer lookup and Grad-CAM extraction across standard backbones."""
    results = {}
    for name, spec in MODEL_SPECS.items():
        try:
            model = build_transfer_model(name)
            layer = find_target_conv_layer(model, spec.last_conv_hint)
            results[name] = {
                "status": "success",
                "resolved_layer": layer.name,
                "hint": spec.last_conv_hint,
            }
        except Exception as exc:
            results[name] = {
                "status": "failed",
                "error": str(exc),
                "hint": spec.last_conv_hint,
            }
    return results


def run_gradcam_validation():
    """Run full validation suite sampling 20 test images per class."""
    random.seed(SEED)
    np.random.seed(SEED)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting Grad-CAM backbone resolution audit...")
    backbone_audit = audit_backbone_conv_layers()

    LOGGER.info("Loading active classifier for dataset evaluation...")
    active_model = load_model()

    records = []
    scans_by_class = {}

    for cls_name in CLASS_NAMES:
        cls_dir = TEST_DIR / cls_name
        if not cls_dir.exists():
            LOGGER.warning("Test directory %s not found.", cls_dir)
            continue

        images = sorted(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")))
        if not images:
            LOGGER.warning("No image files in %s", cls_dir)
            continue

        sampled = random.sample(images, min(SAMPLES_PER_CLASS, len(images)))
        scans_by_class[cls_name] = []

        cls_idx = CLASS_NAMES.index(cls_name)

        for img_path in sampled:
            raw_bgr = cv2.imread(str(img_path))
            if raw_bgr is None:
                continue
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            raw_rgb_resized = cv2.resize(raw_rgb, (224, 224))

            # Normalize for inference
            img_inp = (raw_rgb_resized / 255.0).astype(np.float32)
            img_inp_batch = np.expand_dims(img_inp, axis=0)

            preds = active_model.predict(img_inp_batch, verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            pred_cls = CLASS_NAMES[pred_idx]
            confidence = float(preds[pred_idx])

            heatmap = extract_raw_heatmap(active_model, img_inp_batch, pred_idx)
            if heatmap is None:
                heatmap = np.zeros((224, 224), dtype=np.float32)

            metrics = compute_heatmap_metrics(heatmap, raw_rgb_resized)
            flagged = metrics["border_share"] > FLAG_THRESHOLD or metrics["background_share"] > FLAG_THRESHOLD or metrics["combined_share"] > FLAG_THRESHOLD

            overlay = generate_grad_cam(active_model, img_inp_batch, pred_idx, original_rgb=raw_rgb_resized)

            record = {
                "file_name": img_path.name,
                "ground_truth": cls_name,
                "predicted_class": pred_cls,
                "confidence": round(confidence, 4),
                "border_share": metrics["border_share"],
                "background_share": metrics["background_share"],
                "combined_share": metrics["combined_share"],
                "flagged": flagged,
            }
            records.append(record)
            scans_by_class[cls_name].append({
                "path": img_path,
                "raw_rgb": raw_rgb_resized,
                "overlay": overlay,
                "record": record,
            })

    # Save visual grid plots for each class
    for cls_name, items in scans_by_class.items():
        if not items:
            continue
        num_items = len(items)
        cols = 5
        rows = (num_items + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        fig.suptitle(f"Grad-CAM Visual Validation Grid: Class {cls_name}", fontsize=14, fontweight="bold")
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for idx, item in enumerate(items):
            ax = axes_flat[idx]
            display_img = item["overlay"] if item["overlay"] is not None else item["raw_rgb"]
            ax.imshow(display_img)
            rec = item["record"]
            flag_str = " [FLAGGED]" if rec["flagged"] else ""
            ax.set_title(
                f"Pred: {rec['predicted_class']}\nBdr: {rec['border_share']:.2f} | Bg: {rec['background_share']:.2f}{flag_str}",
                fontsize=8,
                color="red" if rec["flagged"] else "black",
            )
            ax.axis("off")

        for idx in range(num_items, len(axes_flat)):
            axes_flat[idx].axis("off")

        plt.tight_layout()
        grid_plot_path = PLOTS_DIR / f"gradcam_grid_{cls_name}.png"
        plt.savefig(grid_plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    # Save summary report JSON and CSV
    flagged_count = sum(1 for r in records if r["flagged"])
    total_eval = len(records)
    flag_rate = round(flagged_count / max(total_eval, 1), 4)

    json_report = {
        "dataset_summary": {
            "samples_evaluated": total_eval,
            "samples_flagged": flagged_count,
            "flag_rate": flag_rate,
            "flag_threshold": FLAG_THRESHOLD,
        },
        "backbone_audit": backbone_audit,
        "records": records,
    }

    # Save JSON report to logs/ and PLOTS_DIR
    (LOGS_DIR / "gradcam_validation_report.json").write_text(json.dumps(json_report, indent=2), encoding="utf-8")
    (PLOTS_DIR / "gradcam_validation_report.json").write_text(json.dumps(json_report, indent=2), encoding="utf-8")

    # Save CSV report to PLOTS_DIR
    csv_path = PLOTS_DIR / "gradcam_validation_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_name",
                "ground_truth",
                "predicted_class",
                "confidence",
                "border_share",
                "background_share",
                "combined_share",
                "flagged",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    LOGGER.info(
        "Grad-CAM Validation complete. %d/%d scans flagged (Flag Rate: %.2f%%). Reports saved.",
        flagged_count,
        total_eval,
        flag_rate * 100,
    )
    print(f"Validation finished. Evaluated {total_eval} scans across 4 classes.")
    print(f"Flagged scans (border/bg > 0.40): {flagged_count}/{total_eval} ({flag_rate*100:.1f}%)")
    print(f"JSON report saved to: {LOGS_DIR / 'gradcam_validation_report.json'}")
    print(f"CSV report saved to: {csv_path}")


if __name__ == "__main__":
    run_gradcam_validation()

"""Model evaluation script for NeuroScan AI.

Loads exported model (or fallback model), evaluates on the Testing dataset,
and produces confusion matrix, per-class metrics (JSON + CSV), macro/weighted averages,
and One-vs-Rest ROC curves with AUC. Saves outputs to logs/ and data/artifacts/plots/.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from model.cnn_model import CLASS_NAMES, get_model_metadata, load_model
from models.model_registry import get_backbone_and_preprocess
from preprocessing.image_processing import preprocess_array
from utils import LOGS_DIR, PLOTS_DIR, MODEL_EXPORT_DIR, configure_logging, ensure_project_dirs

matplotlib.use("Agg")
LOGGER = configure_logging(__name__)

CLASS_ALIAS_MAP = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary",
    "notumor": "Normal",
    "normal": "Normal",
}
TUMOUR_CLASSES = ["Glioma", "Meningioma", "Pituitary"]


def load_target_model(model_path: str | Path | None = None):
    """Load model from model_path or fall back to default model loader."""
    if model_path is None:
        model_path = MODEL_EXPORT_DIR / "best_model.keras"

    model_path = Path(model_path)
    if model_path.exists():
        try:
            import tensorflow as tf

            LOGGER.info("Loading model from %s", model_path)
            return tf.keras.models.load_model(model_path)
        except Exception as err:
            LOGGER.warning("Failed loading %s (%s). Falling back to app model.", model_path, err)

    LOGGER.info("Using default app model loader.")
    return load_model()


def resolve_testing_directory(custom_path: str | Path | None = None) -> Path:
    """Locate the testing dataset directory from candidate paths."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p

    candidates = [
        Path("data/splits/test"),
        Path("data/raw/archive/Testing"),
        Path("data/splits/Testing"),
        Path("data/raw/Testing"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Testing dataset directory not found in candidate locations.")


def load_test_samples(testing_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and preprocess test images and ground truth label indices from directory."""
    raw_images: list[np.ndarray] = []
    y_true_indices: list[int] = []
    sample_files: list[str] = []

    for folder in sorted(testing_dir.iterdir()):
        if not folder.is_dir():
            continue
        canonical_class = CLASS_ALIAS_MAP.get(folder.name.lower(), folder.name)
        if canonical_class not in CLASS_NAMES:
            continue

        class_idx = CLASS_NAMES.index(canonical_class)
        for img_path in sorted(folder.glob("*.*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                pil_img = Image.open(img_path).convert("RGB")
                img_array = np.array(pil_img)
                prep = preprocess_array(img_array)
                raw_images.append(prep["resized"])
                y_true_indices.append(class_idx)
                sample_files.append(img_path.name)
            except Exception as err:
                LOGGER.warning("Skipping file %s: %s", img_path, err)

    if not raw_images:
        raise ValueError(f"No valid test images found in {testing_dir}")

    X_raw = np.array(raw_images, dtype=np.float32)
    y_true = np.array(y_true_indices, dtype=int)
    LOGGER.info("Loaded %d test samples from %s", len(X_raw), testing_dir)
    return X_raw, y_true, sample_files


def run_model_inference(model: Any, X_raw: np.ndarray) -> np.ndarray:
    """Apply backbone preprocessing and run model inference."""
    metadata = get_model_metadata()
    model_name = metadata.get("model_name", "VGG16")

    try:
        _, preprocess_fn = get_backbone_and_preprocess(model_name)
        preprocessed_input = preprocess_fn(X_raw.copy())
    except Exception:
        preprocessed_input = X_raw / 255.0

    try:
        y_probs = model.predict(preprocessed_input, batch_size=32, verbose=0)
    except Exception:
        y_probs = model.predict(preprocessed_input)

    return np.array(y_probs)


def compute_roc_data(y_true: np.ndarray, y_probs: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    """Compute One-vs-Rest ROC curve FPR, TPR, and AUC for each class and macro/micro averages."""
    num_classes = len(class_names)
    y_onehot = np.eye(num_classes)[y_true]

    fpr_dict: dict[str, list[float]] = {}
    tpr_dict: dict[str, list[float]] = {}
    auc_dict: dict[str, float] = {}

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_onehot[:, i], y_probs[:, i])
        score = float(auc(fpr, tpr))
        fpr_dict[class_name] = fpr.tolist()
        tpr_dict[class_name] = tpr.tolist()
        auc_dict[class_name] = round(score, 4)

    # Micro average
    fpr_micro, tpr_micro, _ = roc_curve(y_onehot.ravel(), y_probs.ravel())
    auc_dict["micro"] = round(float(auc(fpr_micro, tpr_micro)), 4)
    fpr_dict["micro"] = fpr_micro.tolist()
    tpr_dict["micro"] = tpr_micro.tolist()

    # Macro average
    all_fpr = np.unique(np.concatenate([np.array(fpr_dict[c]) for c in class_names]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in class_names:
        mean_tpr += np.interp(all_fpr, np.array(fpr_dict[c]), np.array(tpr_dict[c]))
    mean_tpr /= num_classes
    auc_dict["macro"] = round(float(auc(all_fpr, mean_tpr)), 4)
    fpr_dict["macro"] = all_fpr.tolist()
    tpr_dict["macro"] = mean_tpr.tolist()

    return {"fpr": fpr_dict, "tpr": tpr_dict, "auc": auc_dict}


def evaluate_metrics(y_true: np.ndarray, y_probs: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    """Calculate per-class, macro, and weighted evaluation metrics."""
    y_pred = y_probs.argmax(axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    per_class_report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    tumour_recalls = {cls: float(per_class_report[cls]["recall"]) for cls in TUMOUR_CLASSES if cls in per_class_report}
    roc_data = compute_roc_data(y_true, y_probs, class_names)

    return {
        "accuracy": round(acc, 4),
        "macro_avg": {
            "precision": round(prec_macro, 4),
            "recall": round(rec_macro, 4),
            "f1_score": round(f1_macro, 4),
        },
        "weighted_avg": {
            "precision": round(prec_weighted, 4),
            "recall": round(rec_weighted, 4),
            "f1_score": round(f1_weighted, 4),
        },
        "per_class": {
            cls: {
                "precision": round(per_class_report[cls]["precision"], 4),
                "recall": round(per_class_report[cls]["recall"], 4),
                "f1_score": round(per_class_report[cls]["f1-score"], 4),
                "support": int(per_class_report[cls]["support"]),
            }
            for cls in class_names
        },
        "tumour_recalls": tumour_recalls,
        "confusion_matrix": cm.tolist(),
        "roc_auc": roc_data["auc"],
        "roc_curves": roc_data,
        "total_samples": len(y_true),
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_paths: Sequence[Path]) -> None:
    """Render and save styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor="#0d1117")
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True,
    )
    ax.set_facecolor("#161b22")
    ax.set_title("Confusion Matrix (Testing Split)", color="#58a6ff", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label", color="#8b949e", fontsize=10)
    ax.set_ylabel("True Label", color="#8b949e", fontsize=10)
    ax.tick_params(colors="#8b949e")

    plt.tight_layout()
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_roc_curves(roc_data: dict[str, Any], class_names: list[str], output_paths: Sequence[Path]) -> None:
    """Render and save One-vs-Rest ROC curves figure."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")

    colors = ["#ff6b6b", "#ffd43b", "#51cf66", "#74c0fc"]
    for i, cls in enumerate(class_names):
        fpr = roc_data["fpr"][cls]
        tpr = roc_data["tpr"][cls]
        auc_score = roc_data["auc"][cls]
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f"{cls} (AUC = {auc_score:.3f})")

    # Macro & Micro average lines
    ax.plot(
        roc_data["fpr"]["macro"],
        roc_data["tpr"]["macro"],
        color="#b197fc",
        linestyle=":",
        lw=2.5,
        label=f"Macro-average (AUC = {roc_data['auc']['macro']:.3f})",
    )
    ax.plot(
        roc_data["fpr"]["micro"],
        roc_data["tpr"]["micro"],
        color="#ff922b",
        linestyle="--",
        lw=2,
        label=f"Micro-average (AUC = {roc_data['auc']['micro']:.3f})",
    )

    # Random baseline
    ax.plot([0, 1], [0, 1], color="#485058", linestyle="--", lw=1.5, label="Random Chance")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", color="#8b949e", fontsize=10)
    ax.set_ylabel("True Positive Rate", color="#8b949e", fontsize=10)
    ax.set_title("One-vs-Rest ROC Curves (Testing Split)", color="#58a6ff", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors="#8b949e")
    ax.grid(True, color="#21262d", linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)

    plt.tight_layout()
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
    plt.close(fig)


def export_metrics(metrics: dict[str, Any], json_paths: Sequence[Path], csv_paths: Sequence[Path]) -> None:
    """Export evaluation metrics to JSON and CSV files."""
    # Write JSON
    json_content = json.dumps(metrics, indent=2)
    for p in json_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json_content, encoding="utf-8")

    # Write CSV
    rows = []
    for cls, stat in metrics["per_class"].items():
        rows.append(
            {
                "Class": cls,
                "Precision": stat["precision"],
                "Recall": stat["recall"],
                "F1-Score": stat["f1_score"],
                "Support": stat["support"],
            }
        )

    rows.append(
        {
            "Class": "Macro Average",
            "Precision": metrics["macro_avg"]["precision"],
            "Recall": metrics["macro_avg"]["recall"],
            "F1-Score": metrics["macro_avg"]["f1_score"],
            "Support": metrics["total_samples"],
        }
    )
    rows.append(
        {
            "Class": "Weighted Average",
            "Precision": metrics["weighted_avg"]["precision"],
            "Recall": metrics["weighted_avg"]["recall"],
            "F1-Score": metrics["weighted_avg"]["f1_score"],
            "Support": metrics["total_samples"],
        }
    )

    fieldnames = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    for p in csv_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def evaluate_and_save(model_path: str | Path | None = None, testing_dir: str | Path | None = None) -> dict[str, Any]:
    """Run end-to-end evaluation pipeline and persist artifacts."""
    ensure_project_dirs()
    model = load_target_model(model_path)
    test_path = resolve_testing_directory(testing_dir)

    X_raw, y_true, _ = load_test_samples(test_path)
    y_probs = run_model_inference(model, X_raw)
    metrics = evaluate_metrics(y_true, y_probs, CLASS_NAMES)

    cm = np.array(metrics["confusion_matrix"])
    cm_paths = [LOGS_DIR / "confusion_matrix.png", PLOTS_DIR / "confusion_matrix.png"]
    roc_paths = [LOGS_DIR / "roc_curves.png", PLOTS_DIR / "roc_curves.png"]
    json_paths = [LOGS_DIR / "evaluation_metrics.json", PLOTS_DIR / "evaluation_metrics.json"]
    csv_paths = [LOGS_DIR / "evaluation_metrics.csv", PLOTS_DIR / "evaluation_metrics.csv"]

    plot_confusion_matrix(cm, CLASS_NAMES, cm_paths)
    plot_roc_curves(metrics["roc_curves"], CLASS_NAMES, roc_paths)
    export_metrics(metrics, json_paths, csv_paths)

    LOGGER.info("Saved evaluation artifacts to logs/ and data/artifacts/plots/")
    return metrics


def main() -> None:
    """CLI entrypoint for running evaluation."""
    print("=======================================================")
    print("        NEUROSCAN AI MODEL EVALUATION PIPELINE        ")
    print("=======================================================\n")

    metrics = evaluate_and_save()

    print(f"Testing Samples Evaluated : {metrics['total_samples']}")
    print(f"Overall Accuracy          : {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision           : {metrics['macro_avg']['precision'] * 100:.2f}%")
    print(f"Macro Recall              : {metrics['macro_avg']['recall'] * 100:.2f}%")
    print(f"Macro F1 Score            : {metrics['macro_avg']['f1_score'] * 100:.2f}%\n")

    print("--- [TARGET] TUMOUR CLASS RECALL HIGHLIGHTS ---")
    for cls in TUMOUR_CLASSES:
        recall_val = metrics["tumour_recalls"].get(cls, 0.0) * 100
        print(f"  • {cls:<12} Recall : {recall_val:6.2f}%")

    print("\n--- PER-CLASS METRICS SUMMARY ---")
    df = pd.DataFrame.from_dict(metrics["per_class"], orient="index")
    print(df.to_string())

    print("\n=======================================================")


if __name__ == "__main__":
    main()

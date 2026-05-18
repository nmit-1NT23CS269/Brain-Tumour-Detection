"""Model evaluation and comparison utilities."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from utils import PLOTS_DIR


matplotlib.use("Agg")


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    """Compute key evaluation metrics from integer labels."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=labels,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_confusion_matrix(metrics: dict, labels: list[str], output_path: str | Path) -> str:
    """Persist a confusion matrix figure to disk."""
    matrix = np.array(metrics["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axis)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Confusion Matrix")
    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def save_training_history(history: dict, output_prefix: str) -> dict[str, str]:
    """Save loss and accuracy training curves."""
    results: dict[str, str] = {}

    for metric_name in ["accuracy", "loss"]:
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.plot(history.get(metric_name, []), label=f"Train {metric_name}")
        axis.plot(history.get(f"val_{metric_name}", []), label=f"Val {metric_name}")
        axis.set_title(f"Training {metric_name.title()}")
        axis.set_xlabel("Epoch")
        axis.legend()
        fig.tight_layout()
        path = PLOTS_DIR / f"{output_prefix}_{metric_name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        results[metric_name] = str(path)
    return results


def persist_metrics(metrics: dict, output_path: str | Path) -> str:
    """Write metrics JSON to disk."""
    output_path = Path(output_path)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return str(output_path)

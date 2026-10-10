"""Confidence calibration using temperature scaling for NeuroScan AI.

Fits a temperature parameter T > 0 on validation set logits, computes ECE (Expected Calibration Error)
before and after calibration, plots a reliability diagram, and saves parameters to models/exports/temperature.json.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar

from model.cnn_model import CLASS_NAMES, get_model_metadata, load_model
from models.model_registry import get_backbone_and_preprocess
from preprocessing.image_processing import preprocess_array
from utils import LOGS_DIR, MODEL_EXPORT_DIR, PLOTS_DIR, configure_logging, ensure_project_dirs

matplotlib.use("Agg")
LOGGER = configure_logging(__name__)

TEMPERATURE_JSON_PATH = MODEL_EXPORT_DIR / "temperature.json"
DEFAULT_CONFIDENCE_THRESHOLD = 0.70


def load_temperature_config() -> dict[str, Any]:
    """Load cached temperature calibration configuration from disk."""
    candidates = [TEMPERATURE_JSON_PATH, LOGS_DIR / "temperature.json"]
    for path in candidates:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if "temperature" in data:
                    return data
            except Exception as err:
                LOGGER.warning("Failed loading temperature config from %s: %s", path, err)

    return {
        "temperature": 1.0,
        "ece_before": None,
        "ece_after": None,
        "low_confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "calibrated_at": None,
    }


def compute_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> tuple[float, dict[str, Any]]:
    """Compute Expected Calibration Error (ECE) and bin statistics."""
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs: list[float] = []
    bin_confs: list[float] = []
    bin_counts: list[int] = []

    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = int(np.sum(in_bin))

        if bin_size > 0:
            avg_acc = float(np.mean(accuracies[in_bin]))
            avg_conf = float(np.mean(confidences[in_bin]))
            ece += (bin_size / total_samples) * abs(avg_acc - avg_conf)

            bin_accs.append(round(avg_acc, 4))
            bin_confs.append(round(avg_conf, 4))
            bin_counts.append(bin_size)
        else:
            bin_accs.append(0.0)
            bin_confs.append(round((bin_lower + bin_upper) / 2.0, 4))
            bin_counts.append(0)

    bin_stats = {
        "bin_boundaries": bin_boundaries.tolist(),
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
    }
    return round(float(ece), 4), bin_stats


def apply_temperature_to_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by temperature and compute softmax probabilities."""
    temperature = max(float(temperature), 0.01)
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def probabilities_to_logits(probs: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Convert softmax probabilities back to pseudo-logits."""
    clipped = np.clip(probs, eps, 1.0 - eps)
    return np.log(clipped)


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Fit optimal temperature scalar T > 0 on validation set using Negative Log-Likelihood loss."""
    num_samples = len(y_true)

    def _nll_objective(temp: float) -> float:
        probs = apply_temperature_to_logits(logits, temp)
        nll = -np.mean(np.log(probs[np.arange(num_samples), y_true] + 1e-12))
        return float(nll)

    res = minimize_scalar(_nll_objective, bounds=(0.1, 10.0), method="bounded")
    best_temp = float(res.x)
    LOGGER.info("Fitted temperature T = %.4f (NLL = %.4f)", best_temp, float(res.fun))
    return round(best_temp, 4)


def plot_reliability_diagram(
    ece_before: float,
    bins_before: dict[str, Any],
    ece_after: float,
    bins_after: dict[str, Any],
    output_paths: Sequence[Path],
) -> None:
    """Render dual reliability diagram before and after temperature scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0d1117")

    titles = [f"Uncalibrated (ECE = {ece_before:.4f})", f"Calibrated (T-Scaled, ECE = {ece_after:.4f})"]
    bin_datasets = [bins_before, bins_after]

    for idx, ax in enumerate(axes):
        ax.set_facecolor("#161b22")
        data = bin_datasets[idx]
        confs = data["bin_confidences"]
        accs = data["bin_accuracies"]

        # Ideal identity line
        ax.plot([0, 1], [0, 1], "--", color="#485058", label="Perfect Calibration")

        # Bar plot of accuracy vs confidence
        ax.bar(
            confs,
            accs,
            width=0.08,
            alpha=0.7,
            color="#58a6ff" if idx == 1 else "#ff6b6b",
            edgecolor="#30363d",
            label="Outputs",
        )

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Confidence", color="#8b949e")
        ax.set_ylabel("Accuracy", color="#8b949e")
        ax.set_title(titles[idx], color="#58a6ff", fontsize=11, fontweight="bold")
        ax.tick_params(colors="#8b949e")
        ax.grid(True, color="#21262d", linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")

    plt.tight_layout()
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
    plt.close(fig)


def save_calibration_config(
    temperature: float,
    ece_before: float,
    ece_after: float,
    output_paths: Sequence[Path],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> None:
    """Persist temperature calibration parameters and metadata to JSON."""
    payload = {
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "low_confidence_threshold": threshold,
        "calibrated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_str = json.dumps(payload, indent=2)

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_str, encoding="utf-8")

    LOGGER.info("Saved calibration config to %s", output_paths[0])


def load_val_data(val_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load preprocessed RGB images and true integer labels from validation split."""
    images: list[np.ndarray] = []
    labels: list[int] = []

    alias_map = {"glioma": "Glioma", "meningioma": "Meningioma", "pituitary": "Pituitary", "notumor": "Normal", "normal": "Normal"}

    for folder in sorted(val_dir.iterdir()):
        if not folder.is_dir():
            continue
        canonical_class = alias_map.get(folder.name.lower(), folder.name)
        if canonical_class not in CLASS_NAMES:
            continue
        class_idx = CLASS_NAMES.index(canonical_class)

        for img_path in sorted(folder.glob("*.*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                pil_img = Image.open(img_path).convert("RGB")
                prep = preprocess_array(np.array(pil_img))
                images.append(prep["resized"])
                labels.append(class_idx)
            except Exception as err:
                LOGGER.warning("Error reading %s: %s", img_path, err)

    if not images:
        raise ValueError(f"No valid validation images found in {val_dir}")

    return np.array(images, dtype=np.float32), np.array(labels, dtype=int)


def calibrate_validation_split(val_dir: Path | str | None = None) -> dict[str, Any]:
    """Fit temperature scaling on validation set and generate calibration artifacts."""
    ensure_project_dirs()

    if val_dir is None:
        candidates = [Path("data/splits/val"), Path("data/splits/test"), Path("data/raw/archive/Testing")]
        val_dir = next((p for p in candidates if p.exists()), None)
        if val_dir is None:
            raise FileNotFoundError("Validation set directory not found.")
    else:
        val_dir = Path(val_dir)

    LOGGER.info("Loading validation data from %s...", val_dir)
    X_raw, y_true = load_val_data(val_dir)

    model = load_model()
    metadata = get_model_metadata()
    model_name = metadata.get("model_name", "VGG16")

    try:
        _, preprocess_fn = get_backbone_and_preprocess(model_name)
        preprocessed_input = preprocess_fn(X_raw.copy())
    except Exception:
        preprocessed_input = X_raw / 255.0

    raw_probs = model.predict(preprocessed_input, batch_size=32, verbose=0)
    logits = probabilities_to_logits(raw_probs)

    ece_before, bins_before = compute_ece(y_true, raw_probs)
    temperature = fit_temperature(logits, y_true)

    calibrated_probs = apply_temperature_to_logits(logits, temperature)
    ece_after, bins_after = compute_ece(y_true, calibrated_probs)

    diagram_paths = [LOGS_DIR / "reliability_diagram.png", PLOTS_DIR / "reliability_diagram.png"]
    plot_reliability_diagram(ece_before, bins_before, ece_after, bins_after, diagram_paths)

    config_paths = [TEMPERATURE_JSON_PATH, LOGS_DIR / "temperature.json"]
    save_calibration_config(temperature, ece_before, ece_after, config_paths)

    return {
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "low_confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "total_val_samples": len(y_true),
    }


def main() -> None:
    """CLI entrypoint for confidence calibration."""
    print("=======================================================")
    print("      NEUROSCAN AI CONFIDENCE CALIBRATION PIPELINE     ")
    print("=======================================================\n")

    res = calibrate_validation_split()

    print(f"Validation Samples Evaluated : {res['total_val_samples']}")
    print(f"Fitted Temperature (T)       : {res['temperature']:.4f}")
    print(f"ECE (Before Calibration)     : {res['ece_before']:.4f}")
    print(f"ECE (After Calibration)      : {res['ece_after']:.4f}")
    print(f"Low Confidence Threshold     : {res['low_confidence_threshold'] * 100:.1f}%\n")
    print("Saved reliability diagram and calibration config to models/exports/temperature.json and logs/.")
    print("=======================================================")


if __name__ == "__main__":
    main()

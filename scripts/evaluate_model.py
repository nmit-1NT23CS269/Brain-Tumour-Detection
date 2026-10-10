"""Script to evaluate NeuroScan AI model inference on Testing dataset and sample test images."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from backend.scan_service import analyze_uploaded_scan
from model.cnn_model import CLASS_NAMES, predict, load_model, get_model_metadata
from models.model_registry import MODEL_SPECS, build_transfer_model, get_backbone_and_preprocess
from preprocessing.image_processing import preprocess_array

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

TESTING_DIR = Path("data/raw/archive/Testing")
TEST_IMAGES_DIR = Path("test_images")
PLOTS_DIR = Path("data/artifacts/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "Normal",
    "normal": "Normal",
    "pituitary": "Pituitary",
}


class MockUploadFile:
    """Mock Streamlit UploadedFile interface for backend scan service evaluation."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.name = file_path.name
        self.type = f"image/{file_path.suffix.lstrip('.').lower()}"
        if self.type == "image/jpg":
            self.type = "image/jpeg"
        self._content = file_path.read_bytes()
        self.size = len(self._content)

    def read(self) -> bytes:
        return self._content

    def seek(self, pos: int) -> None:
        pass


def evaluate_model_on_directory(model, testing_dir: Path = TESTING_DIR, name: str = "Active_App_Model") -> dict:
    """Evaluate a Keras model on testing_dir using Keras ImageDataGenerator."""
    if not testing_dir.exists():
        LOGGER.error("Testing directory not found at %s", testing_dir)
        return {}

    metadata = get_model_metadata()
    model_name = metadata.get("model_name", "VGG16")
    try:
        _, preprocess_fn = get_backbone_and_preprocess(model_name)
    except Exception:
        preprocess_fn = lambda x: x / 255.0

    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    test_gen = eval_datagen.flow_from_directory(
        testing_dir,
        target_size=(224, 224),
        batch_size=64,
        class_mode="categorical",
        shuffle=False,
    )

    # Resolve folder class index to app CLASS_NAMES
    folder_classes = [CLASS_MAP.get(k.lower(), k) for k in test_gen.class_indices.keys()]
    y_true_indices = test_gen.classes
    y_true = [folder_classes[idx] for idx in y_true_indices]

    probs = model.predict(test_gen, verbose=0)
    y_pred_indices = probs.argmax(axis=1)
    y_pred = [CLASS_NAMES[idx] for idx in y_pred_indices]
    confidences = [float(probs[i, idx]) for i, idx in enumerate(y_pred_indices)]

    confused_pairs: dict[str, int] = {}
    for true_c, pred_c in zip(y_true, y_pred):
        if true_c != pred_c:
            pair = f"True: {true_c} -> Pred: {pred_c}"
            confused_pairs[pair] = confused_pairs.get(pair, 0) + 1

    acc = accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, labels=CLASS_NAMES, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, labels=CLASS_NAMES, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=CLASS_NAMES, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)

    # Save confusion matrix plot for active model
    if name == "Active_App_Model":
        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0d1117")
        ax.imshow(cm, cmap="Blues")
        ax.set_facecolor("#161b22")
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, color="#8b949e")
        ax.set_yticklabels(CLASS_NAMES, color="#8b949e")
        plt.xlabel("Predicted Class", color="#8b949e")
        plt.ylabel("True Class", color="#8b949e")
        plt.title(f"Confusion Matrix ({name}: Acc {acc*100:.2f}%)", color="#58a6ff")
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] > cm.max()/2 else "white")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "eval_confusion_matrix.png", dpi=150)
        plt.close()

    top_confused = sorted(confused_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
    report_str = classification_report(y_true, y_pred, labels=CLASS_NAMES, zero_division=0)

    return {
        "model_name": name,
        "accuracy": acc,
        "precision": macro_prec,
        "recall": macro_rec,
        "f1_score": macro_f1,
        "confusion_matrix": cm.tolist(),
        "top_confused": top_confused,
        "classification_report": report_str,
        "total_samples": len(y_true),
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
    }


def evaluate_test_images() -> list[dict]:
    """Evaluate images in test_images/ directory using active app pipeline."""
    results = []
    if not TEST_IMAGES_DIR.exists():
        LOGGER.warning("test_images/ directory not found.")
        return results

    for img_path in sorted(TEST_IMAGES_DIR.glob("*.*")):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            mock_file = MockUploadFile(img_path)
            scan_res = analyze_uploaded_scan(mock_file, enable_gradcam=False, enable_segmentation=False)
            results.append({
                "filename": img_path.name,
                "prediction": scan_res["class"],
                "confidence": scan_res["confidence"],
                "probabilities": scan_res["probabilities"],
            })
    return results


def main() -> None:
    print("=======================================================")
    print("     NEUROSCAN AI MODEL EVALUATION (PHASE 1 DIAGNOSIS)")
    print("=======================================================\n")

    print("--- 1. ACTIVE APP INFERENCE MODEL EVALUATION ---")
    active_model = load_model()
    app_results = evaluate_model_on_directory(active_model, name="Active_App_Model")

    if app_results:
        print(f"Total Test Samples: {app_results['total_samples']}")
        print(f"Overall Accuracy : {app_results['accuracy'] * 100:.2f}%")
        print(f"Macro Precision  : {app_results['precision'] * 100:.2f}%")
        print(f"Macro Recall     : {app_results['recall'] * 100:.2f}%")
        print(f"Macro F1 Score   : {app_results['f1_score'] * 100:.2f}%")
        print(f"Mean Confidence  : {app_results['mean_confidence'] * 100:.2f}%\n")

        print("--- CLASSIFICATION REPORT (ACTIVE APP MODEL) ---")
        print(app_results["classification_report"])

        print("--- TOP CONFUSED CLASS PAIRS ---")
        for pair, count in app_results["top_confused"]:
            print(f"  {pair}: {count} occurrences")
        print()

    print("--- 2. REGISTRY BACKBONE ARCHITECTURES COMPARISON (UNTRAINED BASELINES) ---")
    print(f"{'Model Architecture':<18} | {'Accuracy':<10} | {'Macro Prec':<12} | {'Macro Rec':<10} | {'Macro F1':<10}")
    print("-" * 72)
    for model_name in MODEL_SPECS:
        bm = build_transfer_model(model_name)
        res = evaluate_model_on_directory(bm, name=model_name)
        print(f"{model_name:<18} | {res['accuracy']*100:8.2f}% | {res['precision']*100:10.2f}% | {res['recall']*100:8.2f}% | {res['f1_score']*100:8.2f}%")
    print("-" * 72 + "\n")

    print("--- 3. EVALUATION ON test_images/ SAMPLE DIRECTORY ---")
    test_img_results = evaluate_test_images()
    for res in test_img_results:
        probs_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in res["probabilities"].items()])
        print(f"Image: {res['filename']:<16} | Pred: {res['prediction']:<10} | Conf: {res['confidence']*100:6.2f}% | Probs: [{probs_str}]")
    print("\n=======================================================")


if __name__ == "__main__":
    main()

"""End-to-end MRI scan orchestration for the Streamlit app."""

from __future__ import annotations

from typing import Any

from explainability import summarize_heatmap_region
from preprocessing import preprocess_pipeline
from segmentation import segment_tumour_region
from utils import (
    build_ai_summary,
    build_medical_assistant_text,
    determine_focus_region,
    estimate_severity,
    interpret_confidence,
)

from model import get_model_metadata, predict


def analyze_uploaded_scan(uploaded_file, enable_gradcam: bool = True, enable_segmentation: bool = True) -> dict[str, Any]:
    """Run preprocessing, inference, explainability, and segmentation on an uploaded scan."""
    steps = preprocess_pipeline(uploaded_file)
    inference = predict(steps["model_input"], include_gradcam=enable_gradcam)
    gradcam_img = inference.get("gradcam")
    segmentation = segment_tumour_region(steps["resized"]) if enable_segmentation else None

    quality = steps["quality"]
    focus_metadata = summarize_heatmap_region(gradcam_img)
    focus_region = determine_focus_region(inference["class_idx"], focus_metadata)
    severity = estimate_severity(inference["class"], inference["confidence"])
    confidence_text = interpret_confidence(inference["confidence"])

    inference["severity"] = severity
    inference["confidence_text"] = confidence_text
    inference["quality"] = quality
    inference["focus_region"] = focus_region
    inference["ai_summary"] = build_ai_summary(
        inference["class"],
        inference["confidence"],
        focus_region,
        quality["label"],
        severity,
    )
    inference["assistant_text"] = build_medical_assistant_text(
        inference["class"],
        inference["confidence"],
        severity,
        quality["label"],
    )
    inference["segmentation"] = segmentation
    inference["steps"] = steps
    inference["model_metadata"] = get_model_metadata()
    return inference

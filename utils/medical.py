"""Medical-style interpretation helpers for MRI scan results."""

from __future__ import annotations

from typing import Any


def interpret_confidence(confidence: float) -> str:
    """Convert a probability score into a human-readable confidence band."""
    if confidence >= 0.9:
        return "Very high confidence"
    if confidence >= 0.75:
        return "High confidence"
    if confidence >= 0.6:
        return "Moderate confidence"
    return "Low confidence - clinical review strongly advised"


def estimate_severity(predicted_class: str, confidence: float) -> str:
    """Estimate qualitative tumour severity from class and confidence."""
    if predicted_class == "Normal":
        return "None"
    if predicted_class == "Glioma":
        return "Critical" if confidence >= 0.85 else "High"
    if predicted_class in {"Meningioma", "Pituitary"}:
        return "Moderate" if confidence >= 0.75 else "Elevated"
    return "Unknown"


def build_ai_summary(
    predicted_class: str,
    confidence: float,
    focus_region: str,
    mri_quality: str,
    severity: str,
) -> str:
    """Generate a concise explainable-AI summary."""
    confidence_text = interpret_confidence(confidence)
    if predicted_class == "Normal":
        return (
            f"The scan is classified as Normal with {confidence_text.lower()}. "
            f"MRI quality appears {mri_quality.lower()}, and the model did not show "
            f"a dominant abnormal focus. Routine clinical correlation is still recommended."
        )
    return (
        f"The model predicts {predicted_class} with {confidence_text.lower()}. "
        f"It focused most strongly on {focus_region.lower()}. "
        f"MRI quality appears {mri_quality.lower()}, and the estimated severity is {severity.lower()}."
    )


def build_medical_assistant_text(
    predicted_class: str,
    confidence: float,
    severity: str,
    quality: str,
) -> str:
    """Create a non-diagnostic assistant-style recommendation."""
    return (
        f"AI assistant note: This scan is flagged as {predicted_class} "
        f"with {confidence * 100:.1f}% confidence. Severity is estimated as {severity}. "
        f"Image quality is {quality}. This output should be reviewed alongside "
        f"radiology findings, patient history, and follow-up imaging when indicated."
    )


def determine_focus_region(class_idx: int, heatmap_metadata: dict[str, Any] | None) -> str:
    """Infer a coarse anatomical focus description from heatmap metadata."""
    if not heatmap_metadata:
        return "a non-specific intracranial region"

    y_center = heatmap_metadata.get("center_y", 0.5)
    x_center = heatmap_metadata.get("center_x", 0.5)

    vertical = "frontal" if y_center < 0.35 else "parietal" if y_center < 0.65 else "occipital"
    horizontal = "left" if x_center < 0.4 else "right" if x_center > 0.6 else "central"
    return f"the {horizontal} {vertical} lobe region"

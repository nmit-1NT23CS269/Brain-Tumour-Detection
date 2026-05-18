"""PDF reporting for NeuroScan AI."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


NAVY = colors.HexColor("#09182c")
BLUE = colors.HexColor("#0f4c81")
ACCENT = colors.HexColor("#16b9ff")
WHITE = colors.white
SILVER = colors.HexColor("#edf4fb")
GREY = colors.HexColor("#7b8ca2")
DARK = colors.HexColor("#1e2937")


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=20, textColor=WHITE, alignment=1),
        "section": ParagraphStyle("section", parent=base["Heading2"], fontSize=12, textColor=NAVY, spaceAfter=6),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=9, textColor=DARK, leading=14),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontSize=8, textColor=GREY, leading=12),
    }


def _numpy_to_rl_image(img_array: np.ndarray, width_cm: float, height_cm: float) -> RLImage:
    pil_img = PILImage.fromarray(img_array.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return RLImage(buf, width=width_cm * cm, height=height_cm * cm)


def _table(data, col_widths):
    table = Table(data, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("GRID", (0, 0), (-1, -1), 0.25, GREY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def generate_pdf_report(
    username: str,
    scan_date: str,
    original_img: np.ndarray,
    preprocessed_img: np.ndarray,
    gradcam_img: Optional[np.ndarray],
    prediction: str,
    confidence: float,
    probabilities: dict,
    class_info: dict,
    scan_id: int,
    segmentation_img: Optional[np.ndarray] = None,
    ai_summary: str | None = None,
    recommendations: list[str] | None = None,
    model_name: str | None = None,
    quality_label: str | None = None,
) -> bytes:
    """Generate an expanded PDF report with images and AI interpretation."""
    styles = _styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm)

    story = []
    header = Table(
        [[
            Paragraph("<b>NeuroScan AI</b><br/>Brain MRI Analysis Report", styles["title"]),
            Paragraph(
                f"<font color='white'>Scan ID: #{scan_id:06d}<br/>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</font>",
                styles["small"],
            ),
        ]],
        colWidths=[11 * cm, 6 * cm],
    )
    header.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("BOX", (0, 0), (-1, -1), 0, NAVY)]))
    story.extend([header, Spacer(1, 0.4 * cm)])

    story.append(Paragraph("Scan Summary", styles["section"]))
    summary_rows = [
        ["Field", "Value"],
        ["User", username],
        ["Scan date", scan_date],
        ["Prediction", prediction],
        ["Confidence", f"{confidence * 100:.2f}%"],
        ["Severity", class_info.get("severity", "Unknown")],
        ["Model", model_name or "NeuroScan model"],
        ["MRI quality", quality_label or "Not assessed"],
    ]
    story.extend([_table(summary_rows, [5 * cm, 11.5 * cm]), Spacer(1, 0.35 * cm)])

    story.append(Paragraph("Class Probabilities", styles["section"]))
    prob_rows = [["Class", "Probability"]]
    for cls, prob in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
        prob_rows.append([cls, f"{prob * 100:.2f}%"])
    story.extend([_table(prob_rows, [7 * cm, 9.5 * cm]), Spacer(1, 0.35 * cm)])

    story.append(Paragraph("Clinical Interpretation", styles["section"]))
    story.append(Paragraph(class_info.get("description", ""), styles["body"]))
    if ai_summary:
        story.append(Spacer(1, 0.15 * cm))
        story.append(Paragraph(ai_summary, styles["body"]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("MRI Visual Outputs", styles["section"]))
    image_row = [_numpy_to_rl_image(original_img, 5, 5), _numpy_to_rl_image(preprocessed_img, 5, 5)]
    label_row = [Paragraph("Original MRI", styles["small"]), Paragraph("Preprocessed MRI", styles["small"])]
    if gradcam_img is not None:
        image_row.append(_numpy_to_rl_image(gradcam_img, 5, 5))
        label_row.append(Paragraph("Grad-CAM", styles["small"]))
    if segmentation_img is not None:
        image_row.append(_numpy_to_rl_image(segmentation_img, 5, 5))
        label_row.append(Paragraph("Segmentation Overlay", styles["small"]))
    story.extend([Table([image_row, label_row]), Spacer(1, 0.35 * cm)])

    story.append(Paragraph("Recommendations", styles["section"]))
    for rec in recommendations or _default_recommendations(prediction):
        story.append(Paragraph(f"• {rec}", styles["body"]))
    story.extend([Spacer(1, 0.25 * cm), HRFlowable(width="100%", thickness=1, color=GREY), Spacer(1, 0.2 * cm)])
    story.append(
        Paragraph(
            "This report is AI-assisted and intended for educational and research use. "
            "It must not replace clinical diagnosis by a qualified medical professional.",
            styles["small"],
        )
    )

    doc.build(story)
    return buffer.getvalue()


def _default_recommendations(prediction: str) -> list[str]:
    mapping = {
        "Glioma": [
            "Urgent neuro-oncology review is recommended.",
            "Correlate with contrast-enhanced MRI and clinical symptoms.",
            "Consider histopathology and multidisciplinary treatment planning.",
        ],
        "Meningioma": [
            "Recommend neurosurgical consultation and interval imaging review.",
            "Correlate with tumour size, mass effect, and symptom profile.",
        ],
        "Pituitary": [
            "Recommend endocrine evaluation and dedicated pituitary MRI review.",
            "Assess for visual symptoms and hormonal imbalance.",
        ],
        "Normal": [
            "No tumour pattern detected by the model.",
            "Continue routine follow-up if clinically indicated.",
        ],
    }
    return mapping.get(prediction, ["Clinical review is recommended."])

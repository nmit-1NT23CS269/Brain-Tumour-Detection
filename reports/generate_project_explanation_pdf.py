"""Generate a demo-ready project explanation PDF without third-party dependencies."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "artifacts" / "reports" / "NeuroScan_AI_Project_Explanation.pdf"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

PAGE_WIDTH = 595
PAGE_HEIGHT = 842
LEFT_MARGIN = 50
TOP_MARGIN = 800
LINE_HEIGHT = 15
FONT_SIZE = 11
TITLE_SIZE = 20
SECTION_SIZE = 14
WRAP_WIDTH = 92


def escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_document_lines() -> list[tuple[str, str]]:
    date_str = datetime.now().strftime("%d %B %Y, %H:%M")
    lines: list[tuple[str, str]] = [
        ("title", "NeuroScan AI"),
        ("title", "Complete Project Explanation and Demo Guide"),
        ("body", f"Generated on {date_str}"),
        ("blank", ""),
        ("section", "1. Project Introduction"),
        ("body", "NeuroScan AI is a full-stack AI/ML web application for brain tumour detection from MRI images. It combines a Streamlit frontend, authentication, SQLite database storage, image preprocessing, deep learning prediction, Grad-CAM explainability, segmentation overlay, PDF reporting, and dashboard analytics in one platform."),
        ("body", "The project is designed as an educational and research-grade demonstration of how artificial intelligence can be integrated into a medical imaging workflow."),
        ("section", "2. Problem Statement"),
        ("body", "Brain tumours can be life-threatening and require timely analysis. MRI scans are widely used for tumour identification, but manual interpretation takes expert time. The goal of this project is to assist MRI-based screening by building a system that can accept MRI uploads, classify the scan, explain the model focus area, highlight suspicious regions, and generate a report."),
        ("section", "3. Main Objectives"),
        ("bullet", "Build a secure web interface for MRI upload and analysis"),
        ("bullet", "Classify MRI scans into Glioma, Meningioma, Pituitary, or Normal"),
        ("bullet", "Preprocess images to create model-ready input"),
        ("bullet", "Provide explainability with Grad-CAM"),
        ("bullet", "Highlight suspicious tumour regions using segmentation overlay"),
        ("bullet", "Generate PDF reports and save prediction history"),
        ("bullet", "Store user, scan, and analytics data in SQLite"),
        ("bullet", "Support future multi-model training and model comparison"),
        ("section", "4. Technologies Used"),
        ("bullet", "Python for the core implementation"),
        ("bullet", "Streamlit for the user interface"),
        ("bullet", "TensorFlow / Keras for deep learning"),
        ("bullet", "OpenCV for preprocessing and segmentation logic"),
        ("bullet", "NumPy and Pandas for data handling"),
        ("bullet", "Plotly, Matplotlib, and Seaborn for analytics and charts"),
        ("bullet", "SQLite for database storage"),
        ("bullet", "bcrypt for password hashing"),
        ("bullet", "Report generation utilities for downloadable scan documents"),
        ("section", "5. Current Project Structure"),
        ("bullet", "app.py: main Streamlit application and page router"),
        ("bullet", "frontend/: UI theming and presentation helpers"),
        ("bullet", "backend/: scan orchestration service"),
        ("bullet", "authentication/: registration, login, OTP, session logic"),
        ("bullet", "database/: SQLite schema and CRUD functions"),
        ("bullet", "preprocessing/: image validation, preprocessing, dataset management"),
        ("bullet", "model/: app-facing inference layer"),
        ("bullet", "models/: training registry, evaluation, comparison pipeline"),
        ("bullet", "explainability/: Grad-CAM generation"),
        ("bullet", "segmentation/: tumour highlighting overlay"),
        ("bullet", "reports/: PDF-related generation scripts"),
        ("bullet", "utils/: shared helpers for paths, logging, summaries, and confidence interpretation"),
        ("section", "6. Dataset Details"),
        ("body", "The extracted MRI archive contains 7,200 images in total. The observed structure is Training and Testing folders with four categories."),
        ("bullet", "Training/glioma: 1400"),
        ("bullet", "Training/meningioma: 1400"),
        ("bullet", "Training/notumor: 1400"),
        ("bullet", "Training/pituitary: 1400"),
        ("bullet", "Testing/glioma: 400"),
        ("bullet", "Testing/meningioma: 400"),
        ("bullet", "Testing/notumor: 400"),
        ("bullet", "Testing/pituitary: 400"),
        ("section", "7. User Workflow"),
        ("body", "The project follows a realistic end-user workflow. This is important to explain in the demo because it shows the system is more than a standalone ML notebook."),
        ("bullet", "User registers and verifies the account with OTP"),
        ("bullet", "User logs in securely"),
        ("bullet", "User uploads an MRI image"),
        ("bullet", "The image is validated and previewed"),
        ("bullet", "The MRI is preprocessed into model input form"),
        ("bullet", "The deep learning model predicts the class"),
        ("bullet", "Grad-CAM and segmentation visuals are generated"),
        ("bullet", "Confidence, severity, quality, and AI summary are shown"),
        ("bullet", "Prediction is stored in SQLite"),
        ("bullet", "User downloads a report"),
        ("section", "8. Authentication Module Explanation"),
        ("body", "The authentication layer improves realism and security. Registration collects username, email, phone, and password. Passwords are hashed using bcrypt. OTP records are generated and verified to simulate phone-based validation. Streamlit session state tracks logged-in users, which restricts sensitive pages like upload, dashboard, and history."),
        ("section", "9. Database Design"),
        ("body", "SQLite is used because it is lightweight, portable, and easy for a demo project. The database schema has been expanded beyond the original project."),
        ("bullet", "users table stores account details and hashed passwords"),
        ("bullet", "otp_records stores OTP values, expiry, and usage state"),
        ("bullet", "patient_records stores optional patient metadata"),
        ("bullet", "predictions stores file details, class result, probabilities, severity, quality, summaries, and model info"),
        ("bullet", "model_versions stores best-model metadata for version tracking"),
        ("bullet", "analytics_events stores structured scan events for future analytics"),
        ("section", "10. Image Preprocessing Pipeline"),
        ("body", "Image preprocessing is necessary because uploaded MRI images can differ in shape, scale, and format. The pipeline standardizes all inputs before inference."),
        ("bullet", "Validate file type and size"),
        ("bullet", "Convert to RGB if needed"),
        ("bullet", "Resize to 224 x 224"),
        ("bullet", "Apply Gaussian blur for mild denoising"),
        ("bullet", "Normalize pixel values into the 0 to 1 range"),
        ("bullet", "Add batch dimension for Keras input"),
        ("bullet", "Estimate MRI quality using contrast and sharpness heuristics"),
        ("section", "11. Deep Learning Model Explanation"),
        ("body", "The app-facing prediction layer loads the current best model for inference. The training design supports transfer learning and multiple backbone families such as VGG16, ResNet50, EfficientNetB0, and MobileNetV2."),
        ("bullet", "Transfer learning uses pretrained ImageNet features"),
        ("bullet", "A classification head converts extracted features into four tumour classes"),
        ("bullet", "Softmax output returns probabilities for all classes"),
        ("bullet", "The highest probability class becomes the final prediction"),
        ("section", "12. Predicted Classes"),
        ("bullet", "Glioma: glial-cell tumour, generally treated as a high-risk class"),
        ("bullet", "Meningioma: tumour related to meninges, often benign but clinically relevant"),
        ("bullet", "Pituitary: tumour in the pituitary region, often linked to endocrine effects"),
        ("bullet", "Normal: no tumour pattern detected by the model"),
        ("section", "13. Grad-CAM Explainability"),
        ("body", "Grad-CAM helps explain why the model predicted a class. Instead of producing only a label, it highlights the image regions that influenced the model most strongly. This makes the system more interpretable and demo-friendly."),
        ("bullet", "Feature maps from the last convolutional layer are extracted"),
        ("bullet", "Gradients are computed for the predicted class"),
        ("bullet", "A heatmap is generated from weighted activations"),
        ("bullet", "The heatmap is overlaid on the MRI"),
        ("bullet", "The app derives a coarse focus-region description from the heatmap center"),
        ("section", "14. Segmentation Feature"),
        ("body", "The segmentation module uses OpenCV heuristics instead of a U-Net for now. It is still useful in a demo because it highlights suspicious bright regions and outlines them with contours."),
        ("bullet", "MRI is converted to grayscale"),
        ("bullet", "Adaptive thresholding is applied"),
        ("bullet", "Morphological operations clean the mask"),
        ("bullet", "Contours are extracted"),
        ("bullet", "Small noise regions are ignored"),
        ("bullet", "A binary mask and overlay image are generated"),
        ("section", "15. AI Summary and Confidence Interpretation"),
        ("body", "The project converts raw prediction outputs into human-readable explanation. Confidence values are mapped to bands like Very high, High, Moderate, or Low confidence. Severity is estimated from the predicted class and confidence. The system also generates a concise AI summary and assistant-style recommendation text."),
        ("section", "16. Dashboard Analytics"),
        ("body", "The dashboard adds a full-stack analytics layer to the project."),
        ("bullet", "Total scans"),
        ("bullet", "Tumour versus normal counts"),
        ("bullet", "Prediction distribution charts"),
        ("bullet", "MRI quality breakdown"),
        ("bullet", "Recent scan history"),
        ("bullet", "Model usage information"),
        ("section", "17. PDF Reporting Feature"),
        ("body", "The reporting flow packages the scan output into a downloadable document so the application behaves more like a practical AI platform."),
        ("bullet", "User and scan details"),
        ("bullet", "Prediction and confidence"),
        ("bullet", "Severity and model information"),
        ("bullet", "Probability distribution"),
        ("bullet", "Original MRI and preprocessed MRI"),
        ("bullet", "Grad-CAM image"),
        ("bullet", "Segmentation overlay"),
        ("bullet", "AI summary and recommendations"),
        ("section", "18. Training Pipeline Design"),
        ("body", "The upgraded training pipeline is designed to support automatic dataset preparation, corrupted file handling, image organization, train/validation/test splitting, transfer learning, model comparison, and best-model selection."),
        ("bullet", "ZIP extraction"),
        ("bullet", "Image validation"),
        ("bullet", "Folder normalization"),
        ("bullet", "Train/validation/test splits"),
        ("bullet", "Data augmentation"),
        ("bullet", "EarlyStopping"),
        ("bullet", "ReduceLROnPlateau"),
        ("bullet", "Checkpoint saving"),
        ("bullet", "TensorBoard and CSV logs"),
        ("bullet", "Best model chosen by evaluation metrics such as F1 score"),
        ("section", "19. Strengths of the Project"),
        ("bullet", "Combines AI and web development in one application"),
        ("bullet", "Provides explainability and segmentation, not just classification"),
        ("bullet", "Includes secure user workflow and stored history"),
        ("bullet", "Uses modular architecture for easier maintenance"),
        ("bullet", "Demonstrates a realistic medical-AI platform flow"),
        ("section", "20. Limitations"),
        ("bullet", "This is an educational and research project, not a clinical diagnostic system"),
        ("bullet", "Segmentation is heuristic and not a medical-grade segmentation model"),
        ("bullet", "Model performance depends on training quality and dataset characteristics"),
        ("bullet", "Hospital deployment would require stronger security, validation, and compliance"),
        ("section", "21. Future Enhancements"),
        ("bullet", "Integrate true U-Net segmentation"),
        ("bullet", "Support DICOM images"),
        ("bullet", "Add doctor/admin role-based access"),
        ("bullet", "Deploy with Docker and cloud infrastructure"),
        ("bullet", "Add REST APIs"),
        ("bullet", "Extend to multi-slice or 3D MRI analysis"),
        ("section", "22. Demo Presentation Flow"),
        ("body", "A simple way to present the project is: problem statement, objectives, technology stack, live application workflow, prediction result, Grad-CAM and segmentation output, dashboard analytics, architecture, and finally future scope."),
        ("section", "23. Common Viva Questions"),
        ("bullet", "Why did you use Streamlit? Because it allows quick AI dashboard development in Python."),
        ("bullet", "Why 224 x 224? Because it matches many pretrained CNN backbones."),
        ("bullet", "Why transfer learning? It improves performance and reduces training time."),
        ("bullet", "What is Grad-CAM? It is a visual explanation method for CNN predictions."),
        ("bullet", "Why SQLite? It is lightweight and easy for demo deployment."),
        ("bullet", "How are passwords stored? Using bcrypt hashing, not plain text."),
        ("bullet", "What is the limitation? The project is educational and not a final clinical tool."),
        ("section", "24. Conclusion"),
        ("body", "NeuroScan AI is a complete educational AI platform for brain MRI analysis. It is strong for demo and viva because it combines machine learning, image preprocessing, explainability, segmentation, authentication, database management, analytics, and reporting in one coherent system."),
    ]
    return lines


def render_pages(lines: list[tuple[str, str]]) -> list[list[tuple[str, str, int]]]:
    pages: list[list[tuple[str, str, int]]] = []
    current_page: list[tuple[str, str, int]] = []
    y = TOP_MARGIN

    def flush():
        nonlocal current_page, y
        if current_page:
            pages.append(current_page)
        current_page = []
        y = TOP_MARGIN

    for kind, text in lines:
        if kind == "blank":
            y -= LINE_HEIGHT
            continue

        if kind == "title":
            font_size = TITLE_SIZE
            wrapped = textwrap.wrap(text, width=55) or [text]
        elif kind == "section":
            font_size = SECTION_SIZE
            wrapped = textwrap.wrap(text, width=70) or [text]
        elif kind == "bullet":
            font_size = FONT_SIZE
            wrapped = textwrap.wrap(text, width=WRAP_WIDTH, subsequent_indent="  ") or [text]
        else:
            font_size = FONT_SIZE
            wrapped = textwrap.wrap(text, width=WRAP_WIDTH) or [text]

        needed = len(wrapped) * LINE_HEIGHT + (10 if kind == "section" else 0)
        if y - needed < 55:
            flush()

        if kind == "section":
            y -= 8

        for line in wrapped:
            current_page.append((kind, line, y))
            y -= LINE_HEIGHT

        if kind in {"title", "section"}:
            y -= 4

    flush()
    return pages


def make_content_stream(page_items: list[tuple[str, str, int]]) -> bytes:
    parts = ["BT"]
    for kind, text, y in page_items:
        safe = escape_pdf_text(text)
        if kind == "title":
            font = "F2"
            size = TITLE_SIZE
            x = LEFT_MARGIN
        elif kind == "section":
            font = "F2"
            size = SECTION_SIZE
            x = LEFT_MARGIN
        elif kind == "bullet":
            font = "F1"
            size = FONT_SIZE
            x = LEFT_MARGIN + 8
        else:
            font = "F1"
            size = FONT_SIZE
            x = LEFT_MARGIN
        parts.append(f"/{font} {size} Tf 1 0 0 1 {x} {y} Tm ({safe}) Tj")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1", errors="replace")


def generate_pdf(output_path: Path) -> None:
    pages = render_pages(build_document_lines())
    objects: list[bytes] = []

    def add_object(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    font1_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font2_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    page_ids: list[int] = []
    content_ids: list[int] = []

    for page in pages:
        content = make_content_stream(page)
        content_id = add_object(b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream")
        content_ids.append(content_id)
        page_ids.append(0)

    kids_placeholder = " ".join(f"{9999} 0 R" for _ in pages).encode()
    pages_id = add_object(b"<< /Type /Pages /Kids [ " + kids_placeholder + b" ] /Count " + str(len(pages)).encode() + b" >>")

    for idx, _page in enumerate(pages):
        page_obj = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font1_id} 0 R /F2 {font2_id} 0 R >> >> "
            f"/Contents {content_ids[idx]} 0 R >>"
        ).encode()
        page_ids[idx] = add_object(page_obj)

    kids = " ".join(f"{pid} 0 R" for pid in page_ids).encode()
    objects[pages_id - 1] = b"<< /Type /Pages /Kids [ " + kids + b" ] /Count " + str(len(page_ids)).encode() + b" >>"
    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode()
    )

    output_path.write_bytes(pdf)


def main() -> None:
    generate_pdf(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()

# NeuroScan AI (Brain-Tumour-Detection)
Capstone Project – Brain Tumor Detection using Deep Learning on MRI Images

NeuroScan AI is a Streamlit-based brain MRI analysis platform for tumour classification, explainability, segmentation overlays, scan reporting, and analytics. The upgraded project preserves the original login, upload, prediction, dashboard, and PDF workflow while expanding the backend into a cleaner production-style architecture.

## Highlights

- Streamlit frontend with a darker medical dashboard theme
- Authentication with OTP verification and session management
- SQLite database with scan history, patient records, analytics events, and model version tracking
- MRI preprocessing with resize, normalization, Gaussian blur, RGB conversion, and quality heuristics
- Multi-model transfer learning support:
  - `VGG16`
  - `ResNet50`
  - `EfficientNetB0`
  - `MobileNetV2`
- Grad-CAM explainability
- OpenCV-based tumour segmentation overlays
- AI-generated confidence interpretation, severity estimate, and medical assistant summary
- PDF report generation with original MRI, preprocessed MRI, Grad-CAM, segmentation, and recommendations
- Model comparison dashboard hooks driven by training artifacts

## Project Structure

```text
brain_tumour_detection/
├── app.py
├── authentication/
├── backend/
├── data/
├── database/
├── explainability/
├── frontend/
├── model/
├── models/
├── preprocessing/
├── reports/
├── segmentation/
├── utils/
├── requirements.txt
├── SETUP.md
├── TRAINING.md
└── DEPLOYMENT.md
```

## Dataset Status

The uploaded MRI archive has already been extracted into:

- [data/raw/archive](D:/MAJOR%20PROJRCT/brain_tumour_detection/data/raw/archive)

Detected structure:

- `Training/glioma`: 1400
- `Training/meningioma`: 1400
- `Training/notumor`: 1400
- `Training/pituitary`: 1400
- `Testing/glioma`: 400
- `Testing/meningioma`: 400
- `Testing/notumor`: 400
- `Testing/pituitary`: 400

## Recommended Runtime

For the TensorFlow 2.x stack in this project, use:

- Python `3.10` or `3.11`
- TensorFlow `2.15.x`

Python 3.12+ is not a safe default here because TensorFlow 2.15 support is narrower there.

## Quick Start

See:

- [SETUP.md](/D:/MAJOR PROJRCT/brain_tumour_detection/SETUP.md)
- [TRAINING.md](/D:/MAJOR PROJRCT/brain_tumour_detection/TRAINING.md)
- [DEPLOYMENT.md](/D:/MAJOR PROJRCT/brain_tumour_detection/DEPLOYMENT.md)

## Medical Disclaimer

This project is for educational and research purposes only. It must not be used as a sole diagnostic system or as a substitute for qualified clinical judgment.

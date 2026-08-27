# 🧠 NeuroScan AI — Advanced Brain Tumor Detection & Clinical Diagnostics Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![TensorFlow 2.15](https://img.shields.io/badge/ML-TensorFlow%202.15-FF6F00.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**NeuroScan AI** is a medical-grade Brain MRI Analysis and Clinical Workflow platform designed for high-accuracy brain tumor detection, deep learning explainability (Grad-CAM), OpenCV tumor segmentation overlays, specialist doctor-patient collaboration, and AI-assisted diagnostics.

---

## 🌟 Key System Capabilities

### 1. 🔬 Deep Learning MRI Classification & Explainability
- **4-Class Tumour Detection**: Classifies MRI scans into **Glioma**, **Meningioma**, **Pituitary**, or **No Tumor (Normal)**.
- **Multi-Model Transfer Learning Registry**: Supports `VGG16`, `ResNet50`, `EfficientNetB0`, and `MobileNetV2`.
- **Grad-CAM Visual Heatmaps**: Highlights exact neural focus areas driving the model's classification decision.
- **OpenCV Tumor Segmentation**: Auto-extracts tumor boundaries, contour masks, and calculates percentage area coverage.
- **Automated PDF Diagnostic Reports**: Generates comprehensive PDF reports including original MRI, preprocessed scan, Grad-CAM heatmap, tumor segmentation overlay, AI summary, and doctor clinical recommendations.

### 2. 🤖 Mastishka AI Medical Assistant
- Integrated offline rule-based and knowledge-driven medical chatbot (`chatbot/`).
- Provides real-time guidance on MRI scan results, tumor terminology (Glioma, Meningioma, Pituitary), platform features, and next clinical steps.

### 3. 👨‍⚕️ Mandatory Doctor Credential Upload & Admin Verification
- **Doctor Registration Security**: Doctors must upload valid medical credentials (**Medical Degree Certificate** and **Medical Registration License**) in PDF, PNG, or JPG formats.
- **Multi-State Application Lifecycle**: Applications progress through `PENDING` → Admin Review → `APPROVED` / `ACTIVE`, `REJECTED`, or `ADDITIONAL_INFO_REQUIRED`.
- **Admin Verification Console**: Administrators can inspect uploaded PDF degree/license certificates, download credentials, approve applications, or request document replacements with custom feedback notes.

### 4. 📋 Doctor Add Patient & Scan Patient Workflows
- **Unique Patient Identifier (`PAT-XXXXXX`)**: Every patient receives an immutable backend-generated patient code (e.g. `PAT-000014`).
- **`+ Add Patient`**:
  - **Search Existing Patient**: Search by Email, Phone, or Patient Code (`PAT-XXXXXX`) to establish active associations without creating duplicate accounts.
  - **Register New Patient**: Full registration form (Name, Email, Phone, DOB, Gender, Address, Medical Notes) creating patient accounts and active doctor relationships.
- **`🔬 Scan Patient`**: Approved doctors can initiate brain MRI scans specifically linked to authorized patient records. Results automatically appear in the patient's personal report history.
- **Strict Backend Authorization**: Doctors cannot scan or view arbitrary patient records unless an active relationship is verified by backend middleware (`can_doctor_scan_patient`).

### 5. 👥 Role-Based Access Control (RBAC)
- **Patient Workspace**: Upload MRI scans, view Grad-CAM heatmaps, download PDF reports, assign specialist doctors, and view attached doctor clinical notes.
- **Doctor Clinical Workspace**: Manage patient roster, perform patient scans, inspect heatmaps, and append signed clinical observations & recommendations.
- **Admin Console**: Doctor verification portal, user account management, platform metrics, and immutable audit logging (`audit_logs`).

---

## 🔑 Demo Accounts for Faculty & Reviewers

To make evaluation simple and seamless, **NeuroScan AI** includes a pre-seeded demonstration environment controlled by `DEMO_MODE=true`.

Below are the pre-configured **Demo Credentials** for manual testing:

| User Role | Email Address | Password | Account Status | Pre-Populated Demo Content |
|---|---|---|---|---|
| **👤 Demo Patient** | `patient.demo@example.com` | `Patient@123` | `ACTIVE` | Patient Code `PAT-000024`, 1 Sample Brain Scan Report (Glioma 96.8%), Assigned Specialist Doctor. |
| **👨‍⚕️ Verified Doctor** | `doctor.demo@example.com` | `Doctor@123` | `APPROVED` | `MBBS, MD Neuro-oncology`, `NMC-DEMO-9948`, uploaded PDF Degree & License certificates, "My Patients" workspace containing `demo_patient`. |
| **⏳ Pending Doctor** | `pending.doctor@example.com` | `Doctor@123` | `PENDING` | `MBBS, MD Pediatric Neurology`, `NMC-PENDING-7712`, uploaded PDF verification certificates for testing the Admin Verification & Approval workflow. |
| **⚙️ Demo Admin** | `admin.demo@example.com` | `Admin@123` | `ACTIVE` | Full administrative privileges, Doctor Approval Queue, User Control, Metrics, & Audit Trail. |

> ⚡ **One-Click Quick Login**: On the Sign In page, click **"Login as Demo Patient"**, **"Login as Verified Doctor"**, or **"Login as Demo Admin"** to authenticate instantly without typing credentials!

---

## 📁 Repository Structure

```text
brain_tumour_detection/
├── app.py                      # Main Streamlit web application & role routing
├── authentication/             # Password hashing, bcrypt verification, OTP, & RBAC auth
├── backend/                    # Core scan service execution & ML inference handler
├── chatbot/                    # Mastishka AI Medical Assistant module & knowledge base
├── data/                       # Raw & processed MRI dataset directory
├── database/                   # SQLite database schema, CRUD helpers, & migration engine
├── explainability/             # Grad-CAM heatmap generation module
├── frontend/                   # Custom dark-theme CSS styling & UI layout tiles
├── model/                      # Transfer learning CNN model architecture wrappers
├── preprocessing/              # Image normalization, resizing, quality checks, & histograms
├── reports/                    # FPDF automated PDF diagnostic report generator
├── segmentation/               # OpenCV contour extraction & tumor area calculation
├── uploads/                    # Secure storage for uploaded doctor medical certificates
├── brain_tumour.db             # Local SQLite database file
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── SETUP.md                    # Environment setup guide
├── TRAINING.md                 # Model training workflow guide
└── DEPLOYMENT.md               # Production deployment instructions
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python **3.10** or **3.11** (Recommended for TensorFlow 2.15 compatibility)
- Git

### 2. Installation

Clone the repository and set up a virtual environment:

```powershell
# Clone the repository
git clone https://github.com/nmit-1NT23CS269/Brain-Tumour-Detection.git
cd Brain-Tumour-Detection

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Upgrade pip & install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch Application

Run the Streamlit server:

```powershell
python -m streamlit run app.py
```

Open your browser and navigate to: **`http://localhost:8501`**

---

## ⚙️ Environment Configuration

| Variable | Default | Description |
|---|---|---|
| `DEMO_MODE` | `true` | When `true`, renders One-Click Demo Credentials Card on the Sign In page. Set `DEMO_MODE=false` for production builds. |

---

## 🧪 Automated Testing

Run system test scripts to verify backend logic, RBAC security, and document verification:

```powershell
# Test Doctor Mandatory Credentials & Admin Verification Workflow
python scratch/test_doctor_credentials_workflow.py

# Test Doctor Add Patient & Scan Patient Integration
python scratch/test_doctor_patient_scan_workflow.py

# Test Demo Accounts Authentication & RBAC Integrity
python scratch/test_demo_credentials_workflow.py
```

---

## ⚠️ Medical Disclaimer

This application is designed for **educational, capstone demonstration, and research purposes only**. It should not be used as a primary diagnostic tool or as a replacement for professional clinical evaluation by certified medical specialists.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more details.

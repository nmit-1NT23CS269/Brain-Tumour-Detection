"""NeuroScan AI Streamlit application."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from authentication import (
    clear_session,
    confirm_otp,
    current_user_id,
    current_username,
    current_user_role,
    current_user_status,
    is_admin,
    is_doctor,
    is_patient,
    is_logged_in,
    login_user,
    register_user,
    send_otp,
    set_session,
    validate_document_file,
    save_uploaded_doctor_document,
    can_doctor_scan_patient,
)
from backend import analyze_uploaded_scan
from chatbot import SUGGESTED_QUESTIONS, get_response as mastishka_response
from database import (
    create_patient_record,
    get_dashboard_stats,
    get_user_predictions,
    initialize_database,
    save_prediction,
    get_pending_doctors,
    get_all_doctors,
    update_doctor_verification,
    get_all_users,
    update_user_status,
    assign_doctor_to_patient,
    get_assigned_doctors_for_patient,
    get_assigned_patients_for_doctor,
    is_patient_assigned_to_doctor,
    add_clinical_note,
    get_clinical_notes_for_prediction,
    get_doctor_assigned_predictions,
    log_audit_event,
    get_audit_logs,
    get_prediction_by_id,
    get_user_by_id,
    save_doctor_document,
    replace_doctor_document,
    get_doctor_documents,
    get_doctor_document_by_id,
    get_patient_code,
    parse_patient_code,
    find_patient_by_identifier,
    create_doctor_added_patient,
    get_doctor_assigned_patients_with_details,
)
from frontend import inject_theme
from model import CLASS_INFO, CLASS_NAMES
from preprocessing import plot_pixel_histogram, plot_preprocessing_steps, validate_image
from reports import generate_pdf_report


matplotlib.use("Agg")
initialize_database()

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()


def _card_start():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)


def _card_end():
    st.markdown("</div>", unsafe_allow_html=True)


def _metric_tile(column, value: str, label: str, color: str = "#16b9ff"):
    column.markdown(
        f"""
        <div class="metric-tile">
          <div class="metric-value" style="color:{color};">{value}</div>
          <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="app-card" style="padding:1.6rem 1.8rem;">
          <div class="hero-title">{title}</div>
          <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## NeuroScan AI")
        st.caption("Medical MRI intelligence platform")

        if is_logged_in():
            role = current_user_role()
            status = current_user_status()
            role_label = f" ({role.capitalize()})" if role else ""
            st.success(f"Signed in as {current_username()}{role_label}")

            if role == "admin":
                pages = {
                    "👤 My Profile": "profile",
                    "Admin Console": "admin_dashboard",
                    "Audit Logs": "audit_logs",
                    "Upload & Scan": "upload",
                    "System Dashboard": "dashboard",
                    "Scan History": "history",
                    "Training Hub": "training",
                    "🧠 Mastishka AI": "mastishka",
                    "About": "about",
                }
            elif role == "doctor":
                if status in ("active", "approved"):
                    pages = {
                        "👤 My Profile": "profile",
                        "Doctor Workspace": "doctor_dashboard",
                        "Training Hub": "training",
                        "🧠 Mastishka AI": "mastishka",
                        "About": "about",
                    }
                else:
                    pages = {
                        "👤 My Profile": "profile",
                        "Application Status": "doctor_pending",
                        "🧠 Mastishka AI": "mastishka",
                        "About": "about",
                    }
            else:  # patient
                pages = {
                    "Home": "home",
                    "👤 My Profile": "profile",
                    "Upload & Scan": "upload",
                    "Dashboard": "dashboard",
                    "Scan History": "history",
                    "👨‍⚕️ Doctor Consultation": "consultation",
                    "Training Hub": "training",
                    "🧠 Mastishka AI": "mastishka",
                    "About": "about",
                }
        else:
            pages = {
                "Home": "home",
                "Login": "login",
                "Register": "register",
                "🧠 Mastishka AI": "mastishka",
                "About": "about",
            }

        st.session_state.setdefault("page", "home")
        for label, page_key in pages.items():
            if st.button(label, use_container_width=True, type="primary" if st.session_state["page"] == page_key else "secondary"):
                st.session_state["page"] = page_key
                st.rerun()

        if is_logged_in() and st.button("Logout", use_container_width=True):
            user_id = current_user_id()
            log_audit_event(user_id, "user_logout", "auth")
            clear_session()
            st.session_state["page"] = "home"
            st.rerun()


def page_home() -> None:
    _hero("NeuroScan AI", "Professional brain MRI tumour analysis with explainability, segmentation, and reporting")
    c1, c2, c3, c4 = st.columns(4)
    _metric_tile(c1, "4", "Supported Classes")
    _metric_tile(c2, "4", "Model Families", "#00d4a5")
    _metric_tile(c3, "224x224", "Preprocessing", "#fcbf49")
    _metric_tile(c4, "Grad-CAM", "Explainability", "#ff6b6b")

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        _card_start()
        st.subheader("Platform Workflow")
        st.markdown(
            """
            1. Create a verified account and sign in.
            2. Upload an MRI image and optionally attach patient details.
            3. The system preprocesses the scan, estimates MRI quality, runs inference, and generates Grad-CAM and segmentation overlays.
            4. Results are logged to SQLite, shown in the dashboard, and exported as a PDF report.
            """
        )
        _card_end()

    with right:
        _card_start()
        st.subheader("Advanced Features")
        st.markdown(
            """
            - Transfer-learning model registry with VGG16, ResNet50, EfficientNetB0, and MobileNetV2
            - Tumour segmentation overlay using OpenCV contour extraction
            - AI summary, confidence interpretation, and MRI quality heuristics
            - Dashboard-ready analytics and model version tracking
            """
        )
        _card_end()


def page_register() -> None:
    _hero("Create Account", "Register as a Patient or apply for Doctor credentials")
    _card_start()

    account_type = st.radio("Select Account Type", ["Patient", "Medical Doctor"], horizontal=True)
    role = "doctor" if account_type == "Medical Doctor" else "patient"

    if role == "doctor":
        st.info("📋 Doctor Registration requires submitting your professional details, medical qualification, and uploading mandatory verification documents (Medical Degree Certificate & Medical License).")

    with st.form("register_form"):
        username = st.text_input("Username *")
        email = st.text_input("Email Address *")
        phone = st.text_input("Phone Number (with country code) *", value="+91")

        specialization = None
        qualification = None
        license_no = None
        hospital = None
        years_exp = None
        address = None

        deg_file = None
        lic_file = None
        id_file = None

        if role == "doctor":
            st.markdown("---")
            st.subheader("Doctor Professional & Qualification Details")
            specialization = st.text_input("Medical Specialization *", placeholder="e.g. Neuro-oncology, Radiology, Neurology")
            qualification = st.text_input("Medical Qualification Degree *", placeholder="e.g. MBBS, MD, DM Neuro-oncology")
            license_no = st.text_input("Medical License / Registration Number *", placeholder="e.g. NMC-994821")
            hospital = st.text_input("Hospital / Clinic Affiliation *", placeholder="e.g. City Central Hospital")
            years_exp_input = st.number_input("Years of Professional Experience", min_value=0, max_value=60, value=5)
            years_exp = int(years_exp_input)
            address = st.text_area("Professional / Clinic Address", placeholder="Enter hospital or clinic address...")

            st.markdown("---")
            st.subheader("📎 Mandatory Medical Verification Documents")
            st.caption("Upload clear PDF, PNG, JPG, or JPEG documents (Max 10 MB per file).")

            deg_file = st.file_uploader("1. Medical Degree Certificate * (Required)", type=["pdf", "png", "jpg", "jpeg"], key="deg_file")
            lic_file = st.file_uploader("2. Medical Registration Certificate / License * (Required)", type=["pdf", "png", "jpg", "jpeg"], key="lic_file")
            id_file = st.file_uploader("3. Professional / Government ID (Optional)", type=["pdf", "png", "jpg", "jpeg"], key="id_file")
            st.markdown("---")

        password = st.text_input("Password *", type="password")
        confirm = st.text_input("Confirm password *", type="password")
        submitted = st.form_submit_button("Submit Registration Application")

    if submitted:
        if role == "doctor":
            if not deg_file or not lic_file:
                st.error("Submission failed: Both Medical Degree Certificate and Medical License Certificate are REQUIRED for doctor verification.")
                return

        success, message, user_id = register_user(
            username=username,
            email=email,
            phone=phone,
            password=password,
            confirm=confirm,
            role=role,
            specialization=specialization,
            qualification=qualification,
            license_no=license_no,
            hospital=hospital,
            years_experience=years_exp,
            address=address,
        )
        if success and user_id:
            st.success(message)
            if role == "doctor":
                if deg_file:
                    save_uploaded_doctor_document(deg_file, user_id, "degree")
                if lic_file:
                    save_uploaded_doctor_document(lic_file, user_id, "license")
                if id_file:
                    save_uploaded_doctor_document(id_file, user_id, "gov_id")

                st.success("✅ Medical credential documents uploaded successfully! Your application is now in PENDING status for administrative verification.")
            else:
                otp = send_otp(phone)
                st.session_state["pending_phone"] = phone
                st.session_state["otp_demo"] = otp
        else:
            st.error(message)

    if st.session_state.get("pending_phone"):
        st.info("Demo mode: OTP delivery is simulated. Check server logs for the code.")
        otp_value = st.text_input("Enter OTP", key="otp_input")
        if st.button("Verify OTP"):
            if confirm_otp(st.session_state["pending_phone"], otp_value):
                st.success("Phone number verified! You can now log in.")
                st.session_state.pop("pending_phone", None)
            else:
                st.error("Invalid or expired OTP.")
    _card_end()


def page_login() -> None:
    _hero("Sign In", "Access your MRI analysis workspace")
    _card_start()
    with st.form("login_form"):
        identifier = st.text_input("Username / Email / Phone", value=st.session_state.get("login_identifier_fill", ""))
        password = st.text_input("Password", type="password", value=st.session_state.get("login_password_fill", ""))
        submitted = st.form_submit_button("Login")

    if submitted:
        st.session_state.pop("login_identifier_fill", None)
        st.session_state.pop("login_password_fill", None)
        success, message, user = login_user(identifier, password)
        if success and user:
            set_session(user)
            role = user.get("role", "patient")
            status = user.get("status", "active")

            if role == "admin":
                st.session_state["page"] = "admin_dashboard"
            elif role == "doctor":
                if status in ("active", "approved"):
                    st.session_state["page"] = "doctor_dashboard"
                else:
                    st.session_state["page"] = "doctor_pending"
            else:
                st.session_state["page"] = "upload"

            st.success(message)
            st.rerun()
        else:
            st.error(message)
    _card_end()

    # CRITICAL SECURITY NOTE: DEMO_MODE renders automated login buttons with demo credentials.
    # THIS MUST REMAIN DISABLED (DEMO_MODE=false) FOR ANY REAL DEPLOYMENT TO PREVENT UNAUTHORIZED ACCESS.
    import os
    demo_mode = os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "yes")

    if demo_mode:
        demo_patient_email = os.getenv("DEMO_PATIENT_EMAIL", "patient.demo@example.com")
        demo_patient_password = os.getenv("DEMO_PATIENT_PASSWORD", "Patient@123")
        demo_doctor_email = os.getenv("DEMO_DOCTOR_EMAIL", "doctor.demo@example.com")
        demo_doctor_password = os.getenv("DEMO_DOCTOR_PASSWORD", "Doctor@123")
        demo_admin_email = os.getenv("DEMO_ADMIN_EMAIL", "admin.demo@example.com")
        demo_admin_password = os.getenv("DEMO_ADMIN_PASSWORD", "Admin@123")

        _card_start()
        st.subheader("🔑 Demonstration Credentials & Quick Login")
        st.caption("Faculty & Evaluator Demo Environment — One-Click Authentication into standard SQLite RBAC accounts.")

        dcol1, dcol2, dcol3 = st.columns(3)

        with dcol1:
            st.markdown("#### 👤 Demo Patient")
            st.write(f"**Email**: `{demo_patient_email}`  \n**Data**: 1 Sample Scan, Assigned Specialist")
            if st.button("⚡ Login as Demo Patient", use_container_width=True, type="secondary"):
                success, message, user = login_user(demo_patient_email, demo_patient_password)
                if success and user:
                    set_session(user)
                    st.session_state["page"] = "upload"
                    st.success("Authenticated as Demo Patient!")
                    st.rerun()
                else:
                    st.error(message)

        with dcol2:
            st.markdown("#### 👨‍⚕️ Verified Doctor — ✓ Approved")
            st.write(f"**Email**: `{demo_doctor_email}`  \n**Data**: Medical License, My Patients, Scan Workspace")
            if st.button("⚡ Login as Verified Doctor", use_container_width=True, type="primary"):
                success, message, user = login_user(demo_doctor_email, demo_doctor_password)
                if success and user:
                    set_session(user)
                    st.session_state["page"] = "doctor_dashboard"
                    st.success("Authenticated as Verified Doctor!")
                    st.rerun()
                else:
                    st.error(message)

        with dcol3:
            st.markdown("#### ⚙️ Demo Administrator")
            st.write(f"**Email**: `{demo_admin_email}`  \n**Data**: Doctor Approval Queue, User Control, Audit Logs")
            if st.button("⚡ Login as Demo Admin", use_container_width=True, type="secondary"):
                success, message, user = login_user(demo_admin_email, demo_admin_password)
                if success and user:
                    set_session(user)
                    st.session_state["page"] = "admin_dashboard"
                    st.success("Authenticated as Administrator!")
                    st.rerun()
                else:
                    st.error(message)

        st.markdown("---")
        st.info("💡 **Pending Doctor Application Demo**: To demonstrate Doctor Document Review and Verification, log in as **Admin**, click **Doctor Verifications**, and inspect `pending.doctor@example.com` (`dr_pending_demo`).")
        _card_end()



def _save_scan_result(result: dict, uploaded_file, patient_id: int | None = None, doctor_id: int | None = None) -> int:
    segmentation = result.get("segmentation") or {}
    target_user_id = patient_id if patient_id else current_user_id()
    return save_prediction(
        user_id=target_user_id,
        patient_id=patient_id,
        doctor_id=doctor_id,
        image_filename=uploaded_file.name,
        image_data=result["steps"]["file_bytes"],
        prediction=result["class"],
        confidence=result["confidence"],
        class_probs=result["probabilities"],
        severity=result["severity"],
        confidence_label=result["confidence_text"],
        quality_label=result["quality"]["label"],
        focus_region=result["focus_region"],
        ai_summary=result["ai_summary"],
        assistant_text=result["assistant_text"],
        model_name=result["model_name"],
        model_version=result["model_version"],
        segmentation_coverage=segmentation.get("coverage_ratio"),
    )


def _render_result_visuals(result: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    info = CLASS_INFO[result["class"]]
    _metric_tile(col1, result["class"], "Prediction", info["color"])
    _metric_tile(col2, f"{result['confidence'] * 100:.1f}%", "Confidence", "#16b9ff")
    _metric_tile(col3, result["severity"], "Severity", "#fcbf49")
    _metric_tile(col4, result["quality"]["label"], "MRI Quality", "#00d4a5")

    prob_df = pd.DataFrame(
        {"Class": list(result["probabilities"].keys()), "Probability": [value * 100 for value in result["probabilities"].values()]}
    )
    left, right = st.columns(2, gap="large")
    with left:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=result["confidence"] * 100,
                number={"suffix": "%"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": info["color"]}},
                title={"text": "Confidence"},
            )
        )
        st.plotly_chart(gauge, use_container_width=True)
    with right:
        fig = px.bar(prob_df, x="Probability", y="Class", orientation="h", color="Class", text="Probability")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)


def page_upload() -> None:
    if not is_logged_in():
        st.warning("Please log in to upload an MRI scan.")
        return

    _hero("Upload & Scan", "Run MRI preprocessing, classification, explainability, and segmentation")
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        _card_start()
        uploaded_file = st.file_uploader("Upload MRI scan", type=["jpg", "jpeg", "png"])
        show_preprocessing = st.checkbox("Show preprocessing visuals", value=True)
        enable_segmentation = st.checkbox("Enable segmentation overlay", value=True)
        enable_gradcam = st.checkbox("Enable Grad-CAM", value=True)
        st.markdown("Optional patient details")
        patient_name = st.text_input("Patient name")
        patient_age = st.number_input("Patient age", min_value=0, max_value=120, value=35)
        patient_gender = st.selectbox("Patient gender", ["Prefer not to say", "Female", "Male", "Other"])
        notes = st.text_area("Clinical notes")
        run_btn = st.button("Analyze MRI", type="primary", use_container_width=True)
        _card_end()

    with right:
        _card_start()
        st.subheader("Analysis Checklist")
        st.markdown(
            """
            - Input validation and RGB conversion
            - Resize to 224x224 and Gaussian denoising
            - Confidence scoring and class probabilities
            - MRI quality estimation and AI summary
            - Grad-CAM heatmap and segmentation overlay
            - PDF report generation and database logging
            """
        )
        _card_end()

    if not uploaded_file:
        return

    valid, message = validate_image(uploaded_file)
    if not valid:
        st.error(message)
        return

    st.image(uploaded_file, caption="MRI preview", use_column_width=True)
    if not run_btn:
        return

    patient_id = None
    if patient_name.strip():
        patient_id = create_patient_record(current_user_id(), patient_name.strip(), int(patient_age), patient_gender, notes or None)

    progress = st.progress(0, text="Starting NeuroScan pipeline")
    progress.progress(15, text="Preprocessing MRI")
    result = analyze_uploaded_scan(uploaded_file, enable_gradcam=enable_gradcam, enable_segmentation=enable_segmentation)
    progress.progress(70, text="Saving prediction and preparing report")
    prediction_id = _save_scan_result(result, uploaded_file, patient_id)
    progress.progress(100, text="Analysis complete")
    st.session_state["last_scan_result"] = result

    _card_start()
    st.subheader("Prediction Summary")
    st.success(f"Scan #{prediction_id:06d} classified as {result['class']}")
    st.write(result["ai_summary"])
    st.caption(result["assistant_text"])
    _card_end()

    _render_result_visuals(result)

    steps = result["steps"]
    if show_preprocessing:
        st.subheader("Preprocessing Review")
        fig = plot_preprocessing_steps(steps)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        hist_fig = plot_pixel_histogram(steps["normalized"])
        st.pyplot(hist_fig, use_container_width=True)
        plt.close(hist_fig)

    image_count = 2 + int(result.get("gradcam") is not None) + int(bool(result.get("segmentation")))
    image_cols = st.columns(image_count)
    image_cols[0].image(steps["original"], caption="Original")
    image_cols[1].image(steps["resized"], caption="Preprocessed")
    idx = 2
    if result.get("gradcam") is not None:
        image_cols[idx].image(result["gradcam"], caption="Grad-CAM")
        idx += 1
    if result.get("segmentation"):
        image_cols[idx].image(result["segmentation"]["overlay"], caption="Segmentation Overlay")

    pdf_bytes = generate_pdf_report(
        username=current_username(),
        scan_date=datetime.now().strftime("%d %B %Y %H:%M:%S"),
        original_img=steps["original"],
        preprocessed_img=steps["resized"],
        gradcam_img=result.get("gradcam"),
        segmentation_img=(result.get("segmentation") or {}).get("overlay"),
        prediction=result["class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        class_info={**CLASS_INFO[result["class"]], "severity": result["severity"]},
        scan_id=prediction_id,
        ai_summary=result["ai_summary"],
        model_name=result["model_name"],
        quality_label=result["quality"]["label"],
    )
    st.download_button(
        "Download PDF report",
        data=pdf_bytes,
        file_name=f"NeuroScan_Report_{prediction_id:06d}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def page_dashboard() -> None:
    if not is_logged_in():
        st.warning("Please log in to view the dashboard.")
        return

    stats = get_dashboard_stats(current_user_id())
    _hero("Dashboard", f"Welcome back, {current_username()}")
    c1, c2, c3, c4 = st.columns(4)
    _metric_tile(c1, str(stats["total"]), "Total Scans")
    _metric_tile(c2, str(sum(count for cls, count in stats["by_class"].items() if cls != "Normal")), "Tumour Cases", "#ff6b6b")
    _metric_tile(c3, str(stats["by_class"].get("Normal", 0)), "Normal Cases", "#00d4a5")
    _metric_tile(c4, str(len(stats.get("models_used", {}))), "Models Used", "#fcbf49")

    left, right = st.columns(2, gap="large")
    with left:
        _card_start()
        st.subheader("Prediction Distribution")
        if stats["by_class"]:
            pie = px.pie(values=list(stats["by_class"].values()), names=list(stats["by_class"].keys()), hole=0.45)
            st.plotly_chart(pie, use_container_width=True)
        else:
            st.info("No predictions yet.")
        _card_end()

    with right:
        _card_start()
        st.subheader("MRI Quality Breakdown")
        if stats["quality_breakdown"]:
            quality_df = pd.DataFrame({"Quality": list(stats["quality_breakdown"].keys()), "Count": list(stats["quality_breakdown"].values())})
            quality_fig = px.bar(quality_df, x="Quality", y="Count", color="Quality")
            st.plotly_chart(quality_fig, use_container_width=True)
        else:
            st.info("No quality analytics yet.")
        _card_end()

    if stats["recent"]:
        st.subheader("Recent Scans")
        st.dataframe(pd.DataFrame(stats["recent"]), use_container_width=True, hide_index=True)


def page_history() -> None:
    if not is_logged_in():
        st.warning("Please log in to view scan history.")
        return

    _hero("Scan History", "All stored MRI analysis records")
    records = get_user_predictions(current_user_id(), limit=200)
    if not records:
        st.info("No previous scans found.")
        return
    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True, hide_index=True)


def page_training() -> None:
    if not is_logged_in():
        st.warning("Please log in to view training analytics.")
        return

    _hero("Training Hub", "Model comparison and experiment outputs")
    summary_path = Path("logs/training_summary.json")
    if not summary_path.exists():
        _card_start()
        st.info("No training summary found yet. Run `python model/train.py --dataset_zip <path>` after restoring a working Python environment.")
        _card_end()
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    leaderboard = pd.DataFrame(summary["leaderboard"])
    best = summary["best_model"]

    c1, c2, c3, c4 = st.columns(4)
    _metric_tile(c1, best["model_name"], "Best Model")
    _metric_tile(c2, f"{best['accuracy'] * 100:.1f}%", "Accuracy", "#00d4a5")
    _metric_tile(c3, f"{best['precision'] * 100:.1f}%", "Precision", "#16b9ff")
    _metric_tile(c4, f"{best['f1_score'] * 100:.1f}%", "F1 Score", "#fcbf49")

    st.dataframe(leaderboard, use_container_width=True, hide_index=True)
    comparison = px.bar(
        leaderboard.melt(id_vars="model_name", value_vars=["accuracy", "precision", "recall", "f1_score"]),
        x="model_name",
        y="value",
        color="variable",
        barmode="group",
    )
    st.plotly_chart(comparison, use_container_width=True)
    st.json(summary["dataset_summary"])


def page_about() -> None:
    _hero("About NeuroScan AI", "Research-grade educational platform for brain MRI analysis")
    _card_start()
    st.markdown(
        """
        NeuroScan AI is a Streamlit-based full-stack brain tumour analysis platform built around:

        - Secure authentication with OTP verification
        - SQLite-backed patient and scan analytics
        - Multi-model deep learning support
        - Explainability via Grad-CAM
        - Segmentation overlays for visual emphasis
        - PDF reporting and dashboard analytics

        Medical disclaimer: this application is intended for educational and research use only and must not be used as a sole diagnostic tool.
        """
    )
    _card_end()


def page_doctor_pending() -> None:
    _hero("Doctor Application Status", "Medical Credential & Verification Dashboard")
    _card_start()
    doctor_id = current_user_id()
    user = get_user_by_id(doctor_id) if doctor_id else None
    status = user.get("status", current_user_status()) if user else current_user_status()
    vnotes = user.get("verification_notes", "") if user else ""

    if status == "pending":
        st.warning("⏳ Your doctor application is currently **PENDING** administrative verification.")
        st.markdown(
            f"""
            An administrator is reviewing your submitted medical license (**{user.get('license_no', 'N/A')}**) and uploaded degree certificates.

            - **Status**: Pending Administrative Verification
            - **Submitted Specialization**: {user.get('specialization', 'N/A')}
            - **Qualification Degree**: {user.get('qualification', 'N/A')}
            - **Hospital Affiliation**: {user.get('hospital', 'N/A')}

            *Access to patient records and the doctor clinical workspace remains restricted until verification is granted.*
            """
        )
    elif status == "additional_info_required":
        st.error("⚠️ **ADDITIONAL INFORMATION REQUIRED**")
        st.markdown(f"**Admin Feedback Notes**: {vnotes or 'Please re-upload your medical qualification or registration certificates as requested by administrative review.'}")

        st.markdown("---")
        st.subheader("📎 Replace Requested Verification Documents")
        st.caption("Upload updated PDF, PNG, or JPG document files. Submitting replacement documents will return your application to PENDING status for re-review.")

        doc_type_choice = st.selectbox(
            "Select Document to Replace",
            ["degree", "license", "gov_id"],
            format_func=lambda x: "Medical Degree Certificate" if x == "degree" else ("Medical Registration License" if x == "license" else "Government ID"),
        )
        replacement_file = st.file_uploader("Upload Replacement Document", type=["pdf", "png", "jpg", "jpeg"], key="repl_file")

        if st.button("Submit Replacement Document"):
            if replacement_file:
                ok, msg, doc_info = save_uploaded_doctor_document(replacement_file, doctor_id, doc_type_choice, is_replacement=True)
                if ok:
                    update_user_status(doctor_id, "pending")
                    log_audit_event(doctor_id, "additional_info_resubmitted", resource=f"user:{doctor_id}", details={"doc_type": doc_type_choice})
                    st.success("Replacement document uploaded successfully! Your application status has been returned to PENDING for admin review.")
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.error("Please select a file to upload.")

    elif status == "rejected":
        st.error("❌ Your doctor application was **REJECTED** by administrative review.")
        if vnotes:
            st.markdown(f"**Rejection Reason**: {vnotes}")
        st.markdown("If you believe this was in error, please contact system support.")

    elif status == "suspended":
        st.error("⛔ Your doctor account has been **SUSPENDED** by system administration.")
        if vnotes:
            st.markdown(f"**Suspension Reason**: {vnotes}")

    else:  # approved / active
        st.success("✅ **VERIFIED DOCTOR** — Your medical credentials have been approved!")
        if st.button("Go to Doctor Workspace"):
            st.session_state["page"] = "doctor_dashboard"
            st.rerun()
    _card_end()


def page_doctor_dashboard() -> None:
    if not is_doctor() or current_user_status() not in ("active", "approved"):
        st.error("Access Restricted: This page requires an approved Doctor account.")
        return

    doctor_id = current_user_id()
    doc_user = get_user_by_id(doctor_id) if doctor_id else None
    doc_name = f"Dr. {doc_user['username']}" if doc_user else "Doctor"

    # Check if doctor is currently scanning a specific patient
    scanning_patient_id = st.session_state.get("scanning_patient_id")
    if scanning_patient_id:
        allowed, auth_msg = can_doctor_scan_patient(doctor_id, scanning_patient_id)
        if not allowed:
            st.error(f"⛔ {auth_msg}")
            if st.button("⬅️ Return to Doctor Dashboard"):
                st.session_state.pop("scanning_patient_id", None)
                st.rerun()
            return

        target_patient = get_user_by_id(scanning_patient_id)
        patient_code = get_patient_code(scanning_patient_id)

        _hero(f"👨‍⚕️ {doc_name} — Scan Patient Workspace", f"Scanning Authorized Patient: {target_patient['username']} ({patient_code})")
        _card_start()
        st.info(f"📋 **Target Patient**: {target_patient['username']} | **Patient ID**: `{patient_code}` | **Email**: {target_patient['email']} | **Attending Doctor**: {doc_name}")

        if st.button("⬅️ Back to Patient List"):
            st.session_state.pop("scanning_patient_id", None)
            st.rerun()

        st.markdown("---")
        st.subheader("🔬 Upload Brain Scan & Run ML Detection")
        uploaded_file = st.file_uploader(
            f"Select MRI Image for Patient {target_patient['username']} ({patient_code})",
            type=["png", "jpg", "jpeg"],
            key=f"doc_scan_uploader_{scanning_patient_id}",
        )

        if uploaded_file is not None:
            if st.button("🚀 Analyze Scan & Save to Patient History", key="btn_run_doc_scan", type="primary"):
                with st.spinner("Executing AI Brain Scan Diagnostic Pipeline..."):
                    result = analyze_uploaded_scan(uploaded_file)

                # Save prediction record linked to patient_id and doctor_id
                pred_id = _save_scan_result(result, uploaded_file, patient_id=scanning_patient_id, doctor_id=doctor_id)
                log_audit_event(
                    user_id=doctor_id,
                    action="DOCTOR_PERFORMED_PATIENT_SCAN",
                    resource=f"patient:{scanning_patient_id}",
                    details={"prediction_id": pred_id, "patient_code": patient_code, "class": result["class"]},
                )
                st.session_state[f"last_doc_scan_res_{scanning_patient_id}"] = (result, pred_id)

        if f"last_doc_scan_res_{scanning_patient_id}" in st.session_state:
            res, pid = st.session_state[f"last_doc_scan_res_{scanning_patient_id}"]
            st.success(f"✅ Brain scan processed and saved to {target_patient['username']}'s medical records (Prediction ID: #{pid})")
            _render_result_visuals(res)

            st.markdown("---")
            st.subheader("✍️ Append Doctor Diagnostic Note & Recommendation")
            with st.form(f"doc_scan_note_form_{pid}"):
                note_text = st.text_area("Clinical Observation / Diagnostic Findings", placeholder="Enter diagnostic observations...")
                recom = st.selectbox(
                    "Clinical Recommendation",
                    [
                        "Confirm ML Diagnosis - Specialist Referral",
                        "Further Imaging Required",
                        "Routine Follow-up in 3 Months",
                        "Normal Scan - Regular Checkup",
                    ],
                )
                if st.form_submit_button("Save Clinical Diagnostic Note"):
                    if note_text.strip():
                        add_clinical_note(pid, doctor_id, note_text, recom)
                        st.success("Clinical note attached successfully!")
                        st.rerun()

            st.markdown("---")
            st.subheader("📄 Export Diagnostic PDF Report")
            from reports import generate_pdf_report
            pdf_bytes = generate_pdf_report(res, patient_id=scanning_patient_id)
            st.download_button(
                label=f"⬇️ Download PDF Report for {target_patient['username']}",
                data=pdf_bytes,
                file_name=f"NeuroScan_Report_{patient_code}_{pid}.pdf",
                mime="application/pdf",
                key=f"pdf_dl_doc_{pid}",
            )
        _card_end()
        return

    # Normal Doctor Dashboard Main View
    _hero(f"👨‍⚕️ {doc_name}'s Clinical Workspace", "Patient Management, Credentials & AI Diagnostics Console")
    assigned_patients = get_doctor_assigned_patients_with_details(doctor_id)
    assigned_predictions = get_doctor_assigned_predictions(doctor_id)

    col1, col2, col3 = st.columns(3)
    _metric_tile(col1, str(len(assigned_patients)), "My Patients", "#16b9ff")
    _metric_tile(col2, str(len(assigned_predictions)), "Clinical Scans", "#00d4a5")
    _metric_tile(col3, "VERIFIED DOCTOR", "Status", "#00d4a5")

    # + Add Patient Section
    _card_start()
    st.subheader("➕ Patient Management")
    ptab1, ptab2 = st.tabs(["🔍 Search Existing Patient", "➕ Register New Patient"])

    with ptab1:
        st.caption("Search an existing patient by Email, Phone, or Patient ID (PAT-XXXXXX) to establish association.")
        search_query = st.text_input("Patient Email / Phone / Patient Code", placeholder="e.g. john.doe@patient.com or PAT-000014", key="doc_search_pat_input")
        if st.button("Search Patient", key="btn_doc_search_pat"):
            if search_query.strip():
                pat_found = find_patient_by_identifier(search_query)
                if pat_found:
                    st.success(f"Found Patient: **{pat_found['username']}** (`{pat_found['patient_code']}`) — Email: {pat_found['email']}")
                    if st.button(f"➕ Establish Patient-Doctor Relationship with {pat_found['username']}", key=f"btn_assoc_{pat_found['id']}"):
                        assign_doctor_to_patient(pat_found['id'], doctor_id, status="active", created_by="doctor")
                        log_audit_event(doctor_id, "doctor_associated_existing_patient", resource=f"user:{pat_found['id']}")
                        st.success(f"Relationship established! Patient {pat_found['username']} is now added to My Patients.")
                        st.rerun()
                else:
                    st.warning("No patient account found matching that search query. Use 'Register New Patient' tab to add a new patient.")
            else:
                st.error("Please enter an email, phone, or Patient Code to search.")

    with ptab2:
        st.caption("Register a new patient account and establish an active doctor-patient relationship.")
        with st.form("doc_create_patient_form"):
            p_name = st.text_input("Full Name *")
            p_email = st.text_input("Email Address *")
            p_phone = st.text_input("Phone Number *", value="+91")
            p_dob = st.date_input("Date of Birth", value=datetime(1990, 1, 1))
            p_gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer Not to Say"])
            p_address = st.text_area("Address", placeholder="Patient address...")
            p_notes = st.text_area("Basic Medical History / Notes", placeholder="Relevant medical observations, history...")
            sub_pat = st.form_submit_button("Create Patient & Add to My Patients")

        if sub_pat:
            if not p_name or not p_email or not p_phone:
                st.error("Full Name, Email, and Phone Number are required.")
            else:
                dob_str = p_dob.strftime("%Y-%m-%d") if p_dob else None
                ok, msg, new_pid = create_doctor_added_patient(
                    doctor_id=doctor_id,
                    full_name=p_name,
                    email=p_email,
                    phone=p_phone,
                    dob=dob_str,
                    gender=p_gender,
                    address=p_address,
                    medical_notes=p_notes,
                )
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
    _card_end()

    # My Patients List Section
    _card_start()
    st.subheader("📋 My Patients List")
    if not assigned_patients:
        st.info("No patients currently associated with your account. Use '+ Add Patient' above to add or search patients.")
    else:
        for p in assigned_patients:
            with st.expander(f"👤 {p['full_name']} (`{p['patient_code']}`) — Email: {p['email']} | Phone: {p['phone']}"):
                col_info1, col_info2, col_actions = st.columns([2, 2, 2])
                with col_info1:
                    st.write(f"**Patient ID Code**: `{p['patient_code']}`")
                    st.write(f"**Full Name**: {p['full_name']}")
                    st.write(f"**Email**: {p['email']}")
                    st.write(f"**Phone**: {p['phone']}")
                with col_info2:
                    st.write(f"**Gender**: {p.get('gender') or 'N/A'}")
                    st.write(f"**Date of Birth**: {p.get('dob') or 'N/A'}")
                    st.write(f"**Relationship Status**: <span style='color:#00d4a5;font-weight:bold;'>{p['relationship_status'].upper()}</span>", unsafe_allow_html=True)
                    if p.get("last_prediction"):
                        lp = p["last_prediction"]
                        st.write(f"**Last Scan**: {lp['prediction']} ({lp['confidence']*100:.1f}%) on {lp['created_at']}")
                    else:
                        st.write("**Last Scan**: No scans on record yet")

                with col_actions:
                    st.markdown("#### Patient Actions")
                    if st.button(f"🔬 Scan Patient {p['full_name']}", key=f"scan_btn_{p['id']}", type="primary"):
                        st.session_state["scanning_patient_id"] = p['id']
                        st.rerun()

                st.markdown("---")
                st.write(f"**Medical History / Notes**: {p.get('medical_history') or 'None recorded'}")

                # Historical Scans and Reports for this patient
                p_preds = get_user_predictions(p['id'])
                if p_preds:
                    st.markdown("##### Historical Scan Reports")
                    for pred in p_preds:
                        st.caption(f"Scan #{pred['id']} - Classification: **{pred['prediction']}** ({pred['confidence']*100:.1f}%) | Date: {pred['created_at']}")
    _card_end()


def page_patient_doctor_consultation() -> None:
    if not is_patient() and not is_admin():
        st.error("Access Restricted: This page is for patient consultations.")
        return

    _hero("👨‍⚕️ Specialist Consultation", "Select an approved medical specialist to review your MRI scans and diagnostic reports.")
    patient_id = current_user_id()

    _card_start()
    st.subheader("Select & Assign a Specialist Doctor")
    approved_doctors = get_all_doctors(status="active")
    if not approved_doctors:
        approved_doctors = get_all_doctors(status="approved")

    if not approved_doctors:
        st.info("No verified doctors are currently available on the platform.")
    else:
        doc_options = {
            f"Dr. {d['username']} - {d.get('specialization', 'General')} ({d.get('hospital', 'Independent')})": d['id']
            for d in approved_doctors
        }
        selected_doc_label = st.selectbox("Choose Verified Doctor", list(doc_options.keys()))
        selected_doc_id = doc_options[selected_doc_label]

        if st.button("Assign Selected Doctor"):
            if assign_doctor_to_patient(patient_id, selected_doc_id):
                st.success("Doctor assigned! The selected specialist can now view your MRI scan records and provide clinical notes.")
                st.rerun()

    st.markdown("---")
    st.subheader("My Assigned Specialists")
    assigned_docs = get_assigned_doctors_for_patient(patient_id)
    if not assigned_docs:
        st.caption("You have not assigned any specialist doctors yet.")
    else:
        adoc_df = pd.DataFrame(assigned_docs)
        st.dataframe(adoc_df[["username", "specialization", "hospital", "email", "assigned_at"]], use_container_width=True)
    _card_end()

    # Show doctor clinical notes on patient scans
    user_preds = get_user_predictions(patient_id)
    if user_preds:
        _card_start()
        st.subheader("Clinical Feedback from Doctors")
        has_notes = False
        for p in user_preds:
            notes = get_clinical_notes_for_prediction(p["id"])
            if notes:
                has_notes = True
                st.markdown(f"#### Scan #{p['id']} - {p['prediction']} (Uploaded: {p['created_at']})")
                for n in notes:
                    st.success(f"**Dr. {n['doctor_name']}** ({n.get('specialization', '')} - {n.get('hospital', '')})\n\n**Recommendation**: {n.get('recommendation', 'N/A')}\n\n**Notes**: {n['note_content']}")
        if not has_notes:
            st.info("No clinical diagnostic notes have been attached to your scans yet.")
        _card_end()


def page_admin_dashboard() -> None:
    if not is_admin():
        st.error("Access Restricted: Administrator privileges required.")
        return

    _hero("⚙️ Admin Management Console", "Doctor verification, credential document inspection, and platform control.")
    admin_id = current_user_id()

    tab1, tab2, tab3 = st.tabs(["🩺 Doctor Verifications", "👥 User Management", "📊 System Metrics"])

    with tab1:
        _card_start()
        st.subheader("Doctor Qualification Verification Portal")
        pending_docs = get_all_doctors()
        pending_list = [d for d in pending_docs if d.get("status") in ("pending", "additional_info_required", "rejected", "suspended")]

        if not pending_list:
            st.success("No doctor verification applications pending review!")
        else:
            for doc in pending_list:
                status_color = "#fcbf49" if doc["status"] == "pending" else ("#ff6b6b" if doc["status"] in ("rejected", "suspended") else "#16b9ff")
                st.markdown(f"### Dr. {doc['username']} — Status: <span style='color:{status_color}; font-weight:bold;'>{doc['status'].upper()}</span>", unsafe_allow_html=True)

                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    st.write(f"**Email**: {doc['email']}")
                    st.write(f"**Phone**: {doc['phone']}")
                    st.write(f"**Specialization**: {doc.get('specialization', 'N/A')}")
                    st.write(f"**Qualification**: {doc.get('qualification', 'N/A')}")
                with dcol2:
                    st.write(f"**License No**: {doc.get('license_no', 'N/A')}")
                    st.write(f"**Hospital**: {doc.get('hospital', 'N/A')}")
                    st.write(f"**Experience**: {doc.get('years_experience', 'N/A')} years")
                    st.write(f"**Address**: {doc.get('address', 'N/A')}")

                docs = get_doctor_documents(doc["id"])
                st.markdown("#### 📎 Uploaded Medical Credentials")
                if not docs:
                    st.warning("No credential document records found for this doctor.")
                else:
                    doc_cols = st.columns(len(docs))
                    for idx, (col, dfile) in enumerate(zip(doc_cols, docs)):
                        with col:
                            st.caption(f"**{dfile['doc_type'].upper()}**: {dfile['original_filename']}")
                            fpath = Path(dfile['file_path'])
                            if fpath.exists():
                                file_data = fpath.read_bytes()
                                st.download_button(
                                    label=f"⬇️ Download {dfile['doc_type'].capitalize()}",
                                    data=file_data,
                                    file_name=dfile['original_filename'],
                                    mime=dfile['mime_type'],
                                    key=f"dl_{dfile['id']}",
                                )
                                if dfile['mime_type'].startswith("image/"):
                                    st.image(file_data, caption=dfile['original_filename'], use_column_width=True)
                            else:
                                st.error("File not found on disk.")

                vnotes = st.text_input(f"Verification / Administrative Notes for Dr. {doc['username']}", key=f"vnote_{doc['id']}")

                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    if st.button(f"✅ Approve Dr. {doc['username']}", key=f"app_{doc['id']}"):
                        update_doctor_verification(doc['id'], "approved", vnotes or "Approved by admin", admin_id)
                        update_user_status(doc['id'], "active", admin_id)
                        log_audit_event(admin_id, "verification_approved", resource=f"user:{doc['id']}")
                        st.success(f"Dr. {doc['username']} approved and verified!")
                        st.rerun()
                with b2:
                    if st.button(f"❓ Request Additional Info", key=f"req_{doc['id']}"):
                        if not vnotes.strip():
                            st.error("Please enter admin notes explaining what additional info is required.")
                        else:
                            update_doctor_verification(doc['id'], "additional_info_required", vnotes, admin_id)
                            log_audit_event(admin_id, "additional_info_requested", resource=f"user:{doc['id']}", details={"notes": vnotes})
                            st.info(f"Requested additional info from Dr. {doc['username']}.")
                            st.rerun()
                with b3:
                    if st.button(f"❌ Reject Application", key=f"rej_{doc['id']}"):
                        update_doctor_verification(doc['id'], "rejected", vnotes or "Rejected by admin", admin_id)
                        log_audit_event(admin_id, "verification_rejected", resource=f"user:{doc['id']}")
                        st.error(f"Dr. {doc['username']} application rejected.")
                        st.rerun()
                with b4:
                    if st.button(f"⛔ Suspend Account", key=f"susp_{doc['id']}"):
                        update_doctor_verification(doc['id'], "suspended", vnotes or "Suspended by admin", admin_id)
                        log_audit_event(admin_id, "account_suspended", resource=f"user:{doc['id']}")
                        st.warning(f"Dr. {doc['username']} account suspended.")
                        st.rerun()
                st.markdown("---")
        _card_end()

    with tab2:
        _card_start()
        st.subheader("All Registered Users")
        users = get_all_users()
        if users:
            udf = pd.DataFrame(users)
            st.dataframe(udf[["id", "username", "email", "role", "status", "specialization", "created_at"]], use_container_width=True)

            st.markdown("---")
            st.subheader("Modify Account Status")
            user_options = {f"{u['username']} (ID: {u['id']} - Role: {u['role']} - Status: {u['status']})": u['id'] for u in users}
            selected_user_label = st.selectbox("Select User", list(user_options.keys()))
            selected_user_id = user_options[selected_user_label]
            new_status = st.selectbox("Set New Status", ["active", "inactive", "pending", "rejected"])

            if st.button("Update User Status"):
                update_user_status(selected_user_id, new_status, admin_id)
                st.success(f"Status for user ID {selected_user_id} updated to {new_status}.")
                st.rerun()
        _card_end()

    with tab3:
        _card_start()
        st.subheader("Platform Aggregate Metrics")
        users = get_all_users()
        total_users = len(users)
        total_docs = len([u for u in users if u["role"] == "doctor"])
        total_patients = len([u for u in users if u["role"] == "patient"])

        mcol1, mcol2, mcol3 = st.columns(3)
        _metric_tile(mcol1, str(total_users), "Total Registered Users", "#16b9ff")
        _metric_tile(mcol2, str(total_docs), "Medical Doctors", "#00d4a5")
        _metric_tile(mcol3, str(total_patients), "Patients", "#fcbf49")
        _card_end()


def page_audit_logs() -> None:
    if not is_admin():
        st.error("Access Restricted: Administrator privileges required.")
        return

    _hero("📜 System Audit Logs", "Immutable audit trail of authentication, authorization, doctor approvals, and record modifications.")
    _card_start()
    logs = get_audit_logs(limit=150)
    if not logs:
        st.info("No audit logs recorded yet.")
    else:
        log_df = pd.DataFrame(logs)
        st.dataframe(
            log_df[["id", "timestamp", "user_id", "username", "role", "action", "resource", "details_json"]],
            use_container_width=True,
        )
    _card_end()


def page_mastishka() -> None:
    _hero("🧠 Mastishka Assistant", "Ask basic questions about NeuroScan AI, tumour types, or your latest scan")

    st.session_state.setdefault(
        "mastishka_chat",
        [
            {
                "role": "assistant",
                "content": (
                    "Hi, I'm **Mastishka** 🧠 — your NeuroScan AI assistant. Ask me about "
                    "tumour types, how the app works, or your latest scan result. I run entirely "
                    "offline (no external AI API), so I stick to basic, pre-defined answers."
                ),
            }
        ],
    )

    _card_start()
    st.caption("Quick questions")
    quick_cols = st.columns(len(SUGGESTED_QUESTIONS))
    quick_pick = None
    for col, question in zip(quick_cols, SUGGESTED_QUESTIONS):
        if col.button(question, use_container_width=True, key=f"quick_{question}"):
            quick_pick = question
    _card_end()

    _card_start()
    for turn in st.session_state["mastishka_chat"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
    _card_end()

    # st.chat_input always docks to the bottom of the page regardless of
    # surrounding containers, so it's kept outside the card above.
    user_message = st.chat_input("Ask Mastishka a question...")
    incoming = quick_pick or user_message

    if incoming:
        st.session_state["mastishka_chat"].append({"role": "user", "content": incoming})
        reply = mastishka_response(incoming, scan_result=st.session_state.get("last_scan_result"))
        st.session_state["mastishka_chat"].append({"role": "assistant", "content": reply})
        st.rerun()

    if st.button("Clear conversation"):
        st.session_state["mastishka_chat"] = []
        st.rerun()


def page_profile() -> None:
    if not is_logged_in():
        st.error("Access Restricted: Please log in to view your profile.")
        return

    user_id = current_user_id()
    user = get_user_by_id(user_id) if user_id else None
    if not user:
        st.error("User record not found.")
        return

    role = user.get("role", "patient").capitalize()
    status = user.get("status", "active").upper()
    _hero(f"👤 {user['username']}'s Profile", f"Account Overview, Credential Verification & Role Settings ({role})")

    col1, col2, col3, col4 = st.columns(4)
    _metric_tile(col1, user["username"], "Username", "#16b9ff")
    _metric_tile(col2, role, "Account Role", "#00d4a5")
    _metric_tile(col3, status, "Verification Status", "#fcbf49" if status == "PENDING" else ("#00d4a5" if status in ("APPROVED", "ACTIVE") else "#ff6b6b"))
    _metric_tile(col4, "Verified" if user.get("is_verified") else "Unverified", "Phone OTP Status", "#00d4a5" if user.get("is_verified") else "#ff6b6b")

    _card_start()
    st.subheader("Account Identity & System Info")
    d1, d2 = st.columns(2)
    with d1:
        st.write(f"**User ID**: #{user['id']:05d}")
        st.write(f"**Email Address**: {user['email']}")
        st.write(f"**Phone Number**: {user['phone']}")
    with d2:
        st.write(f"**Account Registered**: {user.get('created_at', 'N/A')}")
        st.write(f"**Last Login**: {user.get('last_login', 'N/A')}")
        st.write(f"**System Access Level**: {role} Privileges")
    _card_end()

    if user.get("role") == "doctor":
        _card_start()
        st.subheader("🩺 Medical Doctor Credentials & Professional Profile")
        p1, p2 = st.columns(2)
        with p1:
            st.write(f"**Medical Specialization**: {user.get('specialization', 'N/A')}")
            st.write(f"**Qualification Degree**: {user.get('qualification', 'N/A')}")
            st.write(f"**License / Registration No**: {user.get('license_no', 'N/A')}")
        with p2:
            st.write(f"**Hospital / Clinic Affiliation**: {user.get('hospital', 'N/A')}")
            st.write(f"**Years of Experience**: {user.get('years_experience', 'N/A')} years")
            st.write(f"**Clinic Address**: {user.get('address', 'N/A')}")

        st.markdown("---")
        st.subheader("📎 Uploaded Verification Certificates")
        docs = get_doctor_documents(user_id)
        if not docs:
            st.warning("No verification document records found.")
        else:
            doc_cols = st.columns(len(docs))
            for idx, (col, dfile) in enumerate(zip(doc_cols, docs)):
                with col:
                    st.caption(f"**{dfile['doc_type'].upper()}**: {dfile['original_filename']}")
                    fpath = Path(dfile['file_path'])
                    if fpath.exists():
                        file_data = fpath.read_bytes()
                        st.download_button(
                            label=f"⬇️ Download {dfile['doc_type'].capitalize()}",
                            data=file_data,
                            file_name=dfile['original_filename'],
                            mime=dfile['mime_type'],
                            key=f"prof_dl_{dfile['id']}",
                        )
                        if dfile['mime_type'].startswith("image/"):
                            st.image(file_data, caption=dfile['original_filename'], use_column_width=True)
                    else:
                        st.error("File not found on disk.")
        _card_end()

    elif user.get("role") == "patient":
        _card_start()
        st.subheader("🏥 Patient Clinical Summary")
        user_preds = get_user_predictions(user_id)
        assigned_docs = get_assigned_doctors_for_patient(user_id)
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.write(f"**Total MRI Scans Uploaded**: {len(user_preds)}")
        with pcol2:
            st.write(f"**Assigned Specialist Doctors**: {len(assigned_docs)}")
        _card_end()


def main() -> None:
    render_sidebar()
    page = st.session_state.get("page", "home")
    routes = {
        "home": page_home,
        "register": page_register,
        "login": page_login,
        "upload": page_upload,
        "dashboard": page_dashboard,
        "history": page_history,
        "training": page_training,
        "mastishka": page_mastishka,
        "about": page_about,
        "profile": page_profile,
        "doctor_dashboard": page_doctor_dashboard,
        "doctor_pending": page_doctor_pending,
        "consultation": page_patient_doctor_consultation,
        "admin_dashboard": page_admin_dashboard,
        "audit_logs": page_audit_logs,
    }
    routes.get(page, page_home)()


if __name__ == "__main__":
    main()

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
    is_logged_in,
    login_user,
    register_user,
    send_otp,
    set_session,
)
from backend import analyze_uploaded_scan
from database import (
    create_patient_record,
    get_dashboard_stats,
    get_user_predictions,
    initialize_database,
    save_prediction,
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
            st.success(f"Signed in as {current_username()}")
            pages = {
                "Home": "home",
                "Upload & Scan": "upload",
                "Dashboard": "dashboard",
                "Scan History": "history",
                "Training Hub": "training",
                "About": "about",
            }
        else:
            pages = {
                "Home": "home",
                "Login": "login",
                "Register": "register",
                "About": "about",
            }

        st.session_state.setdefault("page", "home")
        for label, page_key in pages.items():
            if st.button(label, use_container_width=True, type="primary" if st.session_state["page"] == page_key else "secondary"):
                st.session_state["page"] = page_key
                st.rerun()

        if is_logged_in() and st.button("Logout", use_container_width=True):
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
    _hero("Create Account", "Secure access with OTP verification")
    _card_start()
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        phone = st.text_input("Phone", value="+91")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account")

    if submitted:
        success, message = register_user(username, email, phone, password, confirm)
        if success:
            st.success(message)
            otp = send_otp(phone)
            st.session_state["pending_phone"] = phone
            st.session_state["otp_demo"] = otp
        else:
            st.error(message)

    if st.session_state.get("pending_phone"):
        st.info(f"Demo OTP: {st.session_state.get('otp_demo')}")
        otp_value = st.text_input("Enter OTP", key="otp_input")
        if st.button("Verify OTP"):
            if confirm_otp(st.session_state["pending_phone"], otp_value):
                st.success("Phone number verified. You can now log in.")
                st.session_state.pop("pending_phone", None)
            else:
                st.error("Invalid or expired OTP.")
    _card_end()


def page_login() -> None:
    _hero("Sign In", "Access your MRI analysis workspace")
    _card_start()
    with st.form("login_form"):
        identifier = st.text_input("Username / Email / Phone")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        success, message, user = login_user(identifier, password)
        if success and user:
            set_session(user)
            st.session_state["page"] = "upload"
            st.success(message)
            st.rerun()
        else:
            st.error(message)
    _card_end()


def _save_scan_result(result: dict, uploaded_file, patient_id: int | None) -> int:
    segmentation = result.get("segmentation") or {}
    return save_prediction(
        user_id=current_user_id(),
        patient_id=patient_id,
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
        "about": page_about,
    }
    routes.get(page, page_home)()


if __name__ == "__main__":
    main()

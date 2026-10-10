"""
authentication/auth.py
======================
Authentication utilities for Brain Tumour Detection System.
Handles password hashing, session management, and OTP generation.
"""

import random
import string
import re
import logging
import uuid
import mimetypes
from pathlib import Path
import streamlit as st

import bcrypt

from database import (
    create_user, create_user_with_role, get_user_by_identifier, verify_user,
    update_last_login, check_duplicate, store_otp, verify_otp, log_audit_event,
    save_doctor_document, replace_doctor_document, get_doctor_documents,
    get_user_by_id, is_patient_assigned_to_doctor,
)

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
SESSION_KEY        = "user_id"
SESSION_USERNAME   = "username"
SESSION_ROLE       = "user_role"
SESSION_STATUS     = "user_status"
SESSION_VERIFIED   = "otp_verified"
OTP_LENGTH         = 6
MIN_PASSWORD_LEN   = 8
PHONE_REGEX        = re.compile(r"^\+?[1-9]\d{9,14}$")
EMAIL_REGEX        = re.compile(r"^[\w\.\+\-]+@[\w\-]+\.[a-z]{2,}$", re.I)

ALLOWED_DOC_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
MAX_DOC_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
DOC_UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads" / "doctor_documents"


# ─── Document Upload Security Helpers ─────────────────────────────────────────
def validate_document_file(file_obj, max_size_mb: int = 10) -> tuple[bool, str]:
    if file_obj is None:
        return False, "No file provided."
    filename = getattr(file_obj, "name", "")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_DOC_EXTENSIONS:
        return False, f"Invalid file extension '{ext}'. Allowed formats: PDF, PNG, JPG, JPEG."

    file_bytes = getattr(file_obj, "size", None)
    if file_bytes is None:
        content = file_obj.getvalue() if hasattr(file_obj, "getvalue") else file_obj.read()
        file_bytes = len(content)

    if file_bytes > max_size_mb * 1024 * 1024:
        return False, f"File size exceeds maximum limit of {max_size_mb} MB."

    return True, ""


def save_uploaded_doctor_document(
    file_obj, user_id: int, doc_type: str, is_replacement: bool = False
) -> tuple[bool, str, dict | None]:
    ok, err = validate_document_file(file_obj)
    if not ok:
        return False, err, None

    filename = getattr(file_obj, "name", "document")
    ext = Path(filename).suffix.lower()
    content = file_obj.getvalue() if hasattr(file_obj, "getvalue") else file_obj.read()

    if ext in (".png", ".jpg", ".jpeg"):
        from preprocessing.image_processing import reencode_uploaded_image
        try:
            fmt = "PNG" if ext == ".png" else "JPEG"
            content = reencode_uploaded_image(content, output_format=fmt)
        except Exception as exc:
            logger.warning("Failed to re-encode image document: %s", exc)

    file_size = len(content)
    mime_type, _ = mimetypes.guess_type(filename)
    mime_type = mime_type or "application/octet-stream"

    user_dir = DOC_UPLOAD_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)

    stored_filename = f"{doc_type}_{uuid.uuid4().hex}{ext}"
    file_path = str(user_dir / stored_filename)

    with open(file_path, "wb") as f:
        f.write(content)

    if is_replacement:
        doc_id = replace_doctor_document(
            user_id=user_id,
            doc_type=doc_type,
            original_filename=filename,
            stored_filename=stored_filename,
            file_path=file_path,
            file_size=file_size,
            mime_type=mime_type,
        )
        log_audit_event(
            user_id=user_id,
            action="document_replaced",
            resource=f"document:{doc_id}",
            details={"doc_type": doc_type, "original_filename": filename, "file_size": file_size},
        )
    else:
        doc_id = save_doctor_document(
            user_id=user_id,
            doc_type=doc_type,
            original_filename=filename,
            stored_filename=stored_filename,
            file_path=file_path,
            file_size=file_size,
            mime_type=mime_type,
        )
        log_audit_event(
            user_id=user_id,
            action="document_uploaded",
            resource=f"document:{doc_id}",
            details={"doc_type": doc_type, "original_filename": filename, "file_size": file_size},
        )

    doc_info = {
        "id": doc_id,
        "user_id": user_id,
        "doc_type": doc_type,
        "original_filename": filename,
        "stored_filename": stored_filename,
        "file_path": file_path,
        "file_size": file_size,
        "mime_type": mime_type,
    }
    return True, "Document uploaded successfully.", doc_info


# ─── Password Utilities ───────────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    """Return a bcrypt hash of the plain-text password."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Check plain password against stored bcrypt hash."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ─── OTP Utilities ────────────────────────────────────────────────────────────
def generate_otp() -> str:
    """Generate a numeric OTP of OTP_LENGTH digits."""
    return "".join(random.choices(string.digits, k=OTP_LENGTH))


def send_otp(phone: str) -> str:
    """Generate OTP, persist hashed record, and dispatch via configured OTP provider."""
    from authentication.otp_provider import get_otp_provider

    otp = generate_otp()
    ok, store_msg = store_otp(phone, otp, ttl_minutes=5)
    if not ok:
        logger.error("Failed to store OTP for %s: %s", phone, store_msg)
        return ""

    provider = get_otp_provider()
    sent_ok, send_msg = provider.send_otp(phone, otp)
    if not sent_ok:
        logger.error("Failed to send OTP to %s: %s", phone, send_msg)
        return ""

    return otp


def confirm_otp(phone: str, entered: str) -> bool:
    """Verify entered OTP against stored hash and mark user as verified on success."""
    from database.db import verify_otp_detailed

    ok, msg = verify_otp_detailed(phone, entered.strip())
    if ok:
        verify_user(phone)
        return True
    logger.warning("OTP verification failed for %s: %s", phone, msg)
    return False


# ─── Validation Helpers ───────────────────────────────────────────────────────
def validate_email(email: str) -> bool:
    return bool(EMAIL_REGEX.match(email.strip()))


def validate_phone(phone: str) -> bool:
    return bool(PHONE_REGEX.match(phone.strip()))


def validate_password(password: str) -> tuple[bool, str]:
    """
    Returns (True, '') if valid, else (False, reason).
    Rules: min 8 chars, at least one digit, one uppercase, one special char.
    """
    if len(password) < MIN_PASSWORD_LEN:
        return False, f"Password must be at least {MIN_PASSWORD_LEN} characters."
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter."
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit."
    if not any(c in "!@#$%^&*()-_=+[]{}|;:',.<>?/`~" for c in password):
        return False, "Password must contain at least one special character."
    return True, ""


# ─── Registration Flow ────────────────────────────────────────────────────────
def register_user(
    username: str,
    email: str,
    phone: str,
    password: str,
    confirm: str,
    role: str = "patient",
    specialization: str | None = None,
    qualification: str | None = None,
    license_no: str | None = None,
    hospital: str | None = None,
    years_experience: int | None = None,
    address: str | None = None,
) -> tuple[bool, str, int | None]:
    """
    Full registration validation + DB insert with role support.
    Returns (success: bool, message: str, user_id: int | None).
    """
    if not all([username, email, phone, password, confirm]):
        return False, "All fields marked as required must be filled.", None

    username = username.strip()
    email    = email.strip()
    phone    = phone.strip()

    if len(username) < 3:
        return False, "Username must be at least 3 characters.", None
    if not validate_email(email):
        return False, "Invalid email address.", None
    if not validate_phone(phone):
        return False, "Invalid phone number (include country code, e.g. +91XXXXXXXXXX).", None

    if role == "doctor":
        if not all([specialization, qualification, license_no, hospital]):
            return False, "Specialization, Qualification Degree, License Number, and Hospital Affiliation are required for doctor registration.", None

    ok, msg = validate_password(password)
    if not ok:
        return False, msg, None
    if password != confirm:
        return False, "Passwords do not match.", None

    dup = check_duplicate(username, email, phone)
    if dup:
        return False, f"That {dup} is already registered. Please use a different one.", None

    hashed = hash_password(password)
    status = "pending" if role == "doctor" else "active"

    user_id = create_user_with_role(
        username=username,
        email=email,
        phone=phone,
        hashed_password=hashed,
        role=role,
        status=status,
        specialization=specialization,
        qualification=qualification,
        license_no=license_no,
        hospital=hospital,
        years_experience=years_experience,
        address=address,
    )
    if user_id:
        log_audit_event(
            user_id=user_id,
            action="doctor_application_submitted" if role == "doctor" else "user_registered",
            resource=f"user:{user_id}",
            details={"role": role, "status": status},
        )
        if role == "doctor":
            return True, "Doctor application submitted! Please verify mandatory credentials below.", user_id
        return True, "Account created! Please verify your phone number via OTP.", user_id
    return False, "Registration failed. Please try again.", None


# ─── Login Flow ───────────────────────────────────────────────────────────────
def login_user(identifier: str, password: str) -> tuple[bool, str, dict | None]:
    """
    Authenticate a user by username/email/phone + password.
    Returns (success, message, user_dict | None).
    """
    if not identifier or not password:
        return False, "Please enter your credentials.", None

    user = get_user_by_identifier(identifier.strip())
    if not user:
        return False, "No account found with those credentials.", None

    if not verify_password(password, user["password"]):
        log_audit_event(
            user_id=user["id"],
            action="login_failed",
            resource="auth",
            details={"reason": "incorrect_password"},
        )
        return False, "Incorrect password.", None

    user_status = user.get("status", "active")
    if user_status in ("inactive", "deactivated"):
        log_audit_event(
            user_id=user["id"],
            action="login_blocked",
            resource="auth",
            details={"reason": "account_deactivated"},
        )
        return False, "Account deactivated. Please contact administrator.", None

    if not user["is_verified"]:
        return False, "Account phone number not verified. Please complete OTP verification.", None

    update_last_login(user["id"])
    log_audit_event(
        user_id=user["id"],
        action="user_login",
        resource="auth",
        details={"role": user.get("role", "patient"), "status": user_status},
    )
    return True, f"Welcome back, {user['username']}!", user


# ─── Streamlit Session Helpers ────────────────────────────────────────────────
def set_session(user: dict):
    """Persist authenticated user info in Streamlit session state."""
    st.session_state[SESSION_KEY]      = user["id"]
    st.session_state[SESSION_USERNAME] = user["username"]
    st.session_state[SESSION_ROLE]     = user.get("role", "patient")
    st.session_state[SESSION_STATUS]   = user.get("status", "active")
    st.session_state["user_email"]     = user["email"]
    st.session_state["user_phone"]     = user["phone"]
    st.session_state["logged_in"]      = True


def clear_session():
    """Wipe all session state data on logout to prevent residual state leaks."""
    st.session_state.clear()


def is_logged_in() -> bool:
    return st.session_state.get("logged_in", False)


def current_user_id() -> int | None:
    return st.session_state.get(SESSION_KEY)


def current_username() -> str:
    return st.session_state.get(SESSION_USERNAME, "Guest")


def current_user_role() -> str:
    return st.session_state.get(SESSION_ROLE, "guest")


def current_user_status() -> str:
    return st.session_state.get(SESSION_STATUS, "active")


def is_admin() -> bool:
    return is_logged_in() and current_user_role() == "admin"


def is_doctor() -> bool:
    return is_logged_in() and current_user_role() == "doctor"


def is_patient() -> bool:
    return is_logged_in() and current_user_role() == "patient"


def can_doctor_scan_patient(doctor_id: int, patient_id: int) -> tuple[bool, str]:
    """
    Backend Authorization Check:
    Verifies that:
    1. Doctor ID exists, role is 'doctor', and status is 'active' or 'approved'.
    2. Patient ID exists and role is 'patient'.
    3. Active relationship exists in patient_doctor_assignments.
    """
    doc = get_user_by_id(doctor_id)
    if not doc or doc.get("role") != "doctor":
        return False, "Authorization Denied: Invalid doctor user account."
    if doc.get("status") not in ("active", "approved"):
        return False, "Authorization Denied: Doctor account is not active or approved."

    patient = get_user_by_id(patient_id)
    if not patient or patient.get("role") != "patient":
        return False, "Authorization Denied: Target patient record not found."

    from database import is_patient_assigned_to_doctor
    if not is_patient_assigned_to_doctor(patient_id, doctor_id):
        return False, f"Authorization Denied: Doctor Dr. {doc['username']} does not have an active authorized relationship with Patient {patient['username']}."

    return True, "Authorized"


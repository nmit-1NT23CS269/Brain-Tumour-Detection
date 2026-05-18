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
import streamlit as st

import bcrypt

from database import (
    create_user, get_user_by_identifier, verify_user,
    update_last_login, check_duplicate, store_otp, verify_otp,
)

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
SESSION_KEY        = "user_id"
SESSION_USERNAME   = "username"
SESSION_VERIFIED   = "otp_verified"
OTP_LENGTH         = 6
MIN_PASSWORD_LEN   = 8
PHONE_REGEX        = re.compile(r"^\+?[1-9]\d{9,14}$")
EMAIL_REGEX        = re.compile(r"^[\w\.\+\-]+@[\w\-]+\.[a-z]{2,}$", re.I)


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
    """
    Generate OTP, persist it, and (in production) send via SMS gateway.
    For demo purposes the OTP is returned and displayed on-screen.
    """
    otp = generate_otp()
    store_otp(phone, otp, ttl_minutes=10)
    logger.info(f"OTP for {phone}: {otp}  [DEMO — replace with SMS API]")
    return otp


def confirm_otp(phone: str, entered: str) -> bool:
    """Verify entered OTP and mark user as verified on success."""
    if verify_otp(phone, entered.strip()):
        verify_user(phone)
        return True
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
def register_user(username: str, email: str, phone: str,
                  password: str, confirm: str) -> tuple[bool, str]:
    """
    Full registration validation + DB insert.
    Returns (success: bool, message: str).
    """
    # Basic blank checks
    if not all([username, email, phone, password, confirm]):
        return False, "All fields are required."

    username = username.strip()
    email    = email.strip()
    phone    = phone.strip()

    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not validate_email(email):
        return False, "Invalid email address."
    if not validate_phone(phone):
        return False, "Invalid phone number (include country code, e.g. +91XXXXXXXXXX)."

    ok, msg = validate_password(password)
    if not ok:
        return False, msg
    if password != confirm:
        return False, "Passwords do not match."

    # Duplicate check
    dup = check_duplicate(username, email, phone)
    if dup:
        return False, f"That {dup} is already registered. Please use a different one."

    hashed = hash_password(password)
    success = create_user(username, email, phone, hashed)
    if success:
        return True, "Account created! Please verify your phone number."
    return False, "Registration failed. Please try again."


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
        return False, "Incorrect password.", None

    if not user["is_verified"]:
        return False, "Account not verified. Please complete OTP verification.", None

    update_last_login(user["id"])
    return True, f"Welcome back, {user['username']}!", user


# ─── Streamlit Session Helpers ────────────────────────────────────────────────
def set_session(user: dict):
    """Persist authenticated user info in Streamlit session state."""
    st.session_state[SESSION_KEY]      = user["id"]
    st.session_state[SESSION_USERNAME] = user["username"]
    st.session_state["user_email"]     = user["email"]
    st.session_state["user_phone"]     = user["phone"]
    st.session_state["logged_in"]      = True


def clear_session():
    """Wipe authentication data from session state."""
    for key in [SESSION_KEY, SESSION_USERNAME, "user_email", "user_phone",
                "logged_in", SESSION_VERIFIED]:
        st.session_state.pop(key, None)


def is_logged_in() -> bool:
    return st.session_state.get("logged_in", False)


def current_user_id() -> int | None:
    return st.session_state.get(SESSION_KEY)


def current_username() -> str:
    return st.session_state.get(SESSION_USERNAME, "Guest")

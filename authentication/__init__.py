from .auth import (
    hash_password, verify_password, generate_otp, send_otp, confirm_otp,
    validate_email, validate_phone, validate_password,
    register_user, login_user,
    set_session, clear_session, is_logged_in, current_user_id, current_username,
    current_user_role, current_user_status, is_admin, is_doctor, is_patient,
    validate_document_file, save_uploaded_doctor_document,
    can_doctor_scan_patient,
)

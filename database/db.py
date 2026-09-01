"""SQLite database handler for NeuroScan AI."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from utils import configure_logging


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "brain_tumour.db")
LOGGER = configure_logging(__name__)


@contextmanager
def get_connection():
    """Yield a transactional SQLite connection."""
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception as exc:
        conn.rollback()
        LOGGER.error("Database error: %s", exc)
        raise
    finally:
        conn.close()


def _ensure_column(conn, table_name: str, column_name: str, column_type: str) -> None:
    if not re.match(r"^[A-Za-z0-9_]+$", table_name):
        raise ValueError(f"Invalid table_name identifier: {table_name}")
    if not re.match(r"^[A-Za-z0-9_]+$", column_name):
        raise ValueError(f"Invalid column_name identifier: {column_name}")
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")



def _create_demo_doctor_files(conn, user_id: int, degree_name: str, license_name: str) -> None:
    upload_dir = Path(__file__).resolve().parent.parent / "uploads" / "doctor_documents" / str(user_id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    deg_path = upload_dir / degree_name
    lic_path = upload_dir / license_name

    pdf_content = b"%PDF-1.4 Mock Medical Verification Certificate Content for NeuroScan AI Demo Evaluation"
    deg_path.write_bytes(pdf_content)
    lic_path.write_bytes(pdf_content)

    conn.execute(
        """
        INSERT INTO doctor_documents (user_id, doc_type, original_filename, stored_filename, file_path, file_size, mime_type, status)
        VALUES (?, 'degree', ?, ?, ?, ?, 'application/pdf', 'uploaded')
        """,
        (user_id, degree_name, degree_name, str(deg_path), len(pdf_content)),
    )
    conn.execute(
        """
        INSERT INTO doctor_documents (user_id, doc_type, original_filename, stored_filename, file_path, file_size, mime_type, status)
        VALUES (?, 'license', ?, ?, ?, ?, 'application/pdf', 'uploaded')
        """,
        (user_id, license_name, license_name, str(lic_path), len(pdf_content)),
    )


def _seed_sample_prediction(conn, patient_id: int, doctor_id: int) -> None:
    dummy_img_bytes = b"MOCK_MRI_IMAGE_BYTES"
    conn.execute(
        """
        INSERT INTO predictions (
            user_id, patient_id, doctor_id, image_filename, image_data, prediction, confidence,
            glioma_prob, meningioma_prob, pituitary_prob, normal_prob, severity,
            confidence_label, quality_label, focus_region, ai_summary, assistant_text,
            model_name, model_version, segmentation_coverage
        ) VALUES (?, ?, ?, 'demo_mri_scan.jpg', ?, 'Glioma', 0.9680, 0.9680, 0.0210, 0.0070, 0.0040,
                  'High Risk', 'High Confidence (96.8%)', 'Acceptable Quality', 'Right Temporal Lobe',
                  'AI Brain Scan Detection indicates a high-confidence Glioma lesion (96.8%) concentrated in the right temporal lobe region.',
                  'Recommend immediate specialist neuro-oncology referral and enhanced contrast-enhanced MRI evaluation.',
                  'VGG16 Transfer Model', 'v1.0-demo', 0.142)
        """,
        (patient_id, patient_id, doctor_id, dummy_img_bytes),
    )


def initialize_database() -> None:
    """Initialize and migrate SQLite schema."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                phone TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                is_verified INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS otp_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone TEXT NOT NULL,
                otp_code TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                is_used INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                patient_name TEXT,
                patient_age INTEGER,
                patient_gender TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                model_path TEXT,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                patient_id INTEGER,
                image_filename TEXT NOT NULL,
                image_data BLOB,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                glioma_prob REAL,
                meningioma_prob REAL,
                pituitary_prob REAL,
                normal_prob REAL,
                severity TEXT,
                confidence_label TEXT,
                quality_label TEXT,
                focus_region TEXT,
                ai_summary TEXT,
                assistant_text TEXT,
                model_name TEXT,
                model_version TEXT,
                segmentation_coverage REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (patient_id) REFERENCES patient_records(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analytics_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_doctor_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                doctor_id INTEGER NOT NULL,
                status TEXT DEFAULT 'active',
                assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES users(id),
                FOREIGN KEY (doctor_id) REFERENCES users(id),
                UNIQUE(patient_id, doctor_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clinical_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                doctor_id INTEGER NOT NULL,
                note_content TEXT NOT NULL,
                recommendation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id),
                FOREIGN KEY (doctor_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                resource TEXT,
                details_json TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS doctor_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                doc_type TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                mime_type TEXT NOT NULL,
                status TEXT DEFAULT 'uploaded',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )

        # Migrate users table columns
        for column_name, column_type in [
            ("role", "TEXT DEFAULT 'patient'"),
            ("status", "TEXT DEFAULT 'active'"),
            ("specialization", "TEXT"),
            ("qualification", "TEXT"),
            ("license_no", "TEXT"),
            ("hospital", "TEXT"),
            ("years_experience", "INTEGER"),
            ("address", "TEXT"),
            ("verification_notes", "TEXT"),
        ]:
            _ensure_column(conn, "users", column_name, column_type)

        # Migrate predictions table columns
        for column_name, column_type in [
            ("patient_id", "INTEGER"),
            ("doctor_id", "INTEGER"),
            ("severity", "TEXT"),
            ("confidence_label", "TEXT"),
            ("quality_label", "TEXT"),
            ("focus_region", "TEXT"),
            ("ai_summary", "TEXT"),
            ("assistant_text", "TEXT"),
            ("model_name", "TEXT"),
            ("model_version", "TEXT"),
            ("segmentation_coverage", "REAL"),
        ]:
            _ensure_column(conn, "predictions", column_name, column_type)

        # Migrate patient_records table columns
        for column_name, column_type in [
            ("dob", "TEXT"),
            ("address", "TEXT"),
            ("medical_history", "TEXT"),
            ("created_by_doctor_id", "INTEGER"),
        ]:
            _ensure_column(conn, "patient_records", column_name, column_type)

        # Migrate patient_doctor_assignments table columns
        for column_name, column_type in [
            ("created_by", "TEXT DEFAULT 'patient'"),
        ]:
            _ensure_column(conn, "patient_doctor_assignments", column_name, column_type)

        # Seed default admin user if not present
        admin_row = conn.execute("SELECT id FROM users WHERE role = 'admin' OR username = 'admin'").fetchone()
        if not admin_row:
            conn.execute(
                """
                INSERT INTO users (username, email, phone, password, role, status, is_verified)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    "admin",
                    "admin@neuroscan.ai",
                    "+10000000000",
                    "$2b$12$8gO8Zs9e0r6Fp3jcMCRn1uSWKzpwTE4OM2a50B6KzyIGtfYZWkNR2",
                    "admin",
                    "active",
                ),
            )

        # Seed Demo Admin (admin.demo@example.com / Admin@123)
        demo_admin_row = conn.execute("SELECT id FROM users WHERE email = 'admin.demo@example.com'").fetchone()
        if not demo_admin_row:
            conn.execute(
                """
                INSERT INTO users (username, email, phone, password, role, status, is_verified)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    "demo_admin",
                    "admin.demo@example.com",
                    "+18005550199",
                    "$2b$12$su/N8JDKHuoIc26luN19LuFvytkuOhQYC2roiszalGKbVqXcLKPbK",
                    "admin",
                    "active",
                ),
            )

        # Seed Verified Demo Doctor (doctor.demo@example.com / Doctor@123)
        demo_doc_row = conn.execute("SELECT id FROM users WHERE email = 'doctor.demo@example.com'").fetchone()
        if not demo_doc_row:
            cursor = conn.execute(
                """
                INSERT INTO users (
                    username, email, phone, password, role, status, specialization,
                    qualification, license_no, hospital, years_experience, address, is_verified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    "dr_demo",
                    "doctor.demo@example.com",
                    "+18005550100",
                    "$2b$12$ue9pQKAaB28OXJ5DR3yEbuyfv9Y1G7525MhiUdnMIzH1LNqcl0W7O",
                    "doctor",
                    "approved",
                    "Neuro-oncology & Radiology",
                    "MBBS, MD, DM Neuro-oncology",
                    "NMC-DEMO-9948",
                    "Metropolitan Neuro-Institute",
                    15,
                    "100 Medical Center Way, Suite 400",
                ),
            )
            demo_doc_id = cursor.lastrowid
            _create_demo_doctor_files(conn, demo_doc_id, "degree_demo.pdf", "license_demo.pdf")
        else:
            demo_doc_id = demo_doc_row["id"]

        # Seed Pending Demo Doctor (pending.doctor@example.com / Doctor@123)
        pending_doc_row = conn.execute("SELECT id FROM users WHERE email = 'pending.doctor@example.com'").fetchone()
        if not pending_doc_row:
            cursor = conn.execute(
                """
                INSERT INTO users (
                    username, email, phone, password, role, status, specialization,
                    qualification, license_no, hospital, years_experience, address, is_verified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    "dr_pending_demo",
                    "pending.doctor@example.com",
                    "+18005550101",
                    "$2b$12$ue9pQKAaB28OXJ5DR3yEbuyfv9Y1G7525MhiUdnMIzH1LNqcl0W7O",
                    "doctor",
                    "pending",
                    "Pediatric Neurology",
                    "MBBS, MD Pediatric Neurology",
                    "NMC-PENDING-7712",
                    "St. Jude Children's Medical Center",
                    7,
                    "45 Children's Health Park",
                ),
            )
            pending_doc_id = cursor.lastrowid
            _create_demo_doctor_files(conn, pending_doc_id, "degree_pending_demo.pdf", "license_pending_demo.pdf")

        # Seed Demo Patient (patient.demo@example.com / Patient@123)
        demo_pat_row = conn.execute("SELECT id FROM users WHERE email = 'patient.demo@example.com'").fetchone()
        if not demo_pat_row:
            cursor = conn.execute(
                """
                INSERT INTO users (username, email, phone, password, role, status, is_verified)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    "demo_patient",
                    "patient.demo@example.com",
                    "+18005550155",
                    "$2b$12$DyD7PF0tmdParSpYNtpdvOtwGAVPbTGlXUTF4VZI99wHkEYcqsUna",
                    "patient",
                    "active",
                ),
            )
            demo_pat_id = cursor.lastrowid
            conn.execute(
                """
                INSERT INTO patient_records (user_id, patient_name, patient_gender, dob, address, notes)
                VALUES (?, 'Demo Patient', 'Male', '1988-04-12', '742 Evergreen Terrace', 'Patient history of recurring temporal headaches.')
                """,
                (demo_pat_id,),
            )
            conn.execute(
                """
                INSERT INTO patient_doctor_assignments (patient_id, doctor_id, status, created_by)
                VALUES (?, ?, 'active', 'doctor')
                ON CONFLICT(patient_id, doctor_id) DO UPDATE SET status = 'active'
                """,
                (demo_pat_id, demo_doc_id),
            )
            # Create sample MRI prediction record for demo patient
            _seed_sample_prediction(conn, demo_pat_id, demo_doc_id)

        # Seed legacy default approved doctor & patient if missing
        doc_row = conn.execute("SELECT id FROM users WHERE username = 'dr_smith'").fetchone()
        if not doc_row:
            conn.execute(
                """
                INSERT INTO users (
                    username, email, phone, password, role, status, specialization,
                    qualification, license_no, hospital, years_experience, address, is_verified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    "dr_smith",
                    "dr.smith@neuroscan.ai",
                    "+919876543210",
                    "$2b$12$QzjxWSbZyk5n8dWTxtHZqugPROXRzKSeGbpNanJTbUgAg86tgNMPS",
                    "doctor",
                    "approved",
                    "Neuro-oncology",
                    "MBBS, MD, DM Neuro-oncology",
                    "NMC-884920",
                    "City Neurological Institute",
                    12,
                    "42 Medical Plaza, Sector 5",
                ),
            )

        pat_row = conn.execute("SELECT id FROM users WHERE username = 'john_doe'").fetchone()
        if not pat_row:
            conn.execute(
                """
                INSERT INTO users (username, email, phone, password, role, status, is_verified)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    "john_doe",
                    "john.doe@patient.com",
                    "+919123456789",
                    "$2b$12$7kMw/MvZ7imFvmCQ2jvEm.YCSNDZMTahhqmu7rT4rxRAR./XCeuJ2",
                    "patient",
                    "active",
                ),
            )


def create_user(username: str, email: str, phone: str, hashed_password: str) -> bool:
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO users (username, email, phone, password) VALUES (?, ?, ?, ?)",
                (username.strip(), email.strip().lower(), phone.strip(), hashed_password),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def get_user_by_identifier(identifier: str):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ? OR phone = ?",
            (identifier, identifier.lower(), identifier),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def verify_user(phone: str) -> bool:
    with get_connection() as conn:
        conn.execute("UPDATE users SET is_verified = 1 WHERE phone = ?", (phone,))
    return True


def update_last_login(user_id: int) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (datetime.now(), user_id))


def check_duplicate(username: str, email: str, phone: str) -> str | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT username, email, phone FROM users WHERE username=? OR email=? OR phone=?",
            (username, email.lower(), phone),
        ).fetchone()
    if not row:
        return None
    if row["username"] == username:
        return "username"
    if row["email"] == email.lower():
        return "email"
    if row["phone"] == phone:
        return "phone"
    return None


def store_otp(phone: str, otp_code: str, ttl_minutes: int = 10) -> bool:
    expires_at = datetime.now() + timedelta(minutes=ttl_minutes)
    with get_connection() as conn:
        conn.execute("UPDATE otp_records SET is_used = 1 WHERE phone = ? AND is_used = 0", (phone,))
        conn.execute(
            "INSERT INTO otp_records (phone, otp_code, expires_at) VALUES (?, ?, ?)",
            (phone, otp_code, expires_at),
        )
    return True


def verify_otp(phone: str, otp_code: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT * FROM otp_records
            WHERE phone = ? AND otp_code = ? AND is_used = 0
            ORDER BY created_at DESC LIMIT 1
            """,
            (phone, otp_code),
        ).fetchone()
        if not row:
            return False
        if datetime.now() > datetime.fromisoformat(str(row["expires_at"])):
            return False
        conn.execute("UPDATE otp_records SET is_used = 1 WHERE id = ?", (row["id"],))
    return True


def create_patient_record(
    user_id: int,
    patient_name: str,
    patient_age: int | None = None,
    patient_gender: str | None = None,
    notes: str | None = None,
) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO patient_records (user_id, patient_name, patient_age, patient_gender, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, patient_name, patient_age, patient_gender, notes),
        )
        return int(cursor.lastrowid)


def register_model_version(
    model_name: str,
    model_version: str,
    model_path: str,
    accuracy: float | None = None,
    precision_score: float | None = None,
    recall_score: float | None = None,
    f1_score: float | None = None,
) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO model_versions (
                model_name, model_version, model_path, accuracy,
                precision_score, recall_score, f1_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (model_name, model_version, model_path, accuracy, precision_score, recall_score, f1_score),
        )
        return int(cursor.lastrowid)


def save_prediction(
    user_id: int,
    image_filename: str,
    image_data: bytes,
    prediction: str,
    confidence: float,
    class_probs: dict,
    patient_id: int | None = None,
    doctor_id: int | None = None,
    severity: str | None = None,
    confidence_label: str | None = None,
    quality_label: str | None = None,
    focus_region: str | None = None,
    ai_summary: str | None = None,
    assistant_text: str | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    segmentation_coverage: float | None = None,
) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (
                user_id, patient_id, doctor_id, image_filename, image_data, prediction, confidence,
                glioma_prob, meningioma_prob, pituitary_prob, normal_prob, severity,
                confidence_label, quality_label, focus_region, ai_summary, assistant_text,
                model_name, model_version, segmentation_coverage
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                patient_id,
                doctor_id,
                image_filename,
                image_data,
                prediction,
                round(confidence, 4),
                round(class_probs.get("Glioma", 0), 4),
                round(class_probs.get("Meningioma", 0), 4),
                round(class_probs.get("Pituitary", 0), 4),
                round(class_probs.get("Normal", 0), 4),
                severity,
                confidence_label,
                quality_label,
                focus_region,
                ai_summary,
                assistant_text,
                model_name,
                model_version,
                segmentation_coverage,
            ),
        )
        prediction_id = int(cursor.lastrowid)

    log_prediction_event(prediction_id, "prediction_saved", {"prediction": prediction, "confidence": confidence})
    return prediction_id


def log_prediction_event(prediction_id: int, event_type: str, payload: dict | None = None) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO analytics_events (prediction_id, event_type, payload) VALUES (?, ?, ?)",
            (prediction_id, event_type, json.dumps(payload or {})),
        )


def get_user_predictions(user_id: int, limit: int = 50) -> list:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, image_filename, prediction, confidence, severity, quality_label,
                   model_name, created_at
            FROM predictions WHERE user_id = ?
            ORDER BY created_at DESC LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def get_prediction_by_id(pred_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,)).fetchone()
    return dict(row) if row else None


def get_dashboard_stats(user_id: int) -> dict:
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) AS cnt FROM predictions WHERE user_id = ?", (user_id,)).fetchone()["cnt"]
        by_class_rows = conn.execute(
            "SELECT prediction, COUNT(*) AS cnt FROM predictions WHERE user_id = ? GROUP BY prediction",
            (user_id,),
        ).fetchall()
        recent = conn.execute(
            """
            SELECT image_filename, prediction, confidence, severity, quality_label, model_name, created_at
            FROM predictions WHERE user_id = ?
            ORDER BY created_at DESC LIMIT 5
            """,
            (user_id,),
        ).fetchall()
        quality_rows = conn.execute(
            "SELECT quality_label, COUNT(*) AS cnt FROM predictions WHERE user_id = ? GROUP BY quality_label",
            (user_id,),
        ).fetchall()
        models = conn.execute(
            "SELECT model_name, COUNT(*) AS cnt FROM predictions WHERE user_id = ? GROUP BY model_name",
            (user_id,),
        ).fetchall()

    return {
        "total": total,
        "by_class": {row["prediction"]: row["cnt"] for row in by_class_rows},
        "recent": [dict(row) for row in recent],
        "quality_breakdown": {row["quality_label"] or "Unknown": row["cnt"] for row in quality_rows},
        "models_used": {row["model_name"] or "Unknown": row["cnt"] for row in models},
    }


def create_user_with_role(
    username: str,
    email: str,
    phone: str,
    hashed_password: str,
    role: str = "patient",
    status: str = "active",
    specialization: str | None = None,
    qualification: str | None = None,
    license_no: str | None = None,
    hospital: str | None = None,
    years_experience: int | None = None,
    address: str | None = None,
) -> int | None:
    try:
        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (
                    username, email, phone, password, role, status,
                    specialization, qualification, license_no, hospital,
                    years_experience, address
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    username.strip(),
                    email.strip().lower(),
                    phone.strip(),
                    hashed_password,
                    role,
                    status,
                    specialization,
                    qualification,
                    license_no,
                    hospital,
                    years_experience,
                    address,
                ),
            )
            return int(cursor.lastrowid)
    except sqlite3.IntegrityError:
        return None


def get_pending_doctors() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, username, email, phone, specialization, license_no, hospital,
                   status, created_at
            FROM users WHERE role = 'doctor' AND status = 'pending'
            ORDER BY created_at ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_all_doctors(status: str | None = None) -> list[dict]:
    with get_connection() as conn:
        if status:
            rows = conn.execute(
                """
                SELECT id, username, email, phone, specialization, license_no, hospital, status
                FROM users WHERE role = 'doctor' AND status = ?
                ORDER BY username ASC
                """,
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, username, email, phone, specialization, license_no, hospital, status
                FROM users WHERE role = 'doctor'
                ORDER BY username ASC
                """
            ).fetchall()
    return [dict(row) for row in rows]


def update_doctor_verification(
    doctor_id: int, status: str, notes: str | None = None, admin_id: int | None = None
) -> bool:
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET status = ?, verification_notes = ? WHERE id = ? AND role = 'doctor'",
            (status, notes, doctor_id),
        )
    log_audit_event(
        user_id=admin_id,
        action="doctor_verification_update",
        resource=f"user:{doctor_id}",
        details={"new_status": status, "verification_notes": notes},
    )
    return True


def get_all_users() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, username, email, phone, role, status, specialization, license_no,
                   hospital, is_verified, created_at, last_login
            FROM users ORDER BY created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def update_user_status(user_id: int, status: str, admin_id: int | None = None) -> bool:
    with get_connection() as conn:
        conn.execute("UPDATE users SET status = ? WHERE id = ?", (status, user_id))
    log_audit_event(
        user_id=admin_id,
        action="user_status_update",
        resource=f"user:{user_id}",
        details={"new_status": status},
    )
    return True


def assign_doctor_to_patient(patient_id: int, doctor_id: int) -> bool:
    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO patient_doctor_assignments (patient_id, doctor_id, status)
                VALUES (?, ?, 'active')
                ON CONFLICT(patient_id, doctor_id) DO UPDATE SET status = 'active'
                """,
                (patient_id, doctor_id),
            )
        log_audit_event(
            user_id=patient_id,
            action="patient_doctor_assignment",
            resource=f"doctor:{doctor_id}",
            details={"patient_id": patient_id, "doctor_id": doctor_id},
        )
        return True
    except sqlite3.Error:
        return False


def get_assigned_doctors_for_patient(patient_id: int) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT u.id, u.username, u.email, u.phone, u.specialization, u.hospital, a.assigned_at
            FROM patient_doctor_assignments a
            JOIN users u ON a.doctor_id = u.id
            WHERE a.patient_id = ? AND a.status = 'active' AND u.status = 'active'
            """,
            (patient_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_assigned_patients_for_doctor(doctor_id: int) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT u.id, u.username, u.email, u.phone, a.assigned_at
            FROM patient_doctor_assignments a
            JOIN users u ON a.patient_id = u.id
            WHERE a.doctor_id = ? AND a.status = 'active' AND u.status = 'active'
            """,
            (doctor_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def is_patient_assigned_to_doctor(patient_id: int, doctor_id: int) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id FROM patient_doctor_assignments
            WHERE patient_id = ? AND doctor_id = ? AND status = 'active'
            """,
            (patient_id, doctor_id),
        ).fetchone()
    return row is not None


def add_clinical_note(
    prediction_id: int, doctor_id: int, note_content: str, recommendation: str | None = None
) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO clinical_notes (prediction_id, doctor_id, note_content, recommendation)
            VALUES (?, ?, ?, ?)
            """,
            (prediction_id, doctor_id, note_content.strip(), recommendation.strip() if recommendation else None),
        )
        note_id = int(cursor.lastrowid)

    log_audit_event(
        user_id=doctor_id,
        action="clinical_note_added",
        resource=f"prediction:{prediction_id}",
        details={"note_id": note_id, "recommendation": recommendation},
    )
    return note_id


def get_clinical_notes_for_prediction(prediction_id: int) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.note_content, c.recommendation, c.created_at,
                   u.username AS doctor_name, u.specialization, u.hospital
            FROM clinical_notes c
            JOIN users u ON c.doctor_id = u.id
            WHERE c.prediction_id = ?
            ORDER BY c.created_at DESC
            """,
            (prediction_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_doctor_assigned_predictions(doctor_id: int) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.user_id, p.image_filename, p.prediction, p.confidence,
                   p.severity, p.quality_label, p.model_name, p.created_at,
                   u.username AS patient_username, u.email AS patient_email
            FROM predictions p
            JOIN patient_doctor_assignments a ON p.user_id = a.patient_id
            JOIN users u ON p.user_id = u.id
            WHERE a.doctor_id = ? AND a.status = 'active'
            ORDER BY p.created_at DESC
            """,
            (doctor_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def log_audit_event(
    user_id: int | None, action: str, resource: str | None = None, details: dict | None = None
) -> None:
    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs (user_id, action, resource, details_json)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, action, resource, json.dumps(details or {})),
            )
    except Exception as exc:
        LOGGER.error("Failed to write audit log: %s", exc)


def get_audit_logs(limit: int = 100) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT a.id, a.user_id, a.action, a.resource, a.details_json, a.timestamp,
                   u.username, u.role
            FROM audit_logs a
            LEFT JOIN users u ON a.user_id = u.id
            ORDER BY a.timestamp DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def save_doctor_document(
    user_id: int,
    doc_type: str,
    original_filename: str,
    stored_filename: str,
    file_path: str,
    file_size: int,
    mime_type: str,
) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO doctor_documents (
                user_id, doc_type, original_filename, stored_filename,
                file_path, file_size, mime_type, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'uploaded')
            """,
            (user_id, doc_type, original_filename, stored_filename, file_path, file_size, mime_type),
        )
        return int(cursor.lastrowid)


def replace_doctor_document(
    user_id: int,
    doc_type: str,
    original_filename: str,
    stored_filename: str,
    file_path: str,
    file_size: int,
    mime_type: str,
) -> int:
    with get_connection() as conn:
        conn.execute(
            "UPDATE doctor_documents SET status = 'replaced' WHERE user_id = ? AND doc_type = ? AND status = 'uploaded'",
            (user_id, doc_type),
        )
        cursor = conn.execute(
            """
            INSERT INTO doctor_documents (
                user_id, doc_type, original_filename, stored_filename,
                file_path, file_size, mime_type, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'uploaded')
            """,
            (user_id, doc_type, original_filename, stored_filename, file_path, file_size, mime_type),
        )
        return int(cursor.lastrowid)


def get_doctor_documents(user_id: int, include_replaced: bool = False) -> list[dict]:
    with get_connection() as conn:
        if include_replaced:
            rows = conn.execute(
                "SELECT * FROM doctor_documents WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM doctor_documents WHERE user_id = ? AND status = 'uploaded' ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
    return [dict(row) for row in rows]


def get_doctor_document_by_id(doc_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM doctor_documents WHERE id = ?", (doc_id,)).fetchone()
    return dict(row) if row else None


def get_patient_code(user_id: int) -> str:
    """Format user ID as PAT-XXXXXX patient code."""
    return f"PAT-{user_id:06d}"


def parse_patient_code(patient_code: str) -> int | None:
    """Extract integer ID from PAT-XXXXXX code."""
    cleaned = patient_code.strip().upper()
    if cleaned.startswith("PAT-"):
        try:
            return int(cleaned.replace("PAT-", ""))
        except ValueError:
            return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def find_patient_by_identifier(identifier: str) -> dict | None:
    """Search patient user account by Email, Phone, or Patient Code (PAT-XXXXXX)."""
    cleaned = identifier.strip()
    parsed_id = parse_patient_code(cleaned)
    with get_connection() as conn:
        if parsed_id is not None:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ? AND role = 'patient'",
                (parsed_id,),
            ).fetchone()
            if row:
                res = dict(row)
                res["patient_code"] = get_patient_code(res["id"])
                return res

        row = conn.execute(
            """
            SELECT * FROM users
            WHERE (email = ? OR phone = ? OR username = ?) AND role = 'patient'
            """,
            (cleaned.lower(), cleaned, cleaned),
        ).fetchone()

    if row:
        res = dict(row)
        res["patient_code"] = get_patient_code(res["id"])
        return res
    return None


def create_doctor_added_patient(
    doctor_id: int,
    full_name: str,
    email: str,
    phone: str,
    dob: str | None = None,
    gender: str | None = None,
    address: str | None = None,
    medical_notes: str | None = None,
    hashed_password: str | None = None,
) -> tuple[bool, str, int | None]:
    """Create a new patient user account, patient record, and active doctor assignment."""
    dup = check_duplicate(full_name, email, phone)
    if dup:
        return False, f"A user with that {dup} already exists.", None

    import secrets
    fallback_hash = hashed_password or secrets.token_hex(16)

    patient_id = create_user_with_role(
        username=full_name.strip(),
        email=email.strip().lower(),
        phone=phone.strip(),
        hashed_password=fallback_hash,
        role="patient",
        status="active",
        address=address,
    )

    if not patient_id:
        return False, "Failed to create patient user account.", None

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO patient_records (
                user_id, patient_name, patient_gender, dob, address, notes, created_by_doctor_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (patient_id, full_name.strip(), gender, dob, address, medical_notes, doctor_id),
        )
        conn.execute(
            """
            INSERT INTO patient_doctor_assignments (patient_id, doctor_id, status, created_by)
            VALUES (?, ?, 'active', 'doctor')
            ON CONFLICT(patient_id, doctor_id) DO UPDATE SET status = 'active'
            """,
            (patient_id, doctor_id),
        )

    log_audit_event(
        user_id=doctor_id,
        action="doctor_added_patient",
        resource=f"user:{patient_id}",
        details={"patient_code": get_patient_code(patient_id), "patient_name": full_name},
    )
    return True, f"Patient {full_name} created successfully! Patient Code: {get_patient_code(patient_id)}", patient_id


def get_doctor_assigned_patients_with_details(doctor_id: int) -> list[dict]:
    """Return all active patients assigned to a doctor with patient codes, records, and last scan metrics."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT u.id, u.username AS full_name, u.email, u.phone, u.created_at,
                   r.dob, r.patient_gender AS gender, r.address, r.notes AS medical_history,
                   a.assigned_at, a.status AS relationship_status
            FROM patient_doctor_assignments a
            JOIN users u ON a.patient_id = u.id
            LEFT JOIN patient_records r ON u.id = r.user_id
            WHERE a.doctor_id = ? AND a.status = 'active' AND u.status = 'active'
            ORDER BY u.username ASC
            """,
            (doctor_id,),
        ).fetchall()

    result = []
    for r in rows:
        pdict = dict(r)
        pdict["patient_code"] = get_patient_code(pdict["id"])

        with get_connection() as conn:
            last_pred = conn.execute(
                """
                SELECT id, prediction, confidence, created_at
                FROM predictions WHERE user_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (pdict["id"],),
            ).fetchone()

        pdict["last_prediction"] = dict(last_pred) if last_pred else None
        result.append(pdict)

    return result

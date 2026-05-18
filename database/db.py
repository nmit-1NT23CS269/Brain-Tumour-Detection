"""SQLite database handler for NeuroScan AI."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

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
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


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

        for column_name, column_type in [
            ("patient_id", "INTEGER"),
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
                user_id, patient_id, image_filename, image_data, prediction, confidence,
                glioma_prob, meningioma_prob, pituitary_prob, normal_prob, severity,
                confidence_label, quality_label, focus_region, ai_summary, assistant_text,
                model_name, model_version, segmentation_coverage
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                patient_id,
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

import os
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import _save_scan_result
from database.db import create_patient_record, get_prediction_by_id


class DummyFile:
    def __init__(self, name: str, bytes_data: bytes):
        self.name = name
        self._bytes = bytes_data

    def read(self):
        return self._bytes

    def seek(self, pos):
        pass


def test_save_scan_result_with_patient_record():
    dummy_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
    dummy_file = DummyFile("test_mri_pytest.png", dummy_bytes)

    # 1. Create patient record (user_id=24: demo_patient)
    patient_rec_id = create_patient_record(
        user_id=24,
        patient_name="Pytest Patient",
        patient_age=40,
        patient_gender="Female",
        notes="Pytest scan test",
    )
    assert patient_rec_id > 0

    result = {
        "class": "Normal",
        "confidence": 0.98,
        "probabilities": {"Glioma": 0.01, "Meningioma": 0.005, "Pituitary": 0.005, "Normal": 0.98},
        "severity": "Low Risk",
        "confidence_text": "High Confidence (98.0%)",
        "quality": {"label": "Optimal Quality"},
        "focus_region": "None",
        "ai_summary": "Pytest summary",
        "assistant_text": "Pytest recommendation",
        "model_name": "VGG16 Transfer Model",
        "model_version": "v1.0",
        "steps": {"file_bytes": dummy_bytes},
        "segmentation": {"coverage_ratio": 0.0},
    }

    # 2. Save prediction passing patient_id (patient_records.id) and target_user_id (users.id)
    pred_id = _save_scan_result(
        result=result,
        uploaded_file=dummy_file,
        patient_id=patient_rec_id,
        target_user_id=24,
    )
    assert pred_id > 0

    # 3. Retrieve and assert
    saved = get_prediction_by_id(pred_id)
    assert saved is not None
    assert saved["user_id"] == 24
    assert saved["patient_id"] == patient_rec_id

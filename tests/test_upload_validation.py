"""Unit tests for upload validation, EXIF metadata stripping, and image re-encoding."""

from __future__ import annotations

import io
from PIL import Image

from preprocessing.image_processing import reencode_uploaded_image, validate_image, validate_mri_image_quality


class DummyUploadedFile:
    def __init__(self, name: str, mime_type: str, content: bytes):
        self.name = name
        self.type = mime_type
        self._content = content
        self.size = len(content)

    def read(self):
        return self._content

    def seek(self, pos):
        pass


def test_validate_image_format_and_size():
    # Valid file
    valid_file = DummyUploadedFile("test.png", "image/png", b"x" * 1024)
    ok, msg = validate_image(valid_file)
    assert ok is True

    # Invalid extension/MIME
    invalid_file = DummyUploadedFile("malicious.exe", "application/x-msdownload", b"x" * 1024)
    ok, msg = validate_image(invalid_file)
    assert ok is False
    assert "Unsupported format" in msg

    # Oversized file (> 10MB)
    huge_file = DummyUploadedFile("huge.jpg", "image/jpeg", b"x" * (11 * 1024 * 1024))
    ok, msg = validate_image(huge_file)
    assert ok is False
    assert "too large" in msg.lower()


def test_exif_stripping_and_reencoding():
    # Create a synthetic image with EXIF metadata
    img = Image.new("RGB", (100, 100), color="blue")
    exif_data = img.getexif()
    exif_data[0x0110] = "Test Camera Model EXIF Payload"  # Model tag

    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif_data)
    dirty_bytes = buf.getvalue()

    # Re-encode image bytes through sanitization pipeline
    clean_bytes = reencode_uploaded_image(dirty_bytes, output_format="JPEG")
    assert len(clean_bytes) > 0

    # Verify that EXIF tag is stripped in the re-encoded output image
    with Image.open(io.BytesIO(clean_bytes)) as clean_img:
        clean_exif = clean_img.getexif()
        assert 0x0110 not in clean_exif

"""Unit tests for OTP security, rate limiting, hashing, and attempt caps."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

from authentication.otp_provider import DemoProvider, EmailProvider, TwilioProvider, get_otp_provider
from database.db import get_connection, store_otp, verify_otp, verify_otp_detailed


@pytest.fixture(autouse=True)
def clean_test_otps():
    """Clean test OTP records before each test."""
    test_phone = "+19998887777"
    with get_connection() as conn:
        conn.execute("DELETE FROM otp_records WHERE phone = ?", (test_phone,))
    yield
    with get_connection() as conn:
        conn.execute("DELETE FROM otp_records WHERE phone = ?", (test_phone,))


def test_otp_hashing_and_verification():
    phone = "+19998887777"
    code = "123456"

    success, msg = store_otp(phone, code, ttl_minutes=5)
    assert success is True

    # Verify that stored OTP code in DB is a SHA-256 hex digest (64 chars), not plain text
    with get_connection() as conn:
        row = conn.execute(
            "SELECT otp_code FROM otp_records WHERE phone = ? ORDER BY created_at DESC LIMIT 1",
            (phone,),
        ).fetchone()
        assert row is not None
        assert row["otp_code"] != code
        assert len(row["otp_code"]) == 64

    # Verify correct code succeeds
    ver_success, ver_msg = verify_otp_detailed(phone, code)
    assert ver_success is True
    assert "successfully" in ver_msg.lower()


def test_otp_attempt_cap():
    phone = "+19998887777"
    code = "654321"

    store_otp(phone, code, ttl_minutes=5)

    # Attempt 4 wrong codes
    for _ in range(4):
        ok, msg = verify_otp_detailed(phone, "000000")
        assert ok is False
        assert "attempt(s) remaining" in msg

    # 5th wrong attempt should lock out the OTP
    ok, msg = verify_otp_detailed(phone, "000000")
    assert ok is False
    assert "exceeded" in msg.lower() or "maximum" in msg.lower()

    # Even right code now fails because OTP is locked
    ok_right, _ = verify_otp_detailed(phone, code)
    assert ok_right is False


def test_otp_rate_limiting():
    phone = "+19998887777"

    # Send 3 OTPs successfully
    for i in range(3):
        ok, msg = store_otp(phone, f"10000{i}", ttl_minutes=5)
        assert ok is True

    # 4th request within 1 hour should be rate-limited
    ok, msg = store_otp(phone, "999999", ttl_minutes=5)
    assert ok is False
    assert "rate limit" in msg.lower()


def test_pluggable_otp_providers():
    os.environ["APP_ENV"] = "development"
    os.environ["OTP_PROVIDER"] = "demo"
    provider = get_otp_provider()
    assert isinstance(provider, DemoProvider)

    os.environ["OTP_PROVIDER"] = "twilio"
    provider = get_otp_provider()
    assert isinstance(provider, TwilioProvider)

    os.environ["OTP_PROVIDER"] = "email"
    provider = get_otp_provider()
    assert isinstance(provider, EmailProvider)

    # Reset default env
    os.environ["OTP_PROVIDER"] = "demo"

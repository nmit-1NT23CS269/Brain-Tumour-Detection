"""Pluggable OTP Delivery Provider Interface and Implementations.

Supports Demo (Development), Twilio (SMS), and SMTP (Email) providers.
"""

from __future__ import annotations

import abc
import logging
import os
import smtplib
from email.mime.text import MIMEText
from typing import Optional

from utils import configure_logging

LOGGER = configure_logging(__name__)


class OTPProvider(abc.ABC):
    """Abstract base class for pluggable OTP delivery providers."""

    @abc.abstractmethod
    def send_otp(self, recipient: str, otp_code: str) -> tuple[bool, str]:
        """Send OTP code to recipient (phone number or email address).

        Returns:
            tuple[bool, str]: (Success status, user-facing message/details)
        """
        pass


class DemoProvider(OTPProvider):
    """Demo OTP provider for development/testing environments."""

    def send_otp(self, recipient: str, otp_code: str) -> tuple[bool, str]:
        app_env = os.getenv("APP_ENV", "development").lower()
        if app_env == "production":
            LOGGER.error("DemoProvider invoked in production environment! Rejecting OTP delivery.")
            return False, "Demo OTP provider is disabled in production environment."

        LOGGER.info("[DEMO OTP] Recipient: %s | Code: %s", recipient, otp_code)
        return True, f"Demo OTP generated and logged for {recipient}."


class TwilioProvider(OTPProvider):
    """Twilio SMS OTP delivery provider."""

    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.from_number = os.getenv("TWILIO_PHONE_NUMBER", "")

    def send_otp(self, recipient: str, otp_code: str) -> tuple[bool, str]:
        if not (self.account_sid and self.auth_token and self.from_number):
            LOGGER.error("Twilio credentials missing in environment variables.")
            return False, "SMS service configuration error."

        try:
            from twilio.rest import Client

            client = Client(self.account_sid, self.auth_token)
            message = client.messages.create(
                body=f"Your NeuroScan AI verification code is: {otp_code}. Valid for 5 minutes.",
                from_=self.from_number,
                to=recipient,
            )
            LOGGER.info("Twilio SMS sent successfully. Message SID: %s", message.sid)
            return True, "OTP SMS sent successfully."
        except Exception as exc:
            LOGGER.error("Twilio SMS delivery failed: %s", exc)
            return False, f"SMS delivery failed: {exc}"


class EmailProvider(OTPProvider):
    """SMTP Email OTP delivery provider."""

    def __init__(self):
        self.host = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.port = int(os.getenv("SMTP_PORT", "587"))
        self.username = os.getenv("SMTP_USERNAME", "")
        self.password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("SMTP_FROM_EMAIL", self.username)

    def send_otp(self, recipient: str, otp_code: str) -> tuple[bool, str]:
        if not (self.username and self.password):
            LOGGER.error("SMTP credentials missing in environment variables.")
            return False, "Email service configuration error."

        try:
            msg = MIMEText(
                f"Your NeuroScan AI security verification code is: {otp_code}\n\n"
                f"This code will expire in 5 minutes. If you did not request this, please ignore."
            )
            msg["Subject"] = "NeuroScan AI - Verification Code"
            msg["From"] = self.from_email
            msg["To"] = recipient

            with smtplib.SMTP(self.host, self.port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_email, [recipient], msg.as_string())

            LOGGER.info("SMTP OTP email sent successfully to %s", recipient)
            return True, "OTP email sent successfully."
        except Exception as exc:
            LOGGER.error("SMTP OTP email delivery failed: %s", exc)
            return False, f"Email delivery failed: {exc}"


def get_otp_provider() -> OTPProvider:
    """Factory function resolving configured OTP provider based on environment variables."""
    app_env = os.getenv("APP_ENV", "development").lower()
    provider_name = os.getenv("OTP_PROVIDER", "demo" if app_env != "production" else "twilio").lower()

    if provider_name == "twilio":
        return TwilioProvider()
    elif provider_name == "email" or provider_name == "smtp":
        return EmailProvider()
    elif provider_name == "demo":
        if app_env == "production":
            LOGGER.warning("Demo OTP provider configured in production! Falling back to TwilioProvider.")
            return TwilioProvider()
        return DemoProvider()
    else:
        LOGGER.warning("Unknown OTP_PROVIDER '%s'. Defaulting to DemoProvider.", provider_name)
        return DemoProvider()

"""Provider-agnostic LLM client for Mastishka with PII sanitization and streaming.

Supports OpenAI, Anthropic, Gemini, and Ollama via .env settings. Uses REST/native API
calls lazily with 20s timeout, 1 retry, PII stripping, and offline fallback mode.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Generator

logger = logging.getLogger(__name__)

# PII Regex patterns for stripping sensitive patient info before LLM transmission
PII_PATTERNS = [
    (re.compile(r"\bPAT-\d+\b", re.IGNORECASE), "[PATIENT_ID_REDACTED]"),
    (re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"), "[EMAIL_REDACTED]"),
    (re.compile(r"(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|(?:\+\d{1,3}[-.\s]?)?\d{3}[-.\s]?\d{4}\b"), "[PHONE_REDACTED]"),
]


def sanitize_pii(text: str) -> str:
    """Strip patient codes, emails, and phone numbers from prompt text."""
    if not text:
        return ""
    sanitized = text
    for pattern, replacement in PII_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


class LLMClient:
    """Provider-agnostic LLM client supporting OpenAI, Gemini, Anthropic, and Ollama."""

    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        self.api_key = os.getenv("LLM_API_KEY", "").strip()
        self.model = os.getenv("LLM_MODEL", "").strip()
        self.timeout = int(os.getenv("LLM_TIMEOUT", "20"))

        # Default model selection per provider if not specified
        if not self.model:
            if self.provider == "openai":
                self.model = "gpt-3.5-turbo"
            elif self.provider == "gemini":
                self.model = "gemini-1.5-flash"
            elif self.provider == "anthropic":
                self.model = "claude-3-haiku-20240307"
            elif self.provider == "ollama":
                self.model = "llama3"

    def is_available(self) -> bool:
        """Return True if an LLM provider is configured."""
        if self.provider in {"openai", "gemini", "anthropic"}:
            return bool(self.api_key)
        if self.provider == "ollama":
            return True
        return False

    def generate(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        """Generate response text from configured provider with 1 retry on timeout/error."""
        if not self.is_available():
            raise RuntimeError("LLM client not configured or API key missing.")

        clean_system = sanitize_pii(system_prompt)
        clean_messages = [
            {"role": m["role"], "content": sanitize_pii(m["content"])}
            for m in messages
        ]

        last_err = None
        for attempt in range(2):
            try:
                if self.provider == "openai":
                    return self._call_openai(clean_system, clean_messages)
                elif self.provider == "gemini":
                    return self._call_gemini(clean_system, clean_messages)
                elif self.provider == "anthropic":
                    return self._call_anthropic(clean_system, clean_messages)
                elif self.provider == "ollama":
                    return self._call_ollama(clean_system, clean_messages)
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed for provider '{self.provider}': {e}")
                last_err = e
                time.sleep(1)

        raise RuntimeError(f"LLM generation failed after 2 attempts: {last_err}")

    def generate_stream(
        self, system_prompt: str, messages: list[dict[str, str]]
    ) -> Generator[str, None, None]:
        """Stream response chunks from provider or yield whole response in chunks."""
        full_text = self.generate(system_prompt, messages)
        # Yield words with small pauses to simulate natural streaming if native streaming API is unavailable
        words = full_text.split(" ")
        chunk = []
        for w in words:
            chunk.append(w)
            if len(chunk) >= 3:
                yield " ".join(chunk) + " "
                chunk = []
        if chunk:
            yield " ".join(chunk)

    def _call_openai(self, system: str, messages: list[dict[str, str]]) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload_messages = [{"role": "system", "content": system}] + messages
        data = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": 0.3,
        }

        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"].strip()

    def _call_gemini(self, system: str, messages: list[dict[str, str]]) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        data = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.3},
        }

        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _call_anthropic(self, system: str, messages: list[dict[str, str]]) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        data = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.3,
        }

        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["content"][0]["text"].strip()

    def _call_ollama(self, system: str, messages: list[dict[str, str]]) -> str:
        url = os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
        headers = {"Content-Type": "application/json"}
        payload_messages = [{"role": "system", "content": system}] + messages
        data = {
            "model": self.model,
            "messages": payload_messages,
            "stream": False,
        }

        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["message"]["content"].strip()


def get_llm_client() -> LLMClient:
    """Return an instance of LLMClient configured from environment."""
    return LLMClient()

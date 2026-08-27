"""Mastishka assistant package — a lightweight rule-based Q&A chatbot for NeuroScan AI."""

from .knowledge_base import SUGGESTED_QUESTIONS
from .mastishka import get_response

__all__ = ["get_response", "SUGGESTED_QUESTIONS"]

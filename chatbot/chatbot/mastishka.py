"""Mastishka: a lightweight, offline, rule-based Q&A assistant for NeuroScan AI.

No external API or API key is required — this is a keyword/fuzzy-matching
FAQ engine, which keeps the capstone project self-contained and free to run.
It optionally becomes "context-aware": if a scan result dict (matching the
shape returned by backend.analyze_uploaded_scan) is passed in, it can answer
questions about the user's most recent scan in plain language.
"""

from __future__ import annotations

import difflib
import random
import re
from typing import Any

from .knowledge_base import FALLBACK_RESPONSES, INTENTS

_WORD_RE = re.compile(r"[a-zA-Z']+")

# Common words filtered out before scoring so generic question phrasing
# ("what does X mean", "tell me about the Y") doesn't inflate overlap with
# the wrong intent just because both messages happen to be questions.
_STOPWORDS = {
    "a", "about", "an", "and", "are", "can", "could", "do", "does", "for",
    "give", "how", "i", "in", "is", "it", "its", "me", "mean", "means",
    "my", "of", "on", "please", "should", "tell", "that", "the", "there",
    "this", "to", "was", "what", "when", "where", "which", "who", "will",
    "with", "would", "you", "your",
}

# Phrases that indicate the user is asking about their own scan result,
# handled separately from the static knowledge base since the answer is dynamic.
_RESULT_QUERY_PATTERNS = [
    "my result", "my scan", "my report", "my prediction", "my diagnosis",
    "explain my result", "explain my scan", "what does my result mean",
    "what did i get", "what was i diagnosed", "explain this result",
    "explain the result", "what does this mean",
]


def _tokenize(text: str, drop_stopwords: bool = True) -> set[str]:
    words = {w.lower() for w in _WORD_RE.findall(text)}
    if not drop_stopwords:
        return words
    filtered = words - _STOPWORDS
    # If filtering stopwords wipes out a very short pattern/message
    # entirely (e.g. the message really is just "what is this"), fall
    # back to the unfiltered set rather than losing all signal.
    return filtered or words


def _score_intent(user_tokens: set[str], patterns: list[str]) -> float:
    """Score how well a message matches an intent's example patterns.

    Uses a token-level F1 score (so both short, on-topic messages AND longer
    messages with extra words score well) plus light per-word fuzzy matching
    to tolerate small typos. Whole-string fuzzy matching is deliberately
    avoided: on short patterns it produces misleading high scores for
    unrelated, coincidentally similar-looking text.
    """
    if not user_tokens:
        return 0.0
    best = 0.0
    for pattern in patterns:
        pattern_tokens = _tokenize(pattern)
        if not pattern_tokens:
            continue
        matched = 0.0
        for token in pattern_tokens:
            if token in user_tokens:
                matched += 1.0
            elif len(token) > 3 and difflib.get_close_matches(token, user_tokens, n=1, cutoff=0.84):
                matched += 0.8  # small credit for a likely typo
        recall = matched / len(pattern_tokens)
        precision = matched / len(user_tokens)
        if recall + precision == 0:
            continue
        f1 = 2 * recall * precision / (recall + precision)
        best = max(best, f1)
    return best


def _is_result_query(user_text: str) -> bool:
    lowered = user_text.lower()
    return any(phrase in lowered for phrase in _RESULT_QUERY_PATTERNS)


def _explain_scan_result(scan_result: dict[str, Any]) -> str:
    predicted_class = scan_result.get("class", "Unknown")
    confidence = scan_result.get("confidence")
    confidence_text = scan_result.get("confidence_text", "")
    severity = scan_result.get("severity", "Unknown")
    quality = (scan_result.get("quality") or {}).get("label", "Unknown")
    focus_region = scan_result.get("focus_region", "an unspecified region")

    confidence_pct = f"{confidence * 100:.1f}%" if isinstance(confidence, (int, float)) else "N/A"

    if predicted_class == "Normal":
        return (
            f"Your most recent scan was classified as **Normal** ({confidence_text.lower() or 'model prediction'}, "
            f"{confidence_pct} confidence). MRI quality was rated **{quality}**. No dominant tumour-like "
            f"pattern was detected, but routine clinical correlation is still recommended."
        )
    return (
        f"Your most recent scan was classified as **{predicted_class}** ({confidence_text.lower() or 'model prediction'}, "
        f"{confidence_pct} confidence). The model focused most on {focus_region}. Estimated severity: "
        f"**{severity}**. MRI quality was rated **{quality}**. Remember: this is an AI screening signal, "
        f"not a clinical diagnosis — please discuss it with a qualified radiologist or clinician."
    )


def get_response(message: str, scan_result: dict[str, Any] | None = None, threshold: float = 0.34) -> str:
    """Return Mastishka's reply to a user message.

    Args:
        message: raw user text.
        scan_result: optional dict from backend.analyze_uploaded_scan for the
            user's most recent scan, enabling context-aware answers.
        threshold: minimum match score (0-1) required to accept an intent.
    """
    message = (message or "").strip()
    if not message:
        return "Ask me anything about NeuroScan AI — tumour types, how to use the app, or your latest result."

    if _is_result_query(message):
        if scan_result:
            return _explain_scan_result(scan_result)
        return (
            "I don't see a scan result yet in this session. Head to 'Upload & Scan', analyze an MRI, "
            "and then ask me again — I'll be able to explain it."
        )

    user_tokens = _tokenize(message)
    best_intent = None
    best_score = 0.0
    for intent in INTENTS:
        score = _score_intent(user_tokens, intent["patterns"])
        if score > best_score:
            best_score = score
            best_intent = intent

    if best_intent is not None and best_score >= threshold:
        return random.choice(best_intent["responses"])

    return random.choice(FALLBACK_RESPONSES)

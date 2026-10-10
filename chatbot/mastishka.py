"""Mastishka: Hybrid RAG Medical AI Assistant for NeuroScan AI.

Combines emergency & diagnostic safety protocols, platform rules, hybrid keyword+TF-IDF
retrieval over NCI/NINDS knowledge bases, provider-agnostic LLM generation (ChatGPT style),
context-aware conversation history follow-ups, and graceful offline fallback.
"""

from __future__ import annotations

import difflib
import os
import random
import re
from typing import Any

from .knowledge_base import FALLBACK_RESPONSES, INTENTS, SUGGESTED_QUESTIONS
from .llm_client import get_llm_client, sanitize_pii
from .retriever import DEFAULT_RETRIEVAL_THRESHOLD, extract_keywords, get_retriever

_WORD_RE = re.compile(r"[a-zA-Z']+")

_RESULT_QUERY_PATTERNS = [
    "my result", "my scan", "my report", "my prediction", "my diagnosis",
    "explain my result", "explain my scan", "what does my result mean",
    "what did i get", "what was i diagnosed", "explain this result",
    "explain the result", "what does this mean",
]

EMERGENCY_KEYWORDS = [
    "sudden severe headache",
    "seizure now",
    "loss of consciousness",
    "unconscious",
    "can't speak",
    "cant speak",
    "cannot speak",
    "suicidal",
    "suicidal thoughts",
    "thinking of suicide",
]

EMERGENCY_RESPONSE = (
    "🚨 **EMERGENCY MEDICAL NOTICE**: If you or someone nearby is experiencing urgent or severe "
    "symptoms (such as a sudden severe headache, active seizure, loss of consciousness, "
    "inability to speak, or suicidal thoughts), **please seek immediate emergency medical care** "
    "(call 911 or your local emergency number, or go to the nearest emergency room immediately)."
)

DIAGNOSE_QUERY_PATTERNS = [
    "do i have a tumor",
    "do i have a brain tumor",
    "do i have cancer",
    "diagnose me",
    "is this a tumor in my brain",
    "confirm if i have a tumor",
    "tell me if i have a tumor",
    "am i sick",
]

DIAGNOSE_DISCLAIMER_RESPONSE = (
    "No. This chatbot provides general educational information and should "
    "not be used as a substitute for professional medical diagnosis or treatment. "
    "If you have concerns about personal symptoms or scan findings, please consult a qualified neurologist or radiologist."
)

DISCLAIMER_CATEGORIES = {
    "Treatment",
    "Prognosis",
    "Symptoms",
    "Medical Guidance",
    "Risk Factors",
}

MEDICAL_DISCLAIMER = (
    "*Disclaimer: This information is for educational purposes only and is not a substitute "
    "for professional medical advice, diagnosis, or treatment.*"
)

SYSTEM_PROMPT = (
    "You are Mastishka, a friendly medical education assistant inside NeuroScan AI. "
    "Answer using the provided KNOWLEDGE facts first. You may add general, well-established "
    "medical context and explain in simple words, but never invent statistics or specific drug doses. "
    "Never diagnose or give personal treatment advice; recommend seeing a doctor for personal concerns. "
    "If the question is unrelated to health/brain/the app, answer briefly and politely steer back, "
    "or answer normally if harmless. Be conversational, use short paragraphs and bullet points "
    "when helpful, and match the user's language."
)


def _is_result_query(user_text: str) -> bool:
    lowered = user_text.lower()
    return any(phrase in lowered for phrase in _RESULT_QUERY_PATTERNS)


def _is_emergency_query(user_text: str) -> bool:
    lowered = user_text.lower()
    return any(kw in lowered for kw in EMERGENCY_KEYWORDS)


def _is_diagnose_query(user_text: str) -> bool:
    lowered = user_text.lower()
    return any(pattern in lowered for pattern in DIAGNOSE_QUERY_PATTERNS)


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
            f"pattern was detected, but routine clinical correlation is still recommended.\n\n"
            f"*Remember: This AI screening result is for research/education only and is not a clinical diagnosis.*"
        )
    return (
        f"Your most recent scan was classified as **{predicted_class}** ({confidence_text.lower() or 'model prediction'}, "
        f"{confidence_pct} confidence). The model focused most on {focus_region}. Estimated severity: "
        f"**{severity}**. MRI quality was rated **{quality}**.\n\n"
        f"*Remember: This AI screening result is for research/education only and is not a clinical diagnosis — "
        f"please discuss it with a qualified radiologist or neuro-oncologist.*"
    )


def extract_context_keywords_from_history(chat_history: list[dict[str, str]] | None) -> set[str]:
    """Extract topic keywords from recent conversation turns to support follow-ups."""
    if not chat_history:
        return set()
    recent = chat_history[-4:]
    text = " ".join([m.get("content", "") for m in recent])
    return extract_keywords(text)


def get_response_dict(
    message: str,
    chat_history: list[dict[str, str]] | None = None,
    scan_result: dict[str, Any] | None = None,
    threshold: float = DEFAULT_RETRIEVAL_THRESHOLD,
) -> dict[str, Any]:
    """Process user message and return structured response dictionary with RAG metadata."""
    message = (message or "").strip()
    message = sanitize_pii(message)
    if not message:
        return {
            "answer": "Ask me anything about NeuroScan AI — tumour types, how to use the app, or your latest result.",
            "intent": "greeting",
            "category": "Platform",
            "source": "System",
            "confidence": 1.0,
            "retrieved_facts": [],
            "suggested_questions": SUGGESTED_QUESTIONS[:3],
        }

    # 1. Emergency Medical Safety Check
    if _is_emergency_query(message):
        return {
            "answer": EMERGENCY_RESPONSE,
            "intent": "emergency",
            "category": "Emergency",
            "source": "Safety Protocol",
            "confidence": 1.0,
            "retrieved_facts": [],
            "suggested_questions": [],
        }

    # 2. Self-Diagnosis Safeguard Check
    if _is_diagnose_query(message):
        return {
            "answer": DIAGNOSE_DISCLAIMER_RESPONSE,
            "intent": "chatbot_limitations",
            "category": "Safety",
            "source": "NCI",
            "confidence": 1.0,
            "retrieved_facts": [],
            "suggested_questions": [],
        }

    # 3. Dynamic Scan Result Explanation
    if _is_result_query(message):
        if scan_result:
            return {
                "answer": _explain_scan_result(scan_result),
                "intent": "explain_scan_result",
                "category": "Platform",
                "source": "NeuroScan Model",
                "confidence": 1.0,
                "retrieved_facts": [],
                "suggested_questions": [],
            }
        return {
            "answer": (
                "I don't see a scan result yet in this session. Head to 'Upload & Scan', analyze an MRI, "
                "and then ask me again — I'll be able to explain it."
            ),
            "intent": "explain_scan_result",
            "category": "Platform",
            "source": "System",
            "confidence": 1.0,
            "retrieved_facts": [],
            "suggested_questions": [],
        }

    # 4. Hybrid Knowledge Base Retrieval
    context_kw = extract_context_keywords_from_history(chat_history)
    retriever = get_retriever()
    retrieved_facts = retriever.retrieve(message, top_k=4, context_keywords=context_kw)

    top_score = retrieved_facts[0]["score"] if retrieved_facts else 0.0
    top_intent = retrieved_facts[0]["intent"] if retrieved_facts else "fallback"
    top_category = retrieved_facts[0]["category"] if retrieved_facts else "Fallback"
    top_sources = list({f["source"] for f in retrieved_facts if f.get("source")})

    # 5. LLM Generation Path (if configured & available)
    llm_client = get_llm_client()
    if llm_client.is_available() and top_score >= threshold:
        try:
            # Build prompt with retrieved facts + chat history
            facts_text = "\n".join([
                f"- Fact {i+1} [{f['source']}]: {f['question']} -> {f['answer']}"
                for i, f in enumerate(retrieved_facts)
            ])
            augmented_system = f"{SYSTEM_PROMPT}\n\nKNOWLEDGE FACTS:\n{facts_text}"

            # Pass last 6 conversation turns
            history_turns = []
            if chat_history:
                for turn in chat_history[-6:]:
                    history_turns.append({
                        "role": turn["role"],
                        "content": sanitize_pii(turn["content"]),
                    })
            history_turns.append({"role": "user", "content": sanitize_pii(message)})

            llm_response = llm_client.generate(augmented_system, history_turns)

            return {
                "answer": llm_response,
                "intent": top_intent,
                "category": top_category,
                "source": ", ".join(top_sources) if top_sources else "NCI/NINDS",
                "confidence": round(top_score, 4),
                "retrieved_facts": retrieved_facts,
                "suggested_questions": [],
            }
        except Exception as e:
            logger.warning(f"LLM generation failed, falling back to offline retrieval: {e}")

    # 6. Offline Fallback Mode
    if retrieved_facts and top_score >= threshold:
        best_fact = retrieved_facts[0]
        answer_text = best_fact["answer"]

        return {
            "answer": answer_text,
            "intent": best_fact["intent"],
            "category": best_fact["category"],
            "source": best_fact["source"],
            "confidence": round(top_score, 4),
            "retrieved_facts": retrieved_facts,
            "suggested_questions": [],
        }

    # Low Confidence Fallback
    fallback_text = (
        "I'm not sure about that. I can answer general questions about brain tumors, "
        "MRI, symptoms and treatment. For personal medical advice please consult a doctor."
    )
    return {
        "answer": fallback_text,
        "intent": "fallback",
        "category": "Fallback",
        "source": "System",
        "confidence": round(top_score, 4),
        "retrieved_facts": retrieved_facts,
        "suggested_questions": [
            "What are common brain tumor symptoms?",
            "What treatments are available for brain tumors?",
            "Why is MRI used for brain tumors?",
        ],
    }


def get_response(
    message: str,
    chat_history: list[dict[str, str]] | None = None,
    scan_result: dict[str, Any] | None = None,
    threshold: float = DEFAULT_RETRIEVAL_THRESHOLD,
) -> str:
    """Return Mastishka's formatted text response for a user message."""
    res = get_response_dict(message, chat_history=chat_history, scan_result=scan_result, threshold=threshold)
    answer = res["answer"]
    category = res.get("category")
    source = res.get("source")

    # Append medical disclaimer if sensitive category
    if category in DISCLAIMER_CATEGORIES and MEDICAL_DISCLAIMER not in answer:
        answer = f"{answer}\n\n{MEDICAL_DISCLAIMER}"

    # Append Source caption if response comes from dataset
    if source in {"NCI", "NINDS", "NCI, NINDS"} and "Source:" not in answer:
        answer = f"{answer}\n\n*Sources: {source}*"

    # Append suggested questions if fallback
    suggested = res.get("suggested_questions")
    if res.get("intent") == "fallback" and suggested:
        suggestions_str = "\n".join([f"• {q}" for q in suggested])
        answer = f"{answer}\n\n**Suggested Questions:**\n{suggestions_str}"

    return answer

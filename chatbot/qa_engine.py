"""QA Retrieval Engine for Mastishka using TF-IDF and RapidFuzz.

Calculates similarity between user queries and dataset questions offline using
scikit-learn TF-IDF (word 1-2 grams + char_wb 3-4 grams) and RapidFuzz scoring.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.35
DATA_PATH = Path(__file__).parent / "data" / "knowledge_base.csv"

FALLBACK_MESSAGE = (
    "I'm not sure about that. I can answer general questions about brain tumors, "
    "MRI, symptoms and treatment. For personal medical advice please consult a doctor."
)

DEFAULT_SUGGESTED_QUESTIONS = [
    "What are common brain tumor symptoms?",
    "What treatments are available for brain tumors?",
    "Why is MRI used for brain tumors?",
]

SYNONYM_REPLACEMENTS = [
    (re.compile(r"\btumour\b", re.IGNORECASE), "tumor"),
    (re.compile(r"\btumours\b", re.IGNORECASE), "tumors"),
    (re.compile(r"\bmri\b", re.IGNORECASE), "magnetic resonance imaging"),
    (re.compile(r"\bmris\b", re.IGNORECASE), "magnetic resonance imaging"),
    (re.compile(r"\bct\b", re.IGNORECASE), "computed tomography"),
]


def preprocess_text(text: str) -> str:
    """Preprocess query or corpus text: lowercase, expand synonyms, strip punctuation."""
    if not text:
        return ""
    lowered = text.lower()
    for pattern, replacement in SYNONYM_REPLACEMENTS:
        lowered = pattern.sub(replacement, lowered)
    cleaned = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


class QAEngine:
    """TF-IDF and RapidFuzz powered Question Answering engine."""

    def __init__(self, data_path: Path | str = DATA_PATH) -> None:
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            from .build_knowledge_base import build_knowledge_base
            build_knowledge_base()

        self.df = pd.read_csv(self.data_path)
        self.df["processed_question"] = self.df["question"].astype(str).apply(preprocess_text)

        self.word_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            stop_words="english",
        )
        self.char_vec = TfidfVectorizer(
            ngram_range=(3, 4),
            analyzer="char_wb",
            sublinear_tf=True,
        )

        questions = self.df["processed_question"].tolist()
        X_word = self.word_vec.fit_transform(questions)
        X_char = self.char_vec.fit_transform(questions)

        # Weighted combination: 0.80 word + 0.20 char_wb
        raw_matrix = hstack([X_word * 0.80, X_char * 0.20]).tocsr()
        self.X_matrix = normalize(raw_matrix, norm="l2", axis=1)

    def query(
        self,
        message: str,
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> dict[str, Any]:
        """Search the knowledge base for the best matching Q&A entry.

        Returns a dictionary containing:
            answer, intent, category, source, confidence, matched_question,
            and suggested_questions (if fallback).
        """
        clean_msg = preprocess_text(message)
        if not clean_msg:
            return {
                "answer": FALLBACK_MESSAGE,
                "intent": "fallback",
                "category": "Fallback",
                "source": "System",
                "confidence": 0.0,
                "matched_question": None,
                "suggested_questions": DEFAULT_SUGGESTED_QUESTIONS,
            }

        # TF-IDF cosine similarity
        q_word = self.word_vec.transform([clean_msg])
        q_char = self.char_vec.transform([clean_msg])

        if q_word.nnz == 0:
            # No non-stopword content match
            return {
                "answer": FALLBACK_MESSAGE,
                "intent": "fallback",
                "category": "Fallback",
                "source": "System",
                "confidence": 0.0,
                "matched_question": None,
                "suggested_questions": DEFAULT_SUGGESTED_QUESTIONS,
            }

        raw_q_vec = hstack([q_word * 0.80, q_char * 0.20]).tocsr()
        q_vec = normalize(raw_q_vec, norm="l2", axis=1)
        cosine_sims = linear_kernel(q_vec, self.X_matrix)[0]

        best_idx = int(np.argmax(cosine_sims))
        best_tfidf_score = float(cosine_sims[best_idx])

        final_confidence = best_tfidf_score
        if HAS_RAPIDFUZZ:
            matched_proc_q = self.df.iloc[best_idx]["processed_question"]
            fuzzy_ratio = fuzz.token_set_ratio(clean_msg, matched_proc_q) / 100.0
            if fuzzy_ratio >= 0.85 and best_tfidf_score < threshold:
                final_confidence = max(best_tfidf_score, fuzzy_ratio * 0.75)

        row = self.df.iloc[best_idx]

        if final_confidence >= threshold:
            return {
                "answer": row["answer"],
                "intent": row["intent"],
                "category": row["category"],
                "source": row["source"],
                "confidence": round(float(final_confidence), 4),
                "matched_question": row["question"],
                "suggested_questions": [],
            }

        return {
            "answer": FALLBACK_MESSAGE,
            "intent": "fallback",
            "category": "Fallback",
            "source": "System",
            "confidence": round(float(final_confidence), 4),
            "matched_question": row["question"],
            "suggested_questions": DEFAULT_SUGGESTED_QUESTIONS,
        }


_SINGLETON_ENGINE: QAEngine | None = None


def get_qa_engine() -> QAEngine:
    """Return a cached lazy singleton instance of QAEngine."""
    global _SINGLETON_ENGINE
    if _SINGLETON_ENGINE is None:
        _SINGLETON_ENGINE = QAEngine()
    return _SINGLETON_ENGINE

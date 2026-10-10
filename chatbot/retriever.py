"""Hybrid Retrieval Engine for Mastishka Q&A Knowledge Base.

Combines 50% keyword/synonym match scoring and 50% TF-IDF cosine similarity
over questions, answers, and categories from knowledge_base.csv.
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

logger = logging.getLogger(__name__)

DEFAULT_RETRIEVAL_THRESHOLD = 0.25
DATA_PATH = Path(__file__).parent / "data" / "knowledge_base.csv"

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "could", "did", "do", "does",
    "doing", "down", "during", "each", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself",
    "hi", "hello", "hey", "patient", "doctor", "please", "tell", "ask", "want",
    "know", "info", "information", "help", "thanks", "thank", "email", "phone",
}

SYNONYM_MAP = {
    "tumour": "tumor",
    "tumours": "tumor",
    "tumors": "tumor",
    "growth": "tumor",
    "mass": "tumor",
    "lesion": "tumor",
    "mri": "mri",
    "scan": "mri",
    "magnetic": "mri",
    "resonance": "mri",
    "imaging": "mri",
    "headache": "headache",
    "headaches": "headache",
    "pain": "headache",
    "cure": "treatment",
    "treat": "treatment",
    "treatment": "treatment",
    "treatments": "treatment",
    "therapy": "treatment",
    "chemo": "chemotherapy",
    "kids": "children",
    "child": "children",
    "pediatric": "children",
    "signs": "symptoms",
    "sign": "symptoms",
    "symptom": "symptoms",
    "symptoms": "symptoms",
    "head": "brain",
    "mind": "brain",
}


def simple_stem(word: str) -> str:
    """Lightweight suffix stripper for stemming words when NLTK is omitted."""
    w = word.lower()
    if len(w) <= 3:
        return w
    for suffix in ["ing", "edly", "able", "ment", "tion", "ness", "ies", "es", "ed", "s"]:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


def extract_keywords(text: str) -> set[str]:
    """Extract clean, stemmed, synonym-expanded keywords from text."""
    if not text:
        return set()
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    words = cleaned.split()
    keywords = set()
    for w in words:
        if w in STOPWORDS or len(w) < 2 or "redacted" in w:
            continue
        mapped = SYNONYM_MAP.get(w, w)
        stemmed = simple_stem(mapped)
        keywords.add(mapped)
        keywords.add(stemmed)
    return keywords


class KnowledgeRetriever:
    """Hybrid RAG retriever combining 50% keyword overlap and 50% TF-IDF similarity."""

    def __init__(self, data_path: Path | str = DATA_PATH) -> None:
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            from .build_knowledge_base import build_knowledge_base
            build_knowledge_base()

        self.df = pd.read_csv(self.data_path)
        for col in ["question", "answer", "intent", "category", "source"]:
            self.df[col] = self.df[col].astype(str).str.strip()

        # Build corpus combining question and answer for full-text index
        self.df["combined_text"] = (
            self.df["question"] + " " + self.df["answer"] + " " + self.df["category"] + " " + self.df["intent"]
        )

        # Build per-row keyword sets
        self.df["keywords"] = self.df["combined_text"].apply(extract_keywords)

        # Precompute normalized clean question strings for exact matching
        self.df["clean_question_norm"] = self.df["question"].apply(lambda q: re.sub(r"[^\w\s]", "", q.lower()).strip())

        # Build TF-IDF vectorizer over combined corpus
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

        corpus_texts = self.df["combined_text"].tolist()
        X_word = self.word_vec.fit_transform(corpus_texts)
        X_char = self.char_vec.fit_transform(corpus_texts)

        raw_matrix = hstack([X_word * 0.80, X_char * 0.20]).tocsr()
        self.X_matrix = normalize(raw_matrix, norm="l2", axis=1)

    def retrieve(
        self,
        query_text: str,
        top_k: int = 4,
        context_keywords: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve top-k matching knowledge base entries using hybrid scoring.

        Args:
            query_text: user question or message text.
            top_k: number of top results to return (default: 4).
            context_keywords: optional set of keywords from conversation history.

        Returns:
            List of dicts, each containing:
                id, question, answer, intent, category, source, keyword_score, tfidf_score, final_score
        """
        if not query_text or not query_text.strip():
            return []

        clean_query = re.sub(r"[^\w\s]", "", query_text.lower()).strip()
        raw_query_kw = extract_keywords(query_text)

        # Exact match boost
        exact_boost = np.zeros(len(self.df), dtype=float)
        for idx, q_norm in enumerate(self.df["clean_question_norm"]):
            if q_norm == clean_query:
                exact_boost[idx] = 1.0

        # 1. Keyword Overlap Scoring
        kw_scores = np.zeros(len(self.df), dtype=float)
        target_kw = raw_query_kw if raw_query_kw else (context_keywords or set())
        if target_kw:
            for idx, candidate_kw in enumerate(self.df["keywords"]):
                if not candidate_kw:
                    continue
                main_match = len(raw_query_kw.intersection(candidate_kw)) if raw_query_kw else 0
                ctx_match = len(context_keywords.intersection(candidate_kw)) if (context_keywords and raw_query_kw) else 0
                denom = max(len(raw_query_kw), 1) if raw_query_kw else max(len(context_keywords or []), 1)
                score_val = (main_match + 0.3 * ctx_match) / denom
                kw_scores[idx] = min(1.0, score_val)

        # 2. TF-IDF Cosine Similarity
        clean_query_space = re.sub(r"\[[A-Z_]+\]", " ", query_text.lower())
        clean_query_space = re.sub(r"[^\w\s]", " ", clean_query_space).strip()
        q_word = self.word_vec.transform([clean_query_space])
        q_char = self.char_vec.transform([clean_query_space])

        if q_word.nnz == 0 and q_char.nnz == 0:
            tfidf_scores = np.zeros(len(self.df), dtype=float)
        else:
            raw_q_vec = hstack([q_word * 0.80, q_char * 0.20]).tocsr()
            q_vec = normalize(raw_q_vec, norm="l2", axis=1)
            tfidf_scores = linear_kernel(q_vec, self.X_matrix)[0]

        # 3. Hybrid Combined Score: 0.50 Keyword + 0.50 TF-IDF (with max credit for exact match)
        hybrid_scores = 0.50 * kw_scores + 0.50 * tfidf_scores
        final_scores = np.maximum(exact_boost, hybrid_scores)

        top_indices = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(final_scores[idx])
            row = self.df.iloc[idx]
            results.append({
                "id": int(row["id"]),
                "question": row["question"],
                "answer": row["answer"],
                "intent": row["intent"],
                "category": row["category"],
                "source": row["source"],
                "keyword_score": round(float(kw_scores[idx]), 4),
                "tfidf_score": round(float(tfidf_scores[idx]), 4),
                "score": round(score, 4),
            })

        return results


_SINGLETON_RETRIEVER: KnowledgeRetriever | None = None


def get_retriever() -> KnowledgeRetriever:
    """Return a cached lazy singleton instance of KnowledgeRetriever."""
    global _SINGLETON_RETRIEVER
    if _SINGLETON_RETRIEVER is None:
        _SINGLETON_RETRIEVER = KnowledgeRetriever()
    return _SINGLETON_RETRIEVER

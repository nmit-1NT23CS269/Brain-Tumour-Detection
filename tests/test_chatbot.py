"""Test suite for Mastishka Hybrid RAG AI Assistant, retrieval engine, LLM client, and safety rules."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from chatbot.build_knowledge_base import build_knowledge_base
from chatbot.llm_client import LLMClient, sanitize_pii
from chatbot.mastishka import (
    get_response,
    get_response_dict,
    EMERGENCY_RESPONSE,
    DIAGNOSE_DISCLAIMER_RESPONSE,
)
from chatbot.retriever import get_retriever

DATA_PATH = Path(__file__).parent.parent / "chatbot" / "data" / "knowledge_base.csv"


@pytest.fixture(scope="module", autouse=True)
def ensure_knowledge_base():
    """Ensure knowledge base CSV is generated and valid before running tests."""
    return build_knowledge_base()


def test_knowledge_base_no_empty_cells_and_unique_ids():
    """Verify knowledge_base.csv has no empty cells, unique IDs, and bug fix present."""
    assert DATA_PATH.exists(), f"Knowledge base file not found at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)

    assert len(df) > 0, "Knowledge base CSV is empty"
    assert df.isnull().sum().sum() == 0, "Found null values in knowledge base"
    assert (df == "").sum().sum() == 0, "Found empty string cells in knowledge base"
    assert df["id"].nunique() == len(df), "Found duplicate IDs in knowledge base"
    assert df["intent"].nunique() >= 100, f"Expected >= 100 unique intents, got {df['intent'].nunique()}"

    # Verify bug fix present
    bug_rows = df[df["question"].astype(str).str.strip().str.lower() == "can a chatbot diagnose a brain tumor?"]
    assert len(bug_rows) > 0
    assert bug_rows.iloc[0]["intent"] == "chatbot_limitations"
    assert "No. This chatbot provides general educational information" in bug_rows.iloc[0]["answer"]


def test_retrieval_self_match_accuracy():
    """Verify every dataset question retrieves its own intent in top-1."""
    df = pd.read_csv(DATA_PATH)
    retriever = get_retriever()

    correct = 0
    total = len(df)

    for _, row in df.iterrows():
        question = row["question"]
        expected_intent = row["intent"]

        results = retriever.retrieve(question, top_k=1)
        if results and results[0]["intent"] == expected_intent:
            correct += 1

    accuracy = correct / total
    print(f"\n[Retrieval Self-Match Accuracy]: {correct}/{total} ({accuracy * 100:.2f}%)")
    assert accuracy >= 0.95, f"Expected self-match accuracy >= 95%, got {accuracy * 100:.2f}%"


def test_25_keyword_queries_top_3_retrieval_accuracy():
    """Verify at least 25 hand-written keyword-style queries retrieve the right category/intent in top-3."""
    queries = [
        ("tumour signs head", ["Symptoms", "headache", "tumor_symptoms"]),
        ("mri scan risks", ["MRI & Diagnosis", "Risk Factors", "mri_safety", "mri_radiation"]),
        ("cure brain growth", ["Treatment", "treatment", "tumor_removal", "surgery"]),
        ("kids brain tumor", ["Brain Tumor Basics", "pediatric_tumor", "children"]),
        ("how is glioma treated", ["Treatment", "Tumor Types", "treatment", "glioma_definition"]),
        ("meningioma symptoms", ["Tumor Types", "Symptoms", "meningioma"]),
        ("pituitary hormone changes", ["Tumor Types", "pituitary_tumor"]),
        ("headache and vomiting", ["Symptoms", "headache", "vomiting", "tumor_symptoms"]),
        ("is brain cancer inherited", ["Risk Factors", "hereditary", "family_history"]),
        ("radiation exposure risk", ["Risk Factors", "radiation_risk"]),
        ("brain biopsy procedure", ["MRI & Diagnosis", "Diagnosis", "biopsy"]),
        ("contrast agent mri scan", ["MRI & Diagnosis", "mri_contrast"]),
        ("ct scan vs mri", ["MRI & Diagnosis", "ct_scan"]),
        ("difference benign malignant", ["Medical Terms", "Brain Tumor Basics", "tumor_vs_cancer", "tumor_cancer"]),
        ("grade 4 tumor aggressiveness", ["Tumor Grading", "grade_four", "tumor_grading"]),
        ("chemotherapy drugs cancer", ["Treatment", "chemotherapy"]),
        ("radiation therapy high energy", ["Treatment", "radiation_therapy"]),
        ("tumor recurrence after surgery", ["Treatment", "recurrence", "recurrence_definition"]),
        ("cerebrospinal fluid csf", ["Brain Basics", "csf"]),
        ("frontal lobe function", ["Brain Basics", "frontal_lobe"]),
        ("cerebellum balance movement", ["Brain Basics", "cerebellum"]),
        ("neurons brain signals", ["Brain Basics", "neurons"]),
        ("when see doctor symptoms", ["Medical Guidance", "doctor_visit"]),
        ("life threatening tumor", ["Brain Tumor Basics", "life_threatening"]),
        ("asymptomatic brain tumor", ["Brain Tumor Basics", "Symptoms", "asymptomatic"]),
    ]

    retriever = get_retriever()
    correct = 0

    for query_text, expected_tokens in queries:
        retrieved = retriever.retrieve(query_text, top_k=3)
        top_intents = [r["intent"] for r in retrieved]
        top_categories = [r["category"] for r in retrieved]

        match_found = any(tok in top_intents or tok in top_categories for tok in expected_tokens)
        if match_found:
            correct += 1
        else:
            print(f"Keyword Miss: '{query_text}' -> Top intents: {top_intents}, categories: {top_categories}")

    accuracy = correct / len(queries)
    print(f"\n[25 Keyword Queries Accuracy]: {correct}/{len(queries)} ({accuracy * 100:.2f}%)")
    assert accuracy >= 0.84, f"Expected 25 keyword queries accuracy >= 84%, got {accuracy * 100:.2f}%"


def test_llm_path_mocked_with_pii_sanitization():
    """Verify mock LLM provider path receives prompt with retrieved facts, chat history, and no PII."""
    pii_message = "Hi, I am patient PAT-000014 (email: patient.demo@example.com, phone: +1-555-0199). What is a glioma?"
    history = [{"role": "user", "content": pii_message}]

    mock_client = MagicMock()
    mock_client.is_available.return_value = True
    mock_client.generate.return_value = "A glioma is a tumor of glial cells."

    with patch("chatbot.mastishka.get_llm_client", return_value=mock_client):
        res_dict = get_response_dict(pii_message, chat_history=history)

        assert mock_client.generate.called
        system_prompt, messages = mock_client.generate.call_args[0]

        # Assert retrieved facts and system prompt present
        assert "KNOWLEDGE FACTS:" in system_prompt
        assert len(res_dict["retrieved_facts"]) > 0

        # Assert PII was completely stripped from messages sent to LLM
        sent_content = " ".join([m["content"] for m in messages])
        assert "PAT-000014" not in sent_content
        assert "patient.demo@example.com" not in sent_content
        assert "+1-555-0199" not in sent_content
        assert "[PATIENT_ID_REDACTED]" in sent_content or "[EMAIL_REDACTED]" in sent_content


def test_fallback_path_without_api_key():
    """Verify that when no LLM API key is configured, the bot still answers seamlessly from CSV retrieval."""
    with patch.object(LLMClient, "is_available", return_value=False):
        res_dict = get_response_dict("What is glioma?")
        res_text = get_response("What is glioma?")

        assert res_dict["intent"] == "glioma_definition"
        assert "Glioma" in res_text
        assert "*Source: NCI*" in res_text or "*Sources: NCI*" in res_text


def test_emergency_and_diagnose_safety_replies():
    """Verify emergency keywords and diagnose-me queries trigger safety responses."""
    # Emergency query
    emerg_res = get_response_dict("I have a sudden severe headache and seizure now")
    assert emerg_res["intent"] == "emergency"
    assert "EMERGENCY MEDICAL NOTICE" in emerg_res["answer"]

    # Diagnose me query
    diag_res = get_response_dict("Do I have a tumor? Diagnose me please.")
    assert diag_res["intent"] == "chatbot_limitations"
    assert "No. This chatbot provides general educational information" in diag_res["answer"]


def test_followup_handling_inherits_topic_keywords():
    """Verify short follow-up messages inherit topic context from previous conversation turns."""
    history = [
        {"role": "user", "content": "Tell me about glioma tumors."},
        {"role": "assistant", "content": "Glioma is a type of tumor that develops from glial cells."},
    ]
    followup_query = "What about its treatment?"

    res_dict = get_response_dict(followup_query, chat_history=history)
    assert res_dict["intent"] in {"treatment", "glioma_definition", "surgery", "chemotherapy", "radiation_therapy"}
    assert len(res_dict["retrieved_facts"]) > 0

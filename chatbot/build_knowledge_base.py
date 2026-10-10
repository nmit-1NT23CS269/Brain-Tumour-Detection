"""Script to merge, clean, validate, and build the canonical Mastishka knowledge base.

Concatenates CSV datasets in chatbot/data/, fixes known data quality bugs,
harmonizes answers so each intent has a single canonical answer (logging reported conflicts),
drops duplicate question-intent pairs, validates intent-to-answer consistency,
and exports both knowledge_base.csv and knowledge_base.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

CANONICAL_CHATBOT_DIAGNOSE_ANSWER = (
    "No. This chatbot provides general educational information and should "
    "not be used as a substitute for professional medical diagnosis or treatment."
)


def build_knowledge_base() -> pd.DataFrame:
    dataset2_path = DATA_DIR / "brain_tumor_chatbot_dataset2.csv"
    dataset100_path = DATA_DIR / "brain_tumor_chatbot_dataset_100.csv"

    if not dataset2_path.exists() or not dataset100_path.exists():
        raise FileNotFoundError(
            f"Input dataset files missing in {DATA_DIR}. Ensure brain_tumor_chatbot_dataset2.csv "
            f"and brain_tumor_chatbot_dataset_100.csv exist."
        )

    df2 = pd.read_csv(dataset2_path)
    df100 = pd.read_csv(dataset100_path)

    # Clean whitespace across all string columns before merging
    for df in [df2, df100]:
        for col in ["question", "answer", "intent", "category", "source"]:
            df[col] = df[col].astype(str).str.strip()

    combined_df = pd.concat([df2, df100], ignore_index=True)
    logger.info(f"Loaded {len(df2)} rows from dataset2 and {len(df100)} rows from dataset100 (Total: {len(combined_df)}).")

    # Fix KNOWN BUG: in dataset2, row with question "Can a chatbot diagnose a brain tumor?"
    bug_mask = combined_df["question"].astype(str).str.strip().str.lower() == "can a chatbot diagnose a brain tumor?"
    bug_count = bug_mask.sum()
    if bug_count > 0:
        logger.info(f"FIXED KNOWN BUG: Found {bug_count} row(s) with question 'Can a chatbot diagnose a brain tumor?'. Updated answer to canonical disclaimer and intent to 'chatbot_limitations'.")
        combined_df.loc[bug_mask, "answer"] = CANONICAL_CHATBOT_DIAGNOSE_ANSWER
        combined_df.loc[bug_mask, "intent"] = "chatbot_limitations"

    # Harmonize canonical answers for each intent
    # Primary canonical map from df100 (which has 1 answer per intent)
    canonical_answer_map: dict[str, str] = {}
    for _, row in df100.iterrows():
        canonical_answer_map[row["intent"]] = row["answer"]

    # For intents in df2 not in df100, use the first answer encountered in df2
    for _, row in df2.iterrows():
        intent = row["intent"]
        if intent not in canonical_answer_map:
            canonical_answer_map[intent] = row["answer"]

    # Detect & report any conflicting answers before replacing
    intent_grouped = combined_df.groupby("intent")["answer"].unique()
    conflicts_reported = 0
    for intent, answers in intent_grouped.items():
        if len(answers) > 1:
            conflicts_reported += 1
            canonical = canonical_answer_map[intent]
            logger.info(f"REPORTED CONFLICT RESOLUTION for intent '{intent}': Found {len(answers)} different answers. Standardizing to canonical answer: '{canonical}'")

    if conflicts_reported > 0:
        logger.info(f"Reported and resolved conflicts across {conflicts_reported} intents.")

    # Apply canonical answers across all rows
    combined_df["answer"] = combined_df["intent"].map(canonical_answer_map)

    # Normalize question and intent for deduplication
    combined_df["q_clean"] = combined_df["question"].astype(str).str.strip().str.lower()
    combined_df["intent_clean"] = combined_df["intent"].astype(str).str.strip().str.lower()

    # Drop exact duplicate (q_clean, intent_clean) pairs
    before_dedup = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=["q_clean", "intent_clean"], keep="first").copy()
    after_dedup = len(combined_df)
    logger.info(f"Deduplicated (question, intent) pairs: dropped {before_dedup - after_dedup} exact duplicates, {after_dedup} rows remaining.")

    # Drop temporary cleaning columns
    combined_df.drop(columns=["q_clean", "intent_clean"], inplace=True)

    # Reassign unique sequential ID starting at 1
    combined_df["id"] = range(1, len(combined_df) + 1)

    # STRICT VALIDATIONS
    # Validation 1: Check for empty / null cells
    null_cells = combined_df.isnull().sum().sum()
    empty_strings = (combined_df == "").sum().sum()
    if null_cells > 0 or empty_strings > 0:
        raise ValueError(f"Validation failed: found {null_cells} null cells and {empty_strings} empty string cells in knowledge base.")

    # Validation 2: Ensure each intent maps to EXACTLY ONE canonical answer
    intent_answers_final = combined_df.groupby("intent")["answer"].unique()
    conflicting_intents = {intent: answers.tolist() for intent, answers in intent_answers_final.items() if len(answers) > 1}
    if conflicting_intents:
        error_msg = "Validation failed: The following intents still have conflicting answers:\n"
        for intent, answers in conflicting_intents.items():
            error_msg += f"  - Intent '{intent}': {answers}\n"
        raise ValueError(error_msg)

    # Validation 3: Unique IDs
    if combined_df["id"].nunique() != len(combined_df):
        raise ValueError("Validation failed: dataset has duplicate IDs after reassigning.")

    logger.info(f"VALIDATION PASSED: {len(combined_df)} rows, {combined_df['intent'].nunique()} unique intents, 0 empty cells, 0 conflicting answers, 0 duplicate IDs.")

    # Output CSV and JSON
    output_csv = DATA_DIR / "knowledge_base.csv"
    output_json = DATA_DIR / "knowledge_base.json"

    combined_df.to_csv(output_csv, index=False)
    logger.info(f"Saved {output_csv}")

    records = combined_df.to_dict(orient="records")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    logger.info(f"Saved {output_json}")

    return combined_df


if __name__ == "__main__":
    build_knowledge_base()

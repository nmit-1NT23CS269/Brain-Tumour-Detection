"""Data Leakage Audit & Near-Duplicate Detection Utility for NeuroScan AI.

Data leakage between training and testing splits inflates model test accuracy and produces
false confidence in evaluation metrics. This utility scans Training and Testing sets for:
1. Exact duplicates using MD5 binary hashing.
2. Near-duplicates using perceptual hashing (pHash) with Hamming distance thresholding.

WHAT TO DO WITH THE RESULTS:
-----------------------------
1. Review the output CSV report (`data/artifacts/reports/data_leakage_report.csv`).
2. If leakage percentage is > 0%, re-run this script with the `--remove` flag:
   `python utils/dedupe_check.py --remove`
   This automatically moves leaked test images out of the testing set into a quarantine directory
   (`data/artifacts/quarantine_leaked_test_images/`).
3. Re-evaluate the model using `python model/evaluate.py` to ensure reported test accuracy
   and recall reflect true generalization performance on unseen patient scans.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
import imagehash

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import LOGS_DIR, REPORTS_DIR, configure_logging, ensure_project_dirs

LOGGER = configure_logging(__name__)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageRecord:
    """Metadata and hash representations for a single image file."""

    path: Path
    filename: str
    split: str
    class_name: str
    md5: str
    phash: imagehash.ImageHash | None


def compute_md5(file_path: Path) -> str:
    """Compute the MD5 hash of a file in 64KB chunks."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_phash(file_path: Path) -> imagehash.ImageHash | None:
    """Compute perceptual hash (pHash) for an image file."""
    try:
        with Image.open(file_path) as img:
            return imagehash.phash(img.convert("RGB"))
    except Exception as err:
        LOGGER.warning("Could not compute pHash for %s: %s", file_path, err)
        return None


def scan_directory(dir_path: Path, split_label: str) -> list[ImageRecord]:
    """Recursively scan a directory and compute MD5 and pHash for all images."""
    records: list[ImageRecord] = []
    if not dir_path.exists():
        LOGGER.error("Directory not found: %s", dir_path)
        return records

    LOGGER.info("Scanning %s split in %s...", split_label, dir_path)
    for file_path in sorted(dir_path.rglob("*.*")):
        if file_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        class_name = file_path.parent.name
        md5_val = compute_md5(file_path)
        phash_val = compute_phash(file_path)

        records.append(
            ImageRecord(
                path=file_path,
                filename=file_path.name,
                split=split_label,
                class_name=class_name,
                md5=md5_val,
                phash=phash_val,
            )
        )

    LOGGER.info("Indexed %d images from %s split.", len(records), split_label)
    return records


def find_duplicates(
    train_records: list[ImageRecord],
    test_records: list[ImageRecord],
    hamming_threshold: int = 6,
) -> list[dict[str, Any]]:
    """Find exact (MD5) and near-duplicate (pHash) pairs between test and train splits."""
    LOGGER.info("Analyzing data leakage between train (%d) and test (%d) sets...", len(train_records), len(test_records))
    duplicates: list[dict[str, Any]] = []

    # Map MD5 -> list of train records for O(1) exact lookup
    train_md5_map: dict[str, list[ImageRecord]] = {}
    for tr in train_records:
        train_md5_map.setdefault(tr.md5, []).append(tr)

    for te in test_records:
        # 1. Exact MD5 matches
        if te.md5 in train_md5_map:
            for tr in train_md5_map[te.md5]:
                duplicates.append(
                    {
                        "test_image_path": str(te.path),
                        "test_filename": te.filename,
                        "test_class": te.class_name,
                        "train_image_path": str(tr.path),
                        "train_filename": tr.filename,
                        "train_class": tr.class_name,
                        "match_type": "exact_md5",
                        "hamming_distance": 0,
                    }
                )
            continue

        # 2. Near-duplicate pHash matches (if not exact match)
        if te.phash is not None:
            for tr in train_records:
                if tr.phash is None:
                    continue
                dist = te.phash - tr.phash  # Hamming distance
                if dist <= hamming_threshold:
                    duplicates.append(
                        {
                            "test_image_path": str(te.path),
                            "test_filename": te.filename,
                            "test_class": te.class_name,
                            "train_image_path": str(tr.path),
                            "train_filename": tr.filename,
                            "train_class": tr.class_name,
                            "match_type": "near_phash",
                            "hamming_distance": int(dist),
                        }
                    )

    LOGGER.info("Found %d duplicate/leaked pairs.", len(duplicates))
    return duplicates


def calculate_leakage_percentage(duplicates: list[dict[str, Any]], total_test_samples: int) -> float:
    """Calculate the percentage of unique test set images leaking from training."""
    if total_test_samples <= 0:
        return 0.0

    unique_leaked_test_paths = {d["test_image_path"] for d in duplicates}
    return (len(unique_leaked_test_paths) / total_test_samples) * 100.0


def export_csv_report(duplicates: list[dict[str, Any]], output_paths: Sequence[Path]) -> None:
    """Export duplicate pairs report to CSV files."""
    fieldnames = [
        "test_image_path",
        "test_filename",
        "test_class",
        "train_image_path",
        "train_filename",
        "train_class",
        "match_type",
        "hamming_distance",
    ]

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(duplicates)
    LOGGER.info("Exported duplicate report to CSV.")


def quarantine_leaked_images(duplicates: list[dict[str, Any]], quarantine_dir: Path) -> int:
    """Move leaked test images out of the testing set into quarantine_dir."""
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    moved_count = 0
    unique_test_paths = {d["test_image_path"] for d in duplicates}

    for test_path_str in unique_test_paths:
        src_path = Path(test_path_str)
        if not src_path.exists():
            continue

        dest_path = quarantine_dir / src_path.parent.name / src_path.name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(src_path, dest_path)
            moved_count += 1
            LOGGER.info("Quarantined leaked file: %s -> %s", src_path.name, dest_path)
        except Exception as err:
            LOGGER.error("Failed to move %s: %s", src_path, err)

    return moved_count


def auto_detect_splits() -> tuple[Path, Path]:
    """Auto-detect train and test directories from project defaults."""
    train_candidates = [Path("data/splits/train"), Path("data/raw/archive/Training"), Path("data/raw/Training")]
    test_candidates = [Path("data/splits/test"), Path("data/raw/archive/Testing"), Path("data/raw/Testing")]

    train_dir = next((p for p in train_candidates if p.exists()), None)
    test_dir = next((p for p in test_candidates if p.exists()), None)

    if not train_dir or not test_dir:
        raise FileNotFoundError("Could not auto-detect Training and Testing split directories.")

    return train_dir, test_dir


def main() -> None:
    """CLI entrypoint for deduplication and data leakage audit."""
    parser = argparse.ArgumentParser(description="Find exact and near-duplicate images between Training and Testing sets.")
    parser.add_argument("--train_dir", type=str, help="Path to Training split directory")
    parser.add_argument("--test_dir", type=str, help="Path to Testing split directory")
    parser.add_argument("--threshold", type=int, default=6, help="pHash Hamming distance threshold (default: 6)")
    parser.add_argument("--remove", action="store_true", help="Move leaked test images out of testing split to quarantine folder")
    args = parser.parse_args()

    ensure_project_dirs()

    if args.train_dir and args.test_dir:
        train_path, test_path = Path(args.train_dir), Path(args.test_dir)
    else:
        train_path, test_path = auto_detect_splits()

    print("=======================================================")
    print("      NEUROSCAN AI DATA LEAKAGE AUDIT & DEDUPE        ")
    print("=======================================================\n")
    print(f"Training Directory : {train_path}")
    print(f"Testing Directory  : {test_path}")
    print(f"pHash Threshold    : {args.threshold}\n")

    train_records = scan_directory(train_path, "train")
    test_records = scan_directory(test_path, "test")

    duplicates = find_duplicates(train_records, test_records, hamming_threshold=args.threshold)
    leakage_pct = calculate_leakage_percentage(duplicates, len(test_records))

    csv_paths = [REPORTS_DIR / "data_leakage_report.csv", LOGS_DIR / "data_leakage_report.csv"]
    export_csv_report(duplicates, csv_paths)

    unique_leaked_count = len({d["test_image_path"] for d in duplicates})
    print("\n--- [SUMMARY] DATA LEAKAGE SUMMARY ---")
    print(f"Total Test Samples          : {len(test_records)}")
    print(f"Leaked Test Images Count    : {unique_leaked_count}")
    print(f"Data Leakage Percentage     : {leakage_pct:.2f}%")
    print(f"CSV Report Saved To         : {csv_paths[0]}")

    if args.remove and duplicates:
        quarantine_dir = Path("data/artifacts/quarantine_leaked_test_images")
        moved = quarantine_leaked_images(duplicates, quarantine_dir)
        print(f"\n[ACTION] Moved {moved} leaked test images to quarantine folder: {quarantine_dir}")
    elif duplicates:
        print("\n[ACTION REQUIRED] Data leakage detected!")
        print("Run with `--remove` option to quarantine leaked images from test split:")
        print("  python utils/dedupe_check.py --remove")

    print("\n=======================================================")


if __name__ == "__main__":
    main()

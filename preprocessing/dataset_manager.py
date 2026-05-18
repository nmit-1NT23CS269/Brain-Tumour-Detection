"""Dataset extraction, validation, organization, and split management."""

from __future__ import annotations

import json
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from utils import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SPLIT_DATA_DIR,
    configure_logging,
    ensure_project_dirs,
)


LOGGER = configure_logging(__name__)
CLASS_ALIAS_MAP = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary",
    "notumor": "Normal",
    "normal": "Normal",
}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class DatasetSummary:
    """Summary of organized dataset contents."""

    zip_path: str
    extracted_to: str
    class_counts: dict[str, int]
    corrupted_files: list[str]
    splits: dict[str, int]


def extract_zip_dataset(zip_path: str | Path, extract_to: str | Path | None = None) -> Path:
    """Extract a zip archive into the raw dataset directory."""
    ensure_project_dirs()
    zip_path = Path(zip_path)
    extract_to = Path(extract_to) if extract_to else RAW_DATA_DIR / zip_path.stem
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_to)

    LOGGER.info("Extracted dataset archive to %s", extract_to)
    return extract_to


def detect_dataset_structure(dataset_root: str | Path) -> dict[str, dict[str, list[Path]]]:
    """Detect split/class structure from an extracted dataset directory."""
    dataset_root = Path(dataset_root)
    structure: dict[str, dict[str, list[Path]]] = {}

    for file_path in dataset_root.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        parts = [part.lower() for part in file_path.relative_to(dataset_root).parts]
        split = next((part for part in parts if part in {"training", "testing", "validation", "train", "test", "val"}), "unspecified")
        class_key = next((CLASS_ALIAS_MAP[part] for part in parts if part in CLASS_ALIAS_MAP), None)
        if class_key is None:
            continue

        normalized_split = {
            "training": "train",
            "train": "train",
            "validation": "val",
            "val": "val",
            "testing": "test",
            "test": "test",
        }.get(split, "train")

        structure.setdefault(normalized_split, {}).setdefault(class_key, []).append(file_path)

    return structure


def validate_dataset_images(dataset_root: str | Path) -> tuple[list[Path], list[Path]]:
    """Return valid and corrupted image paths under a dataset root."""
    dataset_root = Path(dataset_root)
    valid_files: list[Path] = []
    corrupted_files: list[Path] = []

    for file_path in dataset_root.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        try:
            with Image.open(file_path) as img:
                img.verify()
            valid_files.append(file_path)
        except (OSError, UnidentifiedImageError, ValueError):
            corrupted_files.append(file_path)

    LOGGER.info(
        "Validated dataset images: %s valid, %s corrupted",
        len(valid_files),
        len(corrupted_files),
    )
    return valid_files, corrupted_files


def organize_dataset(dataset_root: str | Path) -> dict[str, list[Path]]:
    """Copy validated images into a normalized processed dataset layout."""
    dataset_root = Path(dataset_root)
    valid_files, corrupted_files = validate_dataset_images(dataset_root)

    if PROCESSED_DATA_DIR.exists():
        for child in PROCESSED_DATA_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child)

    organized: dict[str, list[Path]] = {}
    for file_path in valid_files:
        parts = [part.lower() for part in file_path.relative_to(dataset_root).parts]
        class_name = next((CLASS_ALIAS_MAP[part] for part in parts if part in CLASS_ALIAS_MAP), None)
        if class_name is None:
            continue
        target_dir = PROCESSED_DATA_DIR / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / file_path.name
        if target_file.exists():
            target_file = target_dir / f"{file_path.stem}_{abs(hash(str(file_path))) % 10000}{file_path.suffix.lower()}"
        shutil.copy2(file_path, target_file)
        organized.setdefault(class_name, []).append(target_file)

    for corrupted_file in corrupted_files:
        LOGGER.warning("Corrupted image skipped: %s", corrupted_file)

    return organized


def create_dataset_splits(
    organized_files: dict[str, list[Path]],
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> dict[str, int]:
    """Create train/val/test directory splits from organized class folders."""
    ensure_project_dirs()
    if SPLIT_DATA_DIR.exists():
        for child in SPLIT_DATA_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child)

    split_counts: Counter[str] = Counter()

    for class_name, files in organized_files.items():
        train_files, temp_files = train_test_split(
            files,
            test_size=val_size + test_size,
            random_state=random_state,
            shuffle=True,
        )
        relative_test_size = test_size / (val_size + test_size)
        val_files, test_files = train_test_split(
            temp_files,
            test_size=relative_test_size,
            random_state=random_state,
            shuffle=True,
        )

        for split_name, split_files in {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }.items():
            target_dir = SPLIT_DATA_DIR / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for source_file in split_files:
                shutil.copy2(source_file, target_dir / source_file.name)
                split_counts[split_name] += 1

    return dict(split_counts)


def prepare_dataset_from_zip(zip_path: str | Path) -> DatasetSummary:
    """Extract, validate, organize, and split a dataset archive."""
    extracted_root = extract_zip_dataset(zip_path)
    organized = organize_dataset(extracted_root)
    split_counts = create_dataset_splits(organized)
    _, corrupted_files = validate_dataset_images(extracted_root)
    class_counts = {class_name: len(files) for class_name, files in organized.items()}

    summary = DatasetSummary(
        zip_path=str(zip_path),
        extracted_to=str(extracted_root),
        class_counts=class_counts,
        corrupted_files=[str(path) for path in corrupted_files],
        splits=split_counts,
    )

    metadata_path = PROCESSED_DATA_DIR / "dataset_summary.json"
    metadata_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    LOGGER.info("Dataset preparation summary saved to %s", metadata_path)
    return summary

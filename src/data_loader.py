"""
data_loader.py
--------------
Load image paths and labels from folder structure:
    data/train/Normal, data/train/Disease
    data/test/Normal, data/test/Disease

If folders are empty, optionally create a small synthetic dataset so the
pipeline can run end-to-end for learning and demos.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Project root: AI-Medical-Image-Analysis/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Standard class folder names (binary classification)
CLASS_NORMAL = "Normal"
CLASS_DISEASE = "Disease"
CLASS_NAMES = [CLASS_NORMAL, CLASS_DISEASE]

# Image extensions we accept
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def get_project_root() -> Path:
    """Return absolute path to project root."""
    return PROJECT_ROOT


def _collect_images_from_class_folder(class_dir: Path) -> List[Path]:
    """Return sorted list of image file paths under class_dir."""
    if not class_dir.is_dir():
        return []
    paths: List[Path] = []
    for entry in sorted(class_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() in VALID_EXTENSIONS:
            paths.append(entry)
    return paths


def load_image_paths(split: str) -> Tuple[List[Path], List[int]]:
    """
    Load all image paths and integer labels for 'train' or 'test'.

    Label 0 = Normal, 1 = Disease.

    Args:
        split: 'train' or 'test'

    Returns:
        (paths, labels) parallel lists
    """
    base = PROJECT_ROOT / "data" / split
    paths: List[Path] = []
    labels: List[int] = []

    for label, folder in enumerate(CLASS_NAMES):
        class_dir = base / folder
        for p in _collect_images_from_class_folder(class_dir):
            paths.append(p)
            labels.append(label)

    return paths, labels


def count_images_in_split(split: str) -> int:
    """Return total number of images in train or test split."""
    paths, _ = load_image_paths(split)
    return len(paths)


def create_synthetic_dataset(
    train_per_class: int = 40,
    test_per_class: int = 10,
    size: Tuple[int, int] = (256, 256),
    seed: int = 42,
) -> None:
    """
    Create simple synthetic grayscale images for each class so the pipeline
    can train without real medical data (educational / smoke test only).

    Normal: smoother random patches
    Disease: higher local variance (still fake — not for clinical use)
    """
    rng = np.random.default_rng(seed)
    data_root = PROJECT_ROOT / "data"

    specs = [
        ("train", train_per_class),
        ("test", test_per_class),
    ]
    for split, n_per in specs:
        if n_per <= 0:
            continue
        for label_idx, name in enumerate(CLASS_NAMES):
            out_dir = data_root / split / CLASS_NAMES[label_idx]
            out_dir.mkdir(parents=True, exist_ok=True)
            h, w = size
            for i in range(n_per):
                if name == CLASS_NORMAL:
                    base = rng.normal(120, 15, (h, w)).astype(np.float32)
                    img = np.clip(base, 0, 255).astype(np.uint8)
                else:
                    base = rng.normal(100, 35, (h, w)).astype(np.float32)
                    noise = rng.normal(0, 25, (h, w)).astype(np.float32)
                    img = np.clip(base + noise, 0, 255).astype(np.uint8)

                fname = out_dir / f"synth_{name.lower()}_{i:04d}.png"
                cv2.imwrite(str(fname), img)


def ensure_data_or_create_synthetic() -> None:
    """
    If train split has no images, create synthetic train+test data.
    If train has images but test is empty, create synthetic test data only.
    """
    n_train = count_images_in_split("train")
    n_test = count_images_in_split("test")

    if n_train == 0:
        print(
            "[data_loader] No training images found. "
            "Creating a small SYNTHETIC dataset for demo/training.\n"
            "          Replace data/train and data/test with real Normal/Disease images."
        )
        create_synthetic_dataset(train_per_class=40, test_per_class=10)
        return

    if n_test == 0:
        print(
            "[data_loader] Test split empty; creating synthetic test images "
            "(add real test data for proper evaluation)."
        )
        create_synthetic_dataset(train_per_class=0, test_per_class=10)

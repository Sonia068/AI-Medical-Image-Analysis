"""
main.py
-------
End-to-end pipeline:
  1. Ensure dataset exists (or create small synthetic demo data).
  2. Train CNN with augmentation.
  3. Evaluate on test set and save plots.
  4. Copy one training image to images/sample.png for previews.

Run:
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when running from project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_loader import (  # noqa: E402
    count_images_in_split,
    ensure_data_or_create_synthetic,
    load_image_paths,
)
from evaluate import evaluate_model, save_sample_preview  # noqa: E402
from train import train_model  # noqa: E402


def main() -> None:
    print("=== AI Medical Image Analysis — training pipeline ===\n")

    # 1) Data
    ensure_data_or_create_synthetic()
    n_train = count_images_in_split("train")
    n_test = count_images_in_split("test")
    print(f"Training images: {n_train} | Test images: {n_test}\n")

    # 2) Train (epochs tuned for demo; increase for real datasets)
    print("--- Training CNN ---")
    train_model(epochs=25, batch_size=16, validation_split=0.2, learning_rate=1e-4)

    # 3) Evaluate + plots
    print("\n--- Evaluation ---")
    results = evaluate_model()
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print(f"Test images: {results['n_test_images']}")
    print("\nClassification report:\n", results["classification_report"])

    # 4) Sample image for docs / UI
    paths, _ = load_image_paths("train")
    sample_dest = ROOT / "images" / "sample.png"
    if paths:
        save_sample_preview(paths[0], sample_dest)
        print(f"\nSample preview saved to: {sample_dest}")

    print("\nDone. Model: models/cnn_model.h5 | Plots: outputs/")
    print("API: python api/app.py  |  Dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()

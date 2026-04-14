"""
evaluate.py
-----------
Evaluate saved model on data/test: accuracy, confusion matrix, plots.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow import keras

from data_loader import get_project_root
from preprocessing import flow_from_directory_eval


def plot_training_history(history_path: Path, out_dir: Path) -> None:
    """Load joblib history and save accuracy / loss curves."""
    if not history_path.is_file():
        return

    history = joblib.load(history_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    epochs_range = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train accuracy")
    if val_acc:
        plt.plot(epochs_range, val_acc, label="Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train loss")
    if val_loss:
        plt.plot(epochs_range, val_loss, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_loss_plots.png", dpi=150)
    plt.close()

    # Also save separate files if downstream expects these exact names
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, acc, label="Train")
    if val_acc:
        plt.plot(epochs_range, val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Model accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_plot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, loss, label="Train")
    if val_loss:
        plt.plot(epochs_range, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_plot.png", dpi=150)
    plt.close()


def evaluate_model(model_path: Path | None = None) -> dict:
    """
    Run evaluation on test set, save confusion matrix plot.

    Returns:
        dict with accuracy, confusion_matrix, report string
    """
    root = get_project_root()
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_path is None:
        model_path = root / "models" / "cnn_model.h5"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}. Train first (python main.py).")

    model = keras.models.load_model(str(model_path))
    test_dir = root / "data" / "test"

    test_gen = flow_from_directory_eval(test_dir, batch_size=16, seed=42)
    test_gen.reset()

    preds_proba = model.predict(test_gen, verbose=1)
    y_pred = (preds_proba.ravel() >= 0.5).astype(int)
    y_true = test_gen.classes

    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Disease"],
        digits=4,
    )

    # Confusion matrix heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Disease"],
        yticklabels=["Normal", "Disease"],
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Training curves from last run
    plot_training_history(root / "outputs" / "train_history.joblib", out_dir)

    results = {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_test_images": int(test_gen.samples),
    }
    joblib.dump(results, out_dir / "eval_results.joblib")

    return results


def save_sample_preview(image_path: Path, dest: Path) -> None:
    """Copy or reference a sample image for README / dashboard (simple file copy via read/write)."""
    import shutil

    dest.parent.mkdir(parents=True, exist_ok=True)
    if image_path.is_file():
        shutil.copy(image_path, dest)

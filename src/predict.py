"""
predict.py
----------
Single-image inference for the trained MobileNetV2 model.
Used by dashboard/app.py and api/app.py.
"""
 
from __future__ import annotations
from pathlib import Path
 
import numpy as np
from tensorflow import keras
 
 
# Must match what was used during training
IMG_SIZE = (128, 128)   # change to (224, 224) if you retrain with 224
 
 
def predict_image(model: keras.Model, image_path: str | Path) -> dict:
    """
    Run inference on a single image file.
 
    Returns:
        {
            "label":      "Normal" or "Disease",
            "confidence": float (0.0 – 1.0),   # confidence of predicted class
            "prob_disease": float,              # raw sigmoid output
            "prob_normal":  float,
        }
    """
    from PIL import Image
 
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
 
    # ── Load image as RGB (3 channels) ──────────────────────
    # This is the critical fix — MobileNetV2 needs 3 channels
    img = Image.open(image_path).convert("RGB")   # ← always RGB, never grayscale
    img = img.resize(IMG_SIZE)                     # resize to match training size
 
    # ── Normalize to 0–1 ────────────────────────────────────
    arr = np.array(img, dtype=np.float32) / 255.0  # shape: (H, W, 3)
 
    # ── Add batch dimension ──────────────────────────────────
    batch = np.expand_dims(arr, axis=0)             # shape: (1, H, W, 3)
 
    # ── Predict ──────────────────────────────────────────────
    prob_disease = float(model.predict(batch, verbose=0).ravel()[0])
    prob_normal  = 1.0 - prob_disease
 
    # ── Threshold at 0.5 ─────────────────────────────────────
    if prob_disease >= 0.5:
        label      = "Disease"
        confidence = prob_disease
    else:
        label      = "Normal"
        confidence = prob_normal
 
    return {
        "label":        label,
        "confidence":   confidence,
        "prob_disease": prob_disease,
        "prob_normal":  prob_normal,
    }


def load_trained_model(model_path):
    from tensorflow import keras
    return keras.models.load_model(model_path)
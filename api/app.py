"""
Flask API for medical image classification.

Endpoint:
    POST /predict  — multipart form field "image" (file upload)

Returns JSON: label, confidence, p_normal, p_disease
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request

# Project root (parent of api/)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict import load_trained_model, predict_image  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# Load model once
_model = None


def get_model():
    global _model
    if _model is None:
        model_path = ROOT / "models" / "cnn_model.h5"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")

        _model = load_trained_model(model_path)

    return _model


ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "AI-Medical-Image-Analysis"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part 'image' in request."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    ext = Path(file.filename).suffix.lower()

    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported extension {ext}"}), 400

    try:
        model = get_model()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        result = predict_image(model, tmp_path)

        return jsonify(
            {
                "label": result["label"],
                "confidence": round(result["confidence"], 4),
                "p_normal": round(result["prob_normal"], 4),
                "p_disease": round(result["prob_disease"], 4),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            os.remove(tmp_path)
        except:
            pass


if __name__ == "__main__":
    print("🚀 Starting Flask API at http://127.0.0.1:5000")
    print("👉 Use POST /predict with form-data key = 'image'")
    app.run(host="0.0.0.0", port=5000, debug=True)
"""
train.py
--------
Train the CNN with augmented data, save weights to models/cnn_model.h5,
and persist training metadata with joblib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from data_loader import get_project_root
from model import build_model
from preprocessing import make_train_datagen, flow_from_directory_train


def train_model(
    epochs: int = 20,
    batch_size: int = 16,
    validation_split: float = 0.2,
    learning_rate: float = 1e-4,
) -> keras.callbacks.History:
    """
    Train CNN on data/train with augmentation and validation split.

    Saves:
        models/cnn_model.h5
        outputs/train_history.joblib
    """

    root = get_project_root()
    train_dir = root / "data" / "train"
    models_dir = root / "models"
    outputs_dir = root / "outputs"

    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    weights_path = models_dir / "cnn_model.h5"

    #  Data Generators
    datagen = make_train_datagen(validation_split=validation_split, seed=42)

    train_gen = flow_from_directory_train(
        train_dir,
        datagen,
        batch_size=batch_size,
        subset="training",
        seed=42,
    )
    print("Class Indices:", train_gen.class_indices)


    val_gen = flow_from_directory_train(
        train_dir,
        datagen,
        batch_size=batch_size,
        subset="validation",
        seed=42,
    )

    #  Compute class weights (IMPORTANT FIX)
    class_labels = train_gen.classes
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(class_labels),
        y=class_labels
    )

    class_weight_dict = dict(enumerate(class_weights))

    print(" Class Weights:", class_weight_dict)

    #  Steps per epoch
    steps_train = max(1, train_gen.samples // batch_size)
    steps_val = max(1, val_gen.samples // batch_size)

    #  Build model
    model = build_model()

    #  Callbacks (already strong)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(weights_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    #  TRAINING (UPDATED)
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_train,
        validation_data=val_gen,
        validation_steps=steps_val,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,   #  FIX ADDED
        verbose=1,
    )

    #  Save model
    model.save(str(weights_path))

    #  Save metadata
    meta: Dict[str, Any] = {
        "class_indices": train_gen.class_indices,
        "epochs_trained": len(history.history.get("loss", [])),
        "train_samples": int(train_gen.samples),
        "val_samples": int(val_gen.samples),
        "batch_size": batch_size,
        "class_weights": class_weight_dict,
        "label_meaning": "sigmoid output = probability of Disease (class 1); "
        "0 = Normal, 1 = Disease",
    }

    joblib.dump(meta, outputs_dir / "train_meta.joblib")
    joblib.dump(history.history, outputs_dir / "train_history.joblib")

    print(" Training completed and model saved!")

    return history
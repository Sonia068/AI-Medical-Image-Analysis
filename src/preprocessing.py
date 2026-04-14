"""
preprocessing.py
----------------
Image preprocessing: resize 256x256, grayscale, normalize to [0, 1].
Includes strong augmentation for better generalization.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

# 🔹 Reduced size (better for small dataset)
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3  # rgb


def read_and_preprocess_image(image_path: str | Path) -> np.ndarray:
    """
    Load one image, grayscale, resize, normalize.
    """
    path = Path(image_path)

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0

    # Add channel dimension
    img = np.expand_dims(img, axis=-1)

    return img


def make_train_datagen(validation_split: float | None = 0.2, seed: int = 42):
    """
    Strong augmentation → prevents overfitting
    """
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,

        #  STRONG AUGMENTATION
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],

        fill_mode="nearest",
        validation_split=validation_split if validation_split else 0.0,
    )


def make_eval_datagen():
    """Only rescaling for test data"""
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0
    )


def flow_from_directory_train(
    train_dir: str | Path,
    datagen: tf.keras.preprocessing.image.ImageDataGenerator,
    batch_size: int = 8,   # 🔹 smaller batch = better learning
    subset: str = "training",
    seed: int = 42,
):
    """
    Training / validation generator
    """
    return datagen.flow_from_directory(
        directory=str(train_dir),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode="rgb",
        class_mode="binary",
        classes=["Normal", "Disease"],
        batch_size=batch_size,
        subset=subset,
        seed=seed,
        shuffle=True,
    )


def flow_from_directory_eval(
    test_dir: str | Path,
    batch_size: int = 8,
    seed: int = 42,
):
    """Evaluation generator"""
    datagen = make_eval_datagen()

    return datagen.flow_from_directory(
        directory=str(test_dir),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode="rgb",
        class_mode="binary",
        classes=["Normal", "Disease"],
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
    )
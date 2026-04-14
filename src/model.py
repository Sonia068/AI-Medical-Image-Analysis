"""
model.py (TRANSFER LEARNING)
---------------------------
Uses MobileNetV2 pretrained model for better performance on small datasets.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_model():
    base_model = keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze base model
    base_model.trainable = False

    inputs = keras.Input(shape=(128, 128, 3))

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
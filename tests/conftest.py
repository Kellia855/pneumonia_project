"""Test setup: a temporary project folder with tiny synthetic data and a tiny model, so the tests run
without the real dataset or trained model. Environment variables must be set before src is imported."""
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(tempfile.mkdtemp(prefix="pneu_test_"))
os.environ["DATA_DIR"] = str(ROOT / "data")
os.environ["MODELS_DIR"] = str(ROOT / "models")
os.environ["MONGO_URI"] = "mongomock://localhost"
os.environ["MONGO_DB"] = "pneumonia_test"
os.environ["RETRAIN_MIN_NEW_IMAGES"] = "10"
os.environ["RETRAIN_EPOCHS"] = "1"
os.environ["BATCH_SIZE"] = "8"

rng = np.random.default_rng(0)


def make_image_bytes(mean, size=(120, 140), fmt="JPEG"):
    arr = np.clip(rng.normal(mean, 30, size), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def _fill(split, n_per_class):
    for label, mean in (("NORMAL", 100), ("PNEUMONIA", 150)):
        folder = ROOT / "data" / split / label
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            (folder / f"{i}.jpg").write_bytes(make_image_bytes(mean))


for split, n in (("train", 12), ("val", 8), ("test", 8)):
    _fill(split, n)

import tensorflow as tf  # noqa: E402

(ROOT / "models").mkdir(parents=True, exist_ok=True)
L = tf.keras.layers
inputs = tf.keras.Input(shape=(224, 224, 3))
x = L.Rescaling(1 / 255)(inputs)
x = L.Conv2D(4, 3, strides=4, activation="relu")(x)
x = L.GlobalAveragePooling2D()(x)
outputs = L.Dense(1, activation="sigmoid")(x)
tf.keras.Model(inputs, outputs).save(ROOT / "models" / "pneumonia_model.h5")
(ROOT / "models" / "model_metadata.json").write_text(json.dumps(
    {"model_name": "dummy", "version": 1, "classes": ["NORMAL", "PNEUMONIA"], "img_size": [224, 224],
     "threshold": 0.5, "trained_at": "2026-01-01T00:00:00", "n_train_images": 24, "test_metrics": {}}))

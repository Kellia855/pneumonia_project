"""Single-image prediction. Uses the same preprocessing as training (src/preprocessing.py)."""
import json
import os
import threading
import time
from pathlib import Path

import tensorflow as tf

from . import config
from .preprocessing import preprocess_image_bytes

# Match TensorFlow's thread pools to the CPUs this container may use (TF_NUM_THREADS, set in docker-compose).
# By default TF sizes them to the whole machine, which thrashes when Docker limits the container's CPUs.
_tf_threads = int(os.getenv("TF_NUM_THREADS", "0") or 0)
if _tf_threads > 0:
    try:
        tf.config.threading.set_intra_op_parallelism_threads(_tf_threads)
        tf.config.threading.set_inter_op_parallelism_threads(_tf_threads)
    except RuntimeError:          # TensorFlow already started (e.g. imported elsewhere first): keep defaults
        pass

# One model call at a time per container: concurrent requests queue here and are served in turn,
# instead of all sharing the CPU and finishing very late together.
_infer_lock = threading.Lock()


class ModelStore:
    """Holds the deployed model in memory. reload() swaps in a newly promoted model."""

    def __init__(self):
        self._model = None
        self._meta = {}
        self._lock = threading.Lock()
        self._last_check = 0.0
        self.check_interval = 5.0        # seconds between looks at model_metadata.json

    @property
    def loaded(self):
        return self._model is not None

    def load(self):
        model = tf.keras.models.load_model(config.MODEL_PATH)
        meta = json.loads(config.META_PATH.read_text()) if config.META_PATH.exists() else {}
        with self._lock:
            self._model, self._meta = model, meta
            self._last_check = time.time()

    reload = load

    def _disk_version(self):
        try:
            return json.loads(config.META_PATH.read_text()).get("version")
        except Exception:
            return None

    def get(self):
        """Return (model, metadata). If another container promoted a newer model (the shared
        models/ folder shows a higher version), load it first."""
        if self._model is None:
            self.load()
        elif time.time() - self._last_check > self.check_interval:
            self._last_check = time.time()
            version = self._disk_version()
            if version is not None and version != self._meta.get("version"):
                self.load()
        return self._model, self._meta


store = ModelStore()


def predict_bytes(data: bytes, threshold=None):
    """Raw image bytes -> prediction dict."""
    model, meta = store.get()
    threshold = meta.get("threshold", 0.5) if threshold is None else threshold
    x = tf.expand_dims(preprocess_image_bytes(data), 0)
    with _infer_lock:
        p_pneumonia = float(model(x, training=False).numpy()[0][0])
    label = "PNEUMONIA" if p_pneumonia >= threshold else "NORMAL"
    confidence = p_pneumonia if label == "PNEUMONIA" else 1 - p_pneumonia
    return {
        "label": label,
        "probability_pneumonia": round(p_pneumonia, 4),
        "confidence": round(confidence, 4),
        "threshold": round(float(threshold), 4),
        "model_version": meta.get("version"),
    }


def predict_path(path):
    return predict_bytes(Path(path).read_bytes())


if __name__ == "__main__":   # python -m src.prediction path/to/image.jpg
    import sys
    print(predict_path(sys.argv[1]))
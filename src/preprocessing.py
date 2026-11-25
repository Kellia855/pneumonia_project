"""Image preprocessing. These steps match the notebook (decode to RGB, bilinear resize to 224x224;
pixel scaling happens inside the model) so training and serving stay consistent."""
import hashlib
import io
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from . import config


# ----------------------------------------------------------------------------- single images
def file_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def colour_fraction(im: Image.Image) -> float:
    """Share of pixels (0..1) that are clearly coloured. Chest X-rays are greyscale, so this is close to 0.
    Measured on a small copy of the image, so it is fast even for large files."""
    small = im.convert("RGB")
    small.thumbnail((128, 128))
    arr = np.asarray(small, dtype=np.int16)
    spread = arr.max(axis=2) - arr.min(axis=2)          # 0 for grey pixels, high for saturated colours
    return float((spread > config.COLOUR_PIXEL_THRESHOLD).mean())


def validate_image(data: bytes):
    """Check that bytes are a usable image. Returns (ok, reason, (width, height) or None)."""
    if not data:
        return False, "empty file", None
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        return False, f"file larger than {config.MAX_UPLOAD_MB} MB", None
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
            fmt = im.format
            colour = colour_fraction(im) if config.COLOUR_CHECK else 0.0
    except Exception:
        return False, "unreadable or corrupt image", None
    if fmt not in ("JPEG", "PNG"):
        return False, f"unsupported format ({fmt}); use JPG or PNG", None
    if min(width, height) < config.MIN_IMAGE_SIDE:
        return False, f"image too small (minimum {config.MIN_IMAGE_SIDE}px per side)", (width, height)
    if colour > config.MAX_COLOUR_FRACTION:
        return (False, f"does not look like a chest X-ray: {colour:.0%} of the image is in colour, "
                       "while X-rays are greyscale", (width, height))
    return True, "ok", (width, height)


def preprocess_image_bytes(data: bytes) -> tf.Tensor:
    """Bytes -> float32 tensor (224, 224, 3) in 0..255. Same decode + resize as training."""
    img = tf.io.decode_image(data, channels=3, expand_animations=False)
    img = tf.image.resize(img, config.IMG_SIZE, method="bilinear")
    return tf.cast(img, tf.float32)


def standardize_for_storage(data: bytes) -> bytes:
    """Preprocess an uploaded image and encode it as a 224x224 RGB JPEG for the training folder."""
    img = preprocess_image_bytes(data)
    img = tf.cast(tf.round(tf.clip_by_value(img, 0, 255)), tf.uint8)
    return tf.io.encode_jpeg(img, quality=95).numpy()


# ----------------------------------------------------------------------------- bulk uploads
def process_upload(items, known_hashes):
    """Preprocess a batch of uploaded files.

    items: list of (filename, raw_bytes); known_hashes: set of MD5s already stored.
    Returns (accepted, rejected). Each accepted item holds the standardized JPEG bytes."""
    accepted, rejected = [], []
    seen = set(known_hashes)
    for filename, data in items:
        ok, reason, size = validate_image(data)
        if not ok:
            rejected.append({"filename": filename, "reason": reason})
            continue
        md5 = file_md5(data)
        if md5 in seen:
            rejected.append({"filename": filename, "reason": "duplicate of an image already stored"})
            continue
        try:
            jpeg = standardize_for_storage(data)
        except Exception:
            rejected.append({"filename": filename, "reason": "could not be preprocessed"})
            continue
        seen.add(md5)
        accepted.append({"filename": filename, "md5": md5, "width": size[0], "height": size[1],
                         "jpeg": jpeg})
    return accepted, rejected


# ----------------------------------------------------------------------------- datasets
def count_images(folder) -> int:
    folder = Path(folder)
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in config.VALID_EXT)


def class_counts(folder):
    return {c: count_images(Path(folder) / c) for c in config.CLASSES}


def load_directory_dataset(directory, shuffle=False, batch_size=None):
    """Folder with NORMAL/ and PNEUMONIA/ subfolders -> batched tf.data dataset (0..255 pixels)."""
    directory = Path(directory)
    for c in config.CLASSES:
        (directory / c).mkdir(parents=True, exist_ok=True)
    return tf.keras.utils.image_dataset_from_directory(
        directory, labels="inferred", label_mode="binary", class_names=config.CLASSES,
        color_mode="rgb", image_size=config.IMG_SIZE, batch_size=batch_size or config.BATCH_SIZE,
        shuffle=shuffle, seed=config.SEED)


def get_labels(ds):
    return np.concatenate([y.numpy().ravel() for _, y in ds]).astype(int)


def build_augmentation():
    L = tf.keras.layers
    return tf.keras.Sequential([
        L.RandomRotation(0.05), L.RandomZoom(0.10), L.RandomTranslation(0.05, 0.05),
        L.RandomBrightness(0.10, value_range=(0, 255)), L.RandomContrast(0.10),
    ], name="data_augmentation")


def augment_ds(ds):
    """Augmentation lives in the data pipeline, not in the saved model (see notebook section 8.1)."""
    aug = build_augmentation()
    return ds.map(lambda x, y: (aug(x, training=True), y),
                  num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def sample_brightness(directory, per_class=60):
    """Mean pixel intensity per class on a small sample (for the dashboard's data insights)."""
    rng = np.random.default_rng(config.SEED)
    out = {}
    for c in config.CLASSES:
        folder = Path(directory) / c
        files = [p for p in folder.glob("*") if p.suffix.lower() in config.VALID_EXT] if folder.exists() else []
        if not files:
            out[c] = None
            continue
        pick = rng.choice(len(files), size=min(per_class, len(files)), replace=False)
        means = []
        for i in pick:
            with Image.open(files[i]) as im:
                means.append(float(np.asarray(im.convert("L").resize((112, 112)), dtype=np.float32).mean()))
        out[c] = {"n": len(means), "mean": round(float(np.mean(means)), 2),
                  "std": round(float(np.std(means)), 2), "values": [round(m, 1) for m in means]}
    return out
"""Model building, training, evaluation and retraining.

`retrain_model` continues training the currently deployed model (it is the pre-trained model) on the
original replay data plus everything users uploaded, then promotes the result only if it is not worse."""
import datetime
import json
import os
import shutil

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from . import config
from .preprocessing import (augment_ds, class_counts, count_images, get_labels,
                            load_directory_dataset)

L = tf.keras.layers


# ----------------------------------------------------------------------------- building / training
def get_metrics():
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]


def build_mobilenet():
    """MobileNetV2 transfer-learning model, same architecture as the notebook (section 9.2)."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=config.IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = tf.keras.Input(shape=config.IMG_SIZE + (3,))
    x = L.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    x = base(x, training=False)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.3)(x)
    outputs = L.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenetv2_transfer")


def set_finetune_policy(model, n_last=30):
    """Unfreeze the last n_last layers of the pretrained base; keep BatchNorm layers frozen."""
    bases = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)]
    if not bases:
        return model
    base = bases[0]
    base.trainable = True
    for layer in base.layers[:-n_last]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, L.BatchNormalization):
            layer.trainable = False
    return model


def get_callbacks(patience=3):
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
    ]


def train_model(model, train_ds, val_ds, class_weight=None, epochs=10, lr=1e-3, extra_callbacks=None):
    """Compile and fit (used for first-time training; retraining uses retrain_model)."""
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=get_metrics())
    return model.fit(augment_ds(train_ds), validation_data=val_ds, epochs=epochs,
                     class_weight=class_weight, callbacks=get_callbacks() + (extra_callbacks or []))


# ----------------------------------------------------------------------------- metrics / metadata
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(set(np.asarray(y_true).tolist())) > 1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both else None,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def predict_probs(model, ds):
    return model.predict(ds, verbose=0).ravel()


def evaluate_directory(model, directory, threshold, max_images=None):
    """Score a model on a folder of labelled images (used by the /evaluate endpoint)."""
    ds = load_directory_dataset(directory, shuffle=False)
    if max_images:
        ds = ds.take(int(np.ceil(max_images / config.BATCH_SIZE)))
    y_true = get_labels(ds)
    metrics = compute_metrics(y_true, predict_probs(model, ds), threshold)
    metrics["n_images"] = int(len(y_true))
    return metrics


def load_metadata():
    return json.loads(config.META_PATH.read_text())


def save_metadata(meta):
    tmp = config.META_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, config.META_PATH)             # atomic, so readers never see a half-written file


# ----------------------------------------------------------------------------- retraining
def retrain_model(epochs=None, lr=None, progress=None, max_f1_drop=0.01, max_recall_drop=0.02):
    """Fine-tune the deployed model on replay data + uploads; promote it only if it is not worse.

    1. Load the deployed model (the custom pre-trained model) and score it on the validation set.
    2. Combine data/train (replay) with data/uploads, recompute class weights, fine-tune at a low rate.
    3. Score the retrained model on the same validation set.
    4. Promote if val F1 dropped by at most `max_f1_drop` and recall by at most `max_recall_drop`.
    """
    say = progress or (lambda message: None)
    epochs = epochs or config.RETRAIN_EPOCHS
    lr = lr or config.RETRAIN_LR

    meta = load_metadata()
    threshold, old_version = meta["threshold"], meta["version"]

    n_new = count_images(config.UPLOAD_DIR)
    if n_new == 0:
        raise ValueError("There is no uploaded data to retrain on.")
    if count_images(config.VAL_DIR) == 0:
        raise ValueError("The validation folder (data/val) is empty, so a retrained model cannot be checked.")

    say("Loading the deployed model")
    model = tf.keras.models.load_model(config.MODEL_PATH)

    say("Scoring the current model on the validation set")
    val_ds = load_directory_dataset(config.VAL_DIR, shuffle=False)
    y_val = get_labels(val_ds)
    old_val = compute_metrics(y_val, predict_probs(model, val_ds), threshold)

    say("Preparing training data (replay data + uploads)")
    parts = []
    if count_images(config.TRAIN_DIR) > 0:
        parts.append(load_directory_dataset(config.TRAIN_DIR, shuffle=True).unbatch())
    parts.append(load_directory_dataset(config.UPLOAD_DIR, shuffle=True).unbatch())
    combined = parts[0]
    for extra in parts[1:]:
        combined = combined.concatenate(extra)
    combined = augment_ds(combined.shuffle(2000, seed=config.SEED).batch(config.BATCH_SIZE))

    train_counts, upload_counts = class_counts(config.TRAIN_DIR), class_counts(config.UPLOAD_DIR)
    counts = np.array([train_counts[c] + upload_counts[c] for c in config.CLASSES], dtype=float)
    class_weight = ({i: float(counts.sum() / (2 * counts[i])) for i in range(2)}
                    if (counts > 0).all() else {0: 1.0, 1: 1.0})

    set_finetune_policy(model)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=get_metrics())

    def on_epoch_end(epoch, logs):
        say(f"Epoch {epoch + 1}/{epochs}: val_loss={logs.get('val_loss', 0):.3f}, "
            f"val_recall={logs.get('val_recall', 0):.3f}")

    say(f"Fine-tuning for up to {epochs} epochs")
    model.fit(combined, validation_data=val_ds, epochs=epochs, class_weight=class_weight, verbose=0,
              callbacks=get_callbacks(patience=2) + [tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)])

    say("Scoring the retrained model")
    new_val = compute_metrics(y_val, predict_probs(model, val_ds), threshold)
    promoted = bool(new_val["f1"] >= old_val["f1"] - max_f1_drop and
                    new_val["recall"] >= old_val["recall"] - max_recall_drop)

    new_version = old_version + 1
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    archive = config.MODELS_DIR / f"pneumonia_model_v{old_version}.h5"
    if not archive.exists():
        shutil.copy2(config.MODEL_PATH, archive)          # keep the model being replaced
    candidate = config.MODELS_DIR / f"pneumonia_model_v{new_version}.h5"
    model.save(candidate)

    if promoted:
        tmp = config.MODEL_PATH.with_suffix(".tmp")
        shutil.copy2(candidate, tmp)
        os.replace(tmp, config.MODEL_PATH)                # atomic swap of the deployed file
        meta.update({
            "version": new_version,
            "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "n_train_images": int(counts.sum()),
            "val_metrics_at_retrain": {k: round(v, 4) for k, v in new_val.items()
                                       if k in ("accuracy", "precision", "recall", "f1", "roc_auc") and v is not None},
        })
        save_metadata(meta)
        say(f"New model promoted: version {new_version}")
    else:
        say("New model was not better than the current one, so the current model was kept")

    return {"old_version": old_version, "new_version": new_version, "promoted": promoted,
            "n_uploads": int(n_new), "n_replay": int(sum(train_counts.values())),
            "class_weights": class_weight, "old_val": old_val, "new_val": new_val}

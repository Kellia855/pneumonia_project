"""FastAPI service for the pneumonia classifier.

Run:  uvicorn app.api:app --host 0.0.0.0 --port 8000      (interactive docs at /docs)
"""
import socket
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src import config, database as db, model as model_lib, preprocessing as prep, prediction

STARTED_AT = time.time()
STARTED_AT_ISO = datetime.now(timezone.utc).isoformat(timespec="seconds")

_insights_cache = {}


@asynccontextmanager
async def lifespan(app):
    db.init_db()
    for split in (config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR, config.UPLOAD_DIR):
        for c in config.CLASSES:
            (split / c).mkdir(parents=True, exist_ok=True)
    try:
        prediction.store.load()
    except Exception as exc:                      # the API still starts; /health reports the problem
        print(f"WARNING: could not load the model: {exc}")
    yield


app = FastAPI(title="Pneumonia X-ray Classifier API",
              description="Predict, upload new data, retrain and monitor a chest X-ray pneumonia model. "
                          "Educational project, not a medical device.",
              version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ----------------------------------------------------------------------------- helpers
def retrain_trigger():
    """Decides whether retraining is needed (documented in the README)."""
    n_new = db.count_uploads("new")
    reasons = []
    if n_new >= config.RETRAIN_MIN_NEW_IMAGES:
        reasons.append(f"{n_new} new images uploaded (threshold {config.RETRAIN_MIN_NEW_IMAGES})")
    return {"retrain": bool(reasons), "n_new_images": n_new,
            "threshold": config.RETRAIN_MIN_NEW_IMAGES, "reasons": reasons}


def _start_retrain(reason, epochs=None):
    """Start retraining in a background thread.

    The lock and progress live in MongoDB, so with several API containers only one retrains at a time
    and every container reports the same status. Returns run_id, or None if a run is already active."""
    if not db.acquire_retrain_lock():
        return None
    try:
        meta = prediction.store.get()[1] if prediction.store.loaded else model_lib.load_metadata()
        run_id = db.create_run(reason, db.count_uploads("new"), meta.get("version"))
        db.set_retrain_state(run_id=run_id)
    except Exception:
        db.release_retrain_lock("Failed to start")
        raise

    def progress(message):
        db.set_retrain_state(message=message)

    def work():
        final_message = "idle"
        try:
            result = model_lib.retrain_model(epochs=epochs, progress=progress)
            if result["promoted"]:
                prediction.store.reload()
                db.mark_uploads_used(result["new_version"])
            final_message = (f"Promoted v{result['new_version']}" if result["promoted"]
                             else "Retrained model was not better; kept the current model")
            db.finish_run(run_id, "done", final_message, result)
        except Exception as exc:
            final_message = f"Failed: {exc}"
            db.finish_run(run_id, "failed", str(exc))
        finally:
            db.release_retrain_lock(final_message)

    threading.Thread(target=work, daemon=True).start()
    return run_id


def _require_model():
    if not prediction.store.loaded:
        try:
            prediction.store.load()
        except Exception:
            raise HTTPException(503, "Model is not available. Put pneumonia_model.h5 in the models/ folder.")


# ----------------------------------------------------------------------------- endpoints
@app.get("/health", tags=["monitoring"])
async def health():
    """Up-time and model status (shown on the dashboard). Async so it is answered straight away, even
    while prediction requests are queued in the worker threads."""
    meta = prediction.store.get()[1] if prediction.store.loaded else {}
    database_ok = db.ping()
    return {
        "status": "ok" if (prediction.store.loaded and database_ok) else "degraded",
        "model_loaded": prediction.store.loaded,
        "database": "connected" if database_ok else "unreachable",
        "model_version": meta.get("version"),
        "model_name": meta.get("model_name"),
        "container": socket.gethostname(),       # shows which replica answered (load balancing)
        "started_at": STARTED_AT_ISO,
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
        "retraining": db.get_retrain_state()["running"] if database_ok else None,
        "predictions_served": db.prediction_summary()["total"] if database_ok else None,
    }


@app.post("/predict", tags=["prediction"])
def predict(file: UploadFile = File(..., description="A chest X-ray image (JPG or PNG)")):
    """Predict NORMAL or PNEUMONIA for one image."""
    _require_model()
    data = file.file.read()
    ok, reason, _ = prep.validate_image(data)
    if not ok:
        raise HTTPException(400, f"Invalid image: {reason}")
    result = prediction.predict_bytes(data)
    db.insert_prediction(file.filename, result["label"], result["probability_pneumonia"],
                         result["confidence"], result["model_version"])
    return {"filename": file.filename, **result}


@app.post("/upload", tags=["retraining"])
def upload(label: str = Form(..., description="NORMAL or PNEUMONIA: the true class of ALL files in this upload"),
           files: list[UploadFile] = File(..., description="Many images of the same class")):
    """Bulk upload of labelled images for retraining.

    Pipeline: validate -> remove duplicates -> convert to RGB -> resize to 224x224 -> save to disk
    -> record in the MongoDB database. Returns a preprocessing report and the retrain-trigger decision."""
    label = label.strip().upper()
    if label not in config.CLASSES:
        raise HTTPException(400, f"label must be one of {config.CLASSES}")
    items = [(f.filename, f.file.read()) for f in files]
    accepted, rejected = prep.process_upload(items, db.known_hashes())

    folder = config.UPLOAD_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    saved = []
    for a in accepted:
        path = folder / f"{a['md5'][:16]}.jpg"
        if db.insert_upload(a["filename"], label, path, a["md5"], a["width"], a["height"]):
            path.write_bytes(a["jpeg"])
            saved.append(a)
        else:                                   # another container stored the same image meanwhile
            rejected.append({"filename": a["filename"], "reason": "duplicate of an image already stored"})
    accepted = saved

    trigger = retrain_trigger()
    auto_started = None
    if config.AUTO_RETRAIN and trigger["retrain"]:
        auto_started = _start_retrain("; ".join(trigger["reasons"]))
    return {
        "label": label,
        "preprocessing": {
            "received": len(items),
            "saved_to_database": len(accepted),
            "rejected": len(rejected),
            "steps": ["validated readable JPG/PNG", "removed duplicates (MD5 hash)",
                      "converted to 3-channel RGB", "resized to 224x224", "stored on disk + MongoDB record"],
        },
        "rejected_files": rejected,
        "database": db.upload_summary(),
        "retrain_trigger": trigger,
        "auto_retrain_started_run_id": auto_started,
    }


@app.post("/retrain", tags=["retraining"])
def retrain(epochs: int = Form(None, description="Optional: number of fine-tuning epochs")):
    """Start retraining in the background (the deployed model is the pre-trained starting point)."""
    _require_model()
    if db.count_uploads() == 0:
        raise HTTPException(400, "No uploaded data. Upload labelled images first.")
    trigger = retrain_trigger()
    reason = "; ".join(trigger["reasons"]) or "manual trigger"
    run_id = _start_retrain(reason, epochs)
    if run_id is None:
        raise HTTPException(409, "A retraining run is already in progress.")
    return {"run_id": run_id, "status": "started", "reason": reason}


@app.get("/retrain/status", tags=["retraining"])
def retrain_status():
    """Progress of the current or most recent retraining run."""
    state = db.get_retrain_state()
    return {"running": state["running"], "message": state["message"],
            "latest_run": db.latest_run(), "trigger": retrain_trigger()}


@app.get("/retrain/history", tags=["retraining"])
def retrain_history(limit: int = 20):
    return db.list_runs(limit)


@app.get("/uploads", tags=["retraining"])
def uploads(limit: int = 20):
    """What is stored in the database for retraining."""
    return {"summary": db.upload_summary(), "recent": db.recent_uploads(limit)}


@app.post("/evaluate", tags=["monitoring"])
def evaluate(max_images: int = 400):
    """Evaluate the deployed model on the test folder (evaluation in production)."""
    _require_model()
    if prep.count_images(config.TEST_DIR) == 0:
        raise HTTPException(400, "data/test is empty.")
    model, meta = prediction.store.get()
    metrics = model_lib.evaluate_directory(model, config.TEST_DIR, meta.get("threshold", 0.5), max_images)
    return {"model_version": meta.get("version"), "threshold": meta.get("threshold"), "metrics": metrics}


@app.get("/insights", tags=["monitoring"])
def insights():
    """Data behind the dashboard: class balance per split, brightness, predictions, model metrics."""
    if "brightness" not in _insights_cache:
        source = config.TRAIN_DIR if prep.count_images(config.TRAIN_DIR) else config.TEST_DIR
        _insights_cache["brightness"] = prep.sample_brightness(source)
    meta = prediction.store.get()[1] if prediction.store.loaded else {}
    return {
        "class_counts": {split: prep.class_counts(path) for split, path in
                         (("train", config.TRAIN_DIR), ("val", config.VAL_DIR),
                          ("test", config.TEST_DIR), ("uploads", config.UPLOAD_DIR))},
        "brightness": _insights_cache["brightness"],
        "predictions": db.prediction_summary(),
        "uploads": db.upload_summary(),
        "model": {k: meta.get(k) for k in ("model_name", "version", "threshold", "trained_at",
                                           "n_train_images", "test_metrics")},
    }
"""MongoDB storage for uploaded retraining data, predictions and retraining runs.

Uploaded images are saved on disk (data/uploads/<CLASS>/) because TensorFlow reads folders, and every
file is recorded in the `uploads` collection (name, label, path, hash, size, time, status).

Collections
  uploads       one document per uploaded image (unique MD5 hash)
  predictions   one document per prediction served
  retrain_runs  one document per retraining run
  state         a single shared document holding the retraining lock and progress message, so that
                several API containers see the same status and only one of them retrains at a time
  counters      auto-increment integer ids

Tests set MONGO_URI=mongomock:// to use an in-memory fake instead of a real server.
"""
import datetime

from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

from . import config

_client = None
STALE_LOCK_MINUTES = 30          # a retrain lock with no progress for this long is treated as abandoned


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def _iso(value):
    return value.isoformat(timespec="seconds") if isinstance(value, datetime.datetime) else value


def _db():
    global _client
    if _client is None:
        if config.MONGO_URI.startswith("mongomock"):
            import mongomock
            _client = mongomock.MongoClient()
        else:
            _client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client[config.MONGO_DB]


def _next_id(name):
    doc = _db().counters.find_one_and_update(
        {"_id": name}, {"$inc": {"seq": 1}}, upsert=True, return_document=ReturnDocument.AFTER)
    return doc["seq"]


def _clean(doc):
    """Drop Mongo's internal _id and make datetimes JSON-friendly."""
    if doc is None:
        return None
    return {k: _iso(v) for k, v in doc.items() if k != "_id"}


def ping():
    try:
        _db().command("ping")
        return True
    except Exception:
        return False


def init_db():
    d = _db()
    d.uploads.create_index([("md5", ASCENDING)], unique=True)
    d.uploads.create_index([("status", ASCENDING)])
    d.uploads.create_index([("id", DESCENDING)])
    d.predictions.create_index([("id", DESCENDING)])
    d.retrain_runs.create_index([("id", DESCENDING)])


# ----------------------------------------------------------------------------- uploads
def known_hashes():
    return {d["md5"] for d in _db().uploads.find({}, {"md5": 1})}


def insert_upload(filename, label, path, md5, width, height):
    """Returns True if stored, False if an image with the same hash already exists."""
    try:
        _db().uploads.insert_one({
            "id": _next_id("uploads"), "filename": filename, "label": label, "path": str(path),
            "md5": md5, "width": width, "height": height, "uploaded_at": _now(),
            "status": "new", "used_in_version": None})
        return True
    except DuplicateKeyError:
        return False


def count_uploads(status=None):
    return _db().uploads.count_documents({"status": status} if status else {})


def upload_summary():
    out = {"total": 0, "new": 0, "used": 0, "by_label": {}}
    pipeline = [{"$group": {"_id": {"label": "$label", "status": "$status"}, "c": {"$sum": 1}}}]
    for r in _db().uploads.aggregate(pipeline):
        label, status, c = r["_id"]["label"], r["_id"]["status"], r["c"]
        out["total"] += c
        out[status] = out.get(status, 0) + c
        out["by_label"][label] = out["by_label"].get(label, 0) + c
    return out


def recent_uploads(limit=20):
    cur = _db().uploads.find({}, {"md5": 0, "path": 0}).sort("id", DESCENDING).limit(limit)
    return [_clean(d) for d in cur]


def mark_uploads_used(version):
    _db().uploads.update_many({"status": "new"}, {"$set": {"status": "used", "used_in_version": version}})


# ----------------------------------------------------------------------------- predictions
def insert_prediction(filename, label, probability, confidence, model_version):
    _db().predictions.insert_one({
        "id": _next_id("predictions"), "filename": filename, "predicted_label": label,
        "probability_pneumonia": probability, "confidence": confidence,
        "model_version": model_version, "created_at": _now()})


def prediction_summary():
    d = _db().predictions
    by_label = {r["_id"]: r["c"] for r in d.aggregate([{"$group": {"_id": "$predicted_label", "c": {"$sum": 1}}}])}
    probs = [x["probability_pneumonia"] for x in
             d.find({}, {"probability_pneumonia": 1}).sort("id", DESCENDING).limit(500)]
    recent = [_clean(x) for x in d.find({}, {"id": 0}).sort("id", DESCENDING).limit(10)]
    return {"total": d.count_documents({}), "by_label": by_label,
            "recent_probabilities": probs, "recent": recent}


# ----------------------------------------------------------------------------- retraining runs
def create_run(trigger_reason, n_new_images, old_version):
    run_id = _next_id("retrain_runs")
    _db().retrain_runs.insert_one({
        "id": run_id, "started_at": _now(), "finished_at": None, "status": "running",
        "trigger_reason": trigger_reason, "n_new_images": n_new_images, "old_version": old_version,
        "new_version": None, "promoted": None, "old_val_f1": None, "new_val_f1": None,
        "old_val_recall": None, "new_val_recall": None, "message": ""})
    return run_id


def finish_run(run_id, status, message="", result=None):
    result = result or {}
    old_val, new_val = result.get("old_val", {}), result.get("new_val", {})
    _db().retrain_runs.update_one({"id": run_id}, {"$set": {
        "finished_at": _now(), "status": status, "message": message,
        "new_version": result.get("new_version"), "promoted": result.get("promoted"),
        "old_val_f1": old_val.get("f1"), "new_val_f1": new_val.get("f1"),
        "old_val_recall": old_val.get("recall"), "new_val_recall": new_val.get("recall")}})


def list_runs(limit=20):
    return [_clean(d) for d in _db().retrain_runs.find().sort("id", DESCENDING).limit(limit)]


def latest_run():
    runs = list_runs(1)
    return runs[0] if runs else None


# ----------------------------------------------------------------------------- shared retrain lock/state
def acquire_retrain_lock():
    """Atomically claim the right to retrain. Returns False if another run holds the lock."""
    cutoff = _now() - datetime.timedelta(minutes=STALE_LOCK_MINUTES)
    try:
        doc = _db().state.find_one_and_update(
            {"_id": "retrain", "$or": [{"running": False}, {"heartbeat": {"$lt": cutoff}}]},
            {"$set": {"running": True, "message": "Starting", "run_id": None, "heartbeat": _now()}},
            upsert=True, return_document=ReturnDocument.AFTER)
        return doc is not None
    except DuplicateKeyError:                      # the document exists and is locked
        return False


def set_retrain_state(message=None, run_id=None):
    update = {"heartbeat": _now()}
    if message is not None:
        update["message"] = message
    if run_id is not None:
        update["run_id"] = run_id
    _db().state.update_one({"_id": "retrain"}, {"$set": update})


def release_retrain_lock(message="idle"):
    _db().state.update_one({"_id": "retrain"},
                           {"$set": {"running": False, "message": message, "heartbeat": _now()}})


def get_retrain_state():
    doc = _db().state.find_one({"_id": "retrain"})
    if not doc:
        return {"running": False, "message": "idle", "run_id": None}
    return {"running": bool(doc.get("running")), "message": doc.get("message", ""), "run_id": doc.get("run_id")}

import time

from tests.conftest import make_image_bytes  # noqa: F401  (also sets the test environment)

from src import config, database as db, prediction


def setup_module():
    db.init_db()


def test_unique_hash_rejects_duplicates():
    assert db.insert_upload("a.jpg", "NORMAL", "/x/a.jpg", "hash-1", 100, 100) is True
    assert db.insert_upload("copy.jpg", "NORMAL", "/x/copy.jpg", "hash-1", 100, 100) is False
    assert "hash-1" in db.known_hashes()


def test_upload_summary_and_mark_used():
    db.insert_upload("b.jpg", "PNEUMONIA", "/x/b.jpg", "hash-2", 100, 100)
    s = db.upload_summary()
    assert s["new"] >= 2 and s["by_label"]["PNEUMONIA"] >= 1
    db.mark_uploads_used(5)
    assert db.count_uploads("new") == 0 and db.upload_summary()["used"] >= 2


def test_retrain_lock_is_exclusive():
    assert db.acquire_retrain_lock() is True
    assert db.acquire_retrain_lock() is False          # second container cannot start a second run
    assert db.get_retrain_state()["running"] is True
    db.set_retrain_state(message="Epoch 1/3")
    assert db.get_retrain_state()["message"] == "Epoch 1/3"
    db.release_retrain_lock("done")
    assert db.get_retrain_state()["running"] is False
    assert db.acquire_retrain_lock() is True            # free again
    db.release_retrain_lock()


def test_run_records():
    run_id = db.create_run("manual", 12, 1)
    db.finish_run(run_id, "done", "ok", {"new_version": 2, "promoted": True,
                                         "old_val": {"f1": 0.9, "recall": 0.9}, "new_val": {"f1": 0.92, "recall": 0.93}})
    latest = db.latest_run()
    assert latest["id"] == run_id and latest["promoted"] is True and latest["new_val_f1"] == 0.92


def test_model_store_reloads_when_version_changes():
    import json
    store = prediction.ModelStore()
    store.check_interval = 0
    _, meta = store.get()
    version = meta["version"]
    data = json.loads(config.META_PATH.read_text())
    data["version"] = version + 100                    # simulate another container promoting a model
    config.META_PATH.write_text(json.dumps(data))
    try:
        _, meta2 = store.get()
        assert meta2["version"] == version + 100
    finally:
        data["version"] = version
        config.META_PATH.write_text(json.dumps(data))

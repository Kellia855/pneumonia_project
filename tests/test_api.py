import time

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_image_bytes

from app.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # runs the startup (init DB, load model)
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model_loaded"] and r["model_version"] == 1
    assert r["uptime_seconds"] >= 0


def test_predict_valid_image(client):
    r = client.post("/predict", files={"file": ("x.jpg", make_image_bytes(150), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("NORMAL", "PNEUMONIA")
    assert 0 <= body["probability_pneumonia"] <= 1 and body["model_version"] == 1


def test_predict_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.jpg", b"this is not an image", "image/jpeg")})
    assert r.status_code == 400


def test_upload_preprocessing_report(client):
    good = [("a%d.jpg" % i, make_image_bytes(100 + i)) for i in range(6)]
    files = [("files", (n, b, "image/jpeg")) for n, b in good]
    files.append(("files", ("dup.jpg", good[0][1], "image/jpeg")))        # duplicate of the first
    files.append(("files", ("bad.jpg", b"garbage", "image/jpeg")))        # corrupt
    r = client.post("/upload", data={"label": "normal"}, files=files)
    assert r.status_code == 200
    pre = r.json()["preprocessing"]
    assert pre["received"] == 8 and pre["saved_to_database"] == 6 and pre["rejected"] == 2
    reasons = {x["filename"]: x["reason"] for x in r.json()["rejected_files"]}
    assert "duplicate" in reasons["dup.jpg"] and "corrupt" in reasons["bad.jpg"]
    assert r.json()["database"]["new"] == 6
    assert r.json()["retrain_trigger"]["retrain"] is False


def _colour_photo_bytes():
    """A colourful 'photo' (random RGB noise plus colour blocks), like a picture of a cat."""
    import io
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, (160, 200, 3), dtype=np.uint8)
    arr[:80, :100] = (230, 120, 40)          # orange block
    arr[80:, 100:] = (40, 160, 70)           # green block
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def _xray_with_marker_bytes():
    """A greyscale image saved as RGB with a small red marker in the corner (as some X-rays have)."""
    import io
    import numpy as np
    from PIL import Image
    grey = np.clip(np.random.default_rng(2).normal(120, 30, (200, 200)), 0, 255).astype(np.uint8)
    arr = np.stack([grey] * 3, axis=2)
    arr[5:25, 5:25] = (220, 30, 30)          # 1% of the image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def test_predict_rejects_colour_photo(client):
    r = client.post("/predict", files={"file": ("cat.jpg", _colour_photo_bytes(), "image/jpeg")})
    assert r.status_code == 400
    assert "does not look like a chest X-ray" in r.json()["detail"]


def test_predict_accepts_greyscale_with_small_marker(client):
    r = client.post("/predict", files={"file": ("xray.jpg", _xray_with_marker_bytes(), "image/jpeg")})
    assert r.status_code == 200


def test_upload_rejects_colour_photo(client):
    files = [("files", ("cat.jpg", _colour_photo_bytes(), "image/jpeg"))]
    r = client.post("/upload", data={"label": "NORMAL"}, files=files).json()
    assert r["preprocessing"]["saved_to_database"] == 0
    assert "chest X-ray" in r["rejected_files"][0]["reason"]


def test_upload_rejects_bad_label(client):
    r = client.post("/upload", data={"label": "cat"}, files=[("files", ("a.jpg", make_image_bytes(100), "image/jpeg"))])
    assert r.status_code == 400


def test_trigger_fires_after_enough_uploads(client):
    files = [("files", ("p%d.jpg" % i, make_image_bytes(150 + i), "image/jpeg")) for i in range(6)]
    r = client.post("/upload", data={"label": "PNEUMONIA"}, files=files).json()
    assert r["retrain_trigger"]["retrain"] is True        # 12 new images >= threshold of 10
    assert client.get("/uploads").json()["summary"]["total"] == 12


def test_evaluate(client):
    r = client.post("/evaluate", params={"max_images": 16})
    assert r.status_code == 200
    m = r.json()["metrics"]
    assert m["n_images"] == 16 and 0 <= m["accuracy"] <= 1


def test_insights(client):
    r = client.get("/insights").json()
    assert r["class_counts"]["train"] == {"NORMAL": 12, "PNEUMONIA": 12}
    assert r["uploads"]["total"] == 12 and r["predictions"]["total"] >= 1


def test_retrain_flow(client):
    r = client.post("/retrain", data={"epochs": 1})
    assert r.status_code == 200
    for _ in range(180):
        s = client.get("/retrain/status").json()
        if not s["running"]:
            break
        time.sleep(1)
    assert not s["running"], "retraining did not finish"
    run = s["latest_run"]
    assert run["status"] == "done", run
    assert run["old_val_f1"] is not None and run["new_val_f1"] is not None
    hist = client.get("/retrain/history").json()
    assert len(hist) == 1
    if run["promoted"]:
        assert client.get("/health").json()["model_version"] == 2
        assert client.get("/uploads").json()["summary"]["new"] == 0
    # a second run is possible (models reload cleanly) once the first has finished
    r2 = client.post("/retrain", data={"epochs": 1})
    assert r2.status_code in (200, 409)
    for _ in range(180):                      # wait so the background run does not leak into other tests
        if not client.get("/retrain/status").json()["running"]:
            break
        time.sleep(1)
    assert client.get("/retrain/status").json()["running"] is False
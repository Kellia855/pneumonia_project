"""Flood-request simulation for the pneumonia API.

Each simulated user mostly sends chest X-rays to /predict (the expensive call) and sometimes checks /health.

Web UI:    locust -f locustfile.py --host http://localhost:8000          then open http://localhost:8089
Headless:  locust -f locustfile.py --host http://localhost:8000 --headless -u 50 -r 10 -t 2m --csv results/locust/run
"""
import io
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

IMAGE_DIR = Path(os.getenv("LOCUST_IMAGE_DIR", Path(__file__).parent / "data" / "test"))
EXT = {".jpg", ".jpeg", ".png"}


def load_images(limit=60):
    files = [p for p in IMAGE_DIR.rglob("*") if p.suffix.lower() in EXT] if IMAGE_DIR.exists() else []
    random.Random(42).shuffle(files)
    images = [(p.name, p.read_bytes()) for p in files[:limit]]
    if not images:                     # no sample data: send a synthetic grey image so the test still runs
        from PIL import Image
        buf = io.BytesIO()
        Image.new("L", (224, 224), 120).save(buf, format="JPEG")
        images = [("synthetic.jpg", buf.getvalue())]
    return images


IMAGES = load_images()


class XrayUser(HttpUser):
    wait_time = between(0.5, 1.5)      # think time between requests of one user
    connection_timeout = 10.0
    network_timeout = 60.0             # a request with no answer after 60 s counts as a failure

    @task(10)
    def predict(self):
        name, data = random.choice(IMAGES)
        with self.client.post("/predict", files={"file": (name, data, "image/jpeg")},
                              name="/predict", catch_response=True) as r:
            if r.status_code != 200 or "label" not in r.text:
                r.failure(f"status {r.status_code}")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
# Pneumonia Detection from Chest X-rays

> Educational project. Not a medical device: predictions must never replace a clinician's diagnosis.

## Project description
This project classifies chest X-ray images as **NORMAL** or **PNEUMONIA**, extending my earlier COPD detection project
from tabular survey data to medical images in the same healthcare domain, with the aim of supporting early detection of
respiratory conditions. A MobileNetV2 transfer-learning model (compared against a baseline CNN trained from scratch) is
trained on the Kaggle *Chest X-Ray Images (Pneumonia)* dataset, then served through a **FastAPI** service with a
**Streamlit** dashboard. Users can predict a single X-ray, explore the data, upload new labelled images in bulk (stored in
**MongoDB**) and trigger **retraining**, which fine-tunes the deployed model and only promotes the new version if it is not
worse. The service runs in **Docker**, scales behind an **nginx** load balancer, and was load-tested with **Locust**.

## Links
- **Video demo (YouTube):** TODO: add link
- **Live dashboard:** TODO: add URL
- **Live API docs (Swagger):** TODO: add URL `/docs`
- **Notebook:** [`notebook/pneumonia_pipeline.ipynb`](notebook/pneumonia_pipeline.ipynb)
- **Dataset:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## What the solution does
| Requirement | Where |
|---|---|
| Data acquisition, cleaning, splitting | Notebook sections 2 to 4 (integrity check, MD5 de-duplication, stratified 70/15/15 split, held-out "new data" batch) |
| Data processing | Notebook section 6, `src/preprocessing.py` (RGB, 224×224 bilinear resize, augmentation, class weights) |
| Model creation | Notebook sections 8 and 9, `src/model.py` (baseline CNN vs MobileNetV2, two-stage fine-tuning, early stopping, LR scheduling, dropout) |
| Model testing | Notebook section 10: accuracy, precision, recall, specificity, F1, ROC-AUC, confusion matrices, threshold choice, error analysis |
| Retraining + trigger | `src/model.py::retrain_model`, `/upload`, `/retrain`; trigger at 50 new images or the dashboard button |
| API | `app/api.py` (FastAPI) |
| UI: up-time, visualizations, predict, upload, retrain | `app/ui.py` (Streamlit) |
| Evaluation in production | `/evaluate` endpoint and the dashboard's *Evaluate in production* tab |
| Flood-request simulation | `locustfile.py`, `scripts/load_test.sh`, results below |

## Repository layout
```
pneumonia_project/
├── README.md
├── notebook/pneumonia_pipeline.ipynb   full pipeline: preprocessing, training, evaluation, prediction, retraining
├── src/
│   ├── config.py           settings (all overridable with environment variables)
│   ├── preprocessing.py    validation, RGB + 224x224 resize, de-duplication, datasets, augmentation
│   ├── model.py            build / train / evaluate / retrain (with the promotion safety rule)
│   ├── prediction.py       single-image prediction; reloads the model when another container promotes one
│   └── database.py         MongoDB: uploads, predictions, retraining runs, shared retraining lock
├── app/
│   ├── api.py              FastAPI service
│   └── ui.py               Streamlit dashboard
├── data/                   train/ val/ test/ samples (224x224); uploads/ is created at runtime
├── models/                 pneumonia_model.h5 + model_metadata.json (class order, threshold, version, metrics)
├── reports/figures/        figures exported from the notebook (used by the dashboard)
├── deploy/                 start.sh, healthcheck.py, nginx.conf (load balancer)
├── scripts/                make_repo_data.py, load_test.sh, summarize_locust.py
├── results/locust/         flood-test CSV / HTML reports and summary.md
├── tests/                  API and database tests (synthetic data, in-memory MongoDB)
├── locustfile.py
├── Dockerfile
└── docker-compose.yml
```

## Model results (notebook, held-out test set of 814 images)
| Model | Threshold | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | Missed cases (FN) | False alarms (FP) |
|---|---|---|---|---|---|---|---|---|---|
| Baseline CNN | 0.5 | 0.864 | 0.990 | 0.825 | 0.976 | 0.900 | 0.987 | 106 | 5 |
| MobileNetV2 | 0.5 | 0.952 | 0.978 | 0.957 | 0.937 | 0.968 | 0.985 | 26 | 13 |
| **MobileNetV2 (deployed)** | **0.689** | **0.948** | **0.981** | **0.949** | **0.947** | **0.965** | **0.985** | **31** | **11** |
| MobileNetV2 retrained (v2) | 0.689 | 0.942 | 0.983 | 0.939 | 0.952 | 0.960 | 0.988 | 37 | 10 |

- MobileNetV2 was selected on **validation F1** (0.975 vs 0.894). Both models rank images equally well (AUC ≈ 0.99), but the
  baseline misses far more pneumonia cases at the default threshold.
- The decision threshold (0.689) was chosen on the validation set as the one with the best precision at ≥ 95% recall.
- The dataset has a 3 : 1 class imbalance and strong image-size differences between the classes, so part of the
  performance may come from acquisition differences rather than lung disease (see notebook sections 5 and 10.8).

## Flood-request simulation (Locust)
Each simulated user sends chest X-rays to `POST /predict` (10 of every 11 requests) and checks `GET /health`, waiting
0.5 to 1.5 s between requests. The same test was run against **1, 2 and 3 API containers** behind the nginx load
balancer, each container limited to the same CPU share (so adding containers adds capacity, as separate servers would).

**Test settings:** TODO: number of users, spawn rate, duration, CPUs per container, machine.

TODO: paste `results/locust/summary.md` here (created by `scripts/load_test.sh`).

| Containers | Endpoint | Requests | Failures | Req/s | Median (ms) | Mean (ms) | 95th pct (ms) | 99th pct (ms) | Max (ms) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | /predict | | | | | | | | |
| 2 | /predict | | | | | | | | |
| 3 | /predict | | | | | | | | |

**Interpretation:** TODO: e.g. how median and 95th-percentile latency and throughput change as containers are added.

Screenshots and full HTML reports: `results/locust/`.

## Setup

### 1. Get the code, model and data
```bash
git clone <this-repo-url> && cd pneumonia_project
```
- **Model:** copy `pneumonia_model.h5` and `model_metadata.json` from the notebook's output
  (`/kaggle/working/pneumonia_project/models/`) into `models/`.
- **Data samples:** the full dataset is too large for the repo. In the Kaggle session where the notebook ran, add
  `scripts/make_repo_data.py` (for example with a `%%writefile make_repo_data.py` cell) and run:
  ```bash
  !python make_repo_data.py --manifest /kaggle/working/pneumonia_project/data_manifest.csv \
      --out /kaggle/working/repo_data/data --demo-out /kaggle/working/repo_data/demo_uploads
  ```
  Download `repo_data/` from the Output tab and copy `data/` and `demo_uploads/` into the repo. This creates small
  224×224 samples: `data/train` (replay data for retraining), `data/val` (decides promotion), `data/test`
  (production evaluation and demo predictions) and `demo_uploads/` (images the model has never seen, for the
  upload + retrain demo).

### 2a. Run everything with Docker (recommended)
```bash
docker compose up -d --build              # MongoDB + API + nginx + dashboard
```
- Dashboard: http://localhost:8501
- API (through the load balancer): http://localhost:8000/docs
- Scale the API: `docker compose up -d --scale api=3` (nginx picks up new containers automatically)
- Use MongoDB Atlas instead of the local container: `MONGO_URI="mongodb+srv://..." docker compose up -d`
- Start from a clean state (deletes uploads, retrained models and the database): `docker compose down -v`

### 2b. Run without Docker
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export MONGO_URI="mongodb://localhost:27017"      # or your Atlas connection string
uvicorn app.api:app --port 8000                   # API + Swagger at http://localhost:8000/docs
API_URL=http://localhost:8000 streamlit run app/ui.py
```

### 3. Tests
```bash
pytest -q          # uses synthetic images, a tiny model and an in-memory MongoDB (mongomock)
```

### 4. Flood test
```bash
pip install -r requirements-dev.txt
bash scripts/load_test.sh                         # 1, 2 and 3 containers, 50 users for 2 minutes each
USERS=100 DURATION=3m bash scripts/load_test.sh   # heavier run
locust -f locustfile.py --host http://localhost:8000   # interactive Locust UI at http://localhost:8089
```

## Cloud deployment (AWS EC2)
The whole stack (MongoDB, API containers, nginx load balancer, dashboard) runs with Docker Compose on one
Amazon EC2 instance, exactly as it runs locally.

1. **Launch an EC2 instance:** Ubuntu Server 24.04 LTS, type `m7i-flex.large` (2 vCPU, 8 GB RAM), 30 GB gp3 storage.
   Security group: SSH (22) from your IP, TCP 8000 and 8501 from anywhere.
2. **Connect and install Docker:**
   ```bash
   ssh -i your-key.pem ubuntu@<public-ip>
   curl -fsSL https://get.docker.com | sudo sh
   sudo usermod -aG docker ubuntu     # then log out and back in
   ```
3. **Clone and start:**
   ```bash
   git clone https://github.com/Kellia855/pneumonia_project.git
   cd pneumonia_project
   docker compose up -d --build
   ```
4. **Open:** dashboard at `http://<public-ip>:8501`, API docs at `http://<public-ip>:8000/docs`.
   Scale the API with `docker compose up -d --scale api=2`.

Notes:
- TensorFlow needs memory: instances with 1 GB RAM (t3.micro) are too small for the API and for retraining.
- MongoDB runs in its own container with a persistent Docker volume, so uploads, predictions and retraining
  history survive restarts. `docker compose down -v` resets everything.
- The public IP changes when the instance is stopped and started.

## API
| Endpoint | Purpose |
|---|---|
| `GET /health` | Up-time, model version, database status, which container answered |
| `POST /predict` | Predict one image (label, probability of pneumonia, confidence, threshold, model version) |
| `POST /upload` | Bulk upload of labelled images: validate, de-duplicate, RGB, resize to 224×224, save to disk + MongoDB; returns a preprocessing report and the retrain-trigger decision |
| `POST /retrain` | Start retraining in the background from the deployed model |
| `GET /retrain/status`, `GET /retrain/history` | Progress of the current run and past runs (validation metrics before/after) |
| `GET /uploads` | What is stored in the database for retraining |
| `POST /evaluate` | Evaluate the deployed model on `data/test` (evaluation in production) |
| `GET /insights` | Data behind the dashboard charts |

## Retraining
- **Upload + database:** uploaded images are validated (JPG/PNG, readable, at least 64 px), de-duplicated by MD5 hash,
  converted to RGB, resized to 224×224, saved under `data/uploads/<CLASS>/`, and recorded in the MongoDB `uploads`
  collection (filename, label, path, hash, size, time, status).
- **Trigger:** retraining is recommended when at least 50 new images are waiting (`RETRAIN_MIN_NEW_IMAGES`). The dashboard
  button starts it manually; set `AUTO_RETRAIN=1` to start it automatically when the trigger fires.
- **Process:** the deployed model is loaded as the **pre-trained model** and fine-tuned (learning rate 1e-5, early
  stopping) on the replay data in `data/train` plus all uploads, with class weights recomputed.
- **Safety rule:** the new model replaces the deployed one only if validation F1 drops by at most 1 point and recall by
  at most 2 points. Every version is kept as `models/pneumonia_model_v<N>.h5`.
- **Several containers:** the retraining lock and progress live in MongoDB, so only one container retrains at a time;
  every container checks `model_metadata.json` every 5 seconds and loads a newly promoted model.

## Limitations
- Images come from a single source (paediatric patients); results may not transfer to other hospitals, scanners or ages.
- Binary task only (no bacterial vs viral distinction, no other conditions).
- Only exact duplicates were removed; near-duplicates could remain across splits.

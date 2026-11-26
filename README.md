# Pneumonia Detection from Chest X-rays

> Educational project. Not a medical device: The predictions do not a clinician's diagnosis.

## Project description
This project classifies chest X-ray images as **NORMAL** or **PNEUMONIA**, extending my earlier COPD detection project
from tabular survey data to medical images in the same healthcare domain, with the aim of supporting early detection of
respiratory conditions. A MobileNetV2 transfer-learning model (compared against a baseline CNN trained from scratch) is
trained on the Kaggle *Chest X-Ray Images (Pneumonia)* dataset, then served through a **FastAPI** service with a
**Streamlit** dashboard. Users can predict a single X-ray, explore the data, upload new labelled images in bulk (stored in
**MongoDB**) and trigger **retraining**, which fine-tunes the deployed model and only promotes the new version if it is not
worse. The service runs in **Docker** on **AWS EC2**, scales behind an **nginx** load balancer, and was load-tested with
**Locust**.

## Links
- **Video demo (YouTube):** https://youtu.be/LewqhNjjbrY
- **Live dashboard:** http://13.60.222.13:8501/
- **Live API docs (Swagger):** http://13.60.222.13:8000/docs
- **Notebook:** [`notebook/pneumonia_pipeline.ipynb`](notebook/pneumonia_pipeline.ipynb)
- **Dataset:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## What it does
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
| Cloud deployment | AWS EC2 with Docker Compose (see below) |
| Flood-request simulation | `locustfile.py`, `scripts/load_test.sh`, results below |

## Repository layout
```
pneumonia_project/
├── README.md
├── notebook/pneumonia_pipeline.ipynb   full pipeline: preprocessing, training, evaluation, prediction, retraining
├── src/
│   ├── config.py           settings (all overridable with environment variables)
│   ├── preprocessing.py    validation (incl. X-ray colour check), RGB + 224x224 resize, de-duplication, datasets, augmentation
│   ├── model.py            build / train / evaluate / retrain (with the promotion safety rule)
│   ├── prediction.py       single-image prediction; reloads the model when another container promotes one
│   └── database.py         MongoDB: uploads, predictions, retraining runs, shared retraining lock
├── app/
│   ├── api.py              FastAPI service
│   ├── ui.py               Streamlit dashboard
│   └── assets/icons/       dashboard icons
├── data/                   train/ val/ test/ samples (224x224); uploads/ is created at runtime
├── demo_uploads/           unseen NORMAL / PNEUMONIA images for the upload + retraining demo
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

## Data
The full dataset (5,856 chest X-rays, 5,824 after removing duplicates) comes from
[Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
All preprocessing, training and evaluation in the notebook used the **full dataset** (see the notebook outputs).
Because the full dataset is over 1 GB, the repo contains a sample from the notebook's own splits, resized to 224×224,
for the deployed app:

| Folder | Images | Used for |
|---|---|---|
| `data/train` | 300 per class | replay data mixed with uploads during retraining |
| `data/val` | 100 per class | deciding whether a retrained model is promoted |
| `data/test` | 150 per class | evaluation in production and demo predictions |
| `demo_uploads/` | 30 per class | unseen images for the upload and retraining demo |

To recreate the samples, run `scripts/make_repo_data.py` with the notebook's `data_manifest.csv` (see Setup).

## Model results (notebook, held-out test set of 814 images)
| Model | Threshold | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | Missed cases (FN) | False alarms (FP) |
|---|---|---|---|---|---|---|---|---|---|
| Baseline CNN | 0.5 | 0.864 | 0.990 | 0.825 | 0.976 | 0.900 | 0.987 | 106 | 5 |
| MobileNetV2 | 0.5 | 0.952 | 0.978 | 0.957 | 0.937 | 0.968 | 0.985 | 26 | 13 |
| **MobileNetV2 (deployed, v1)** | **0.689** | **0.948** | **0.981** | **0.949** | **0.947** | **0.965** | **0.985** | **31** | **11** |
| MobileNetV2 retrained in the notebook | 0.689 | 0.942 | 0.983 | 0.939 | 0.952 | 0.960 | 0.988 | 37 | 10 |

- MobileNetV2 was selected on **validation F1** (0.975 vs 0.894). Both models rank images equally well (AUC ≈ 0.99), but the
  baseline misses far more pneumonia cases at the default threshold.
- The decision threshold (0.689) was chosen on the validation set as the one with the best precision at ≥ 95% recall.
- The original model (v1) is deployed rather than the notebook's retrained version: it misses fewer pneumonia cases on
  the test set (31 vs 37), which matters most for screening.
- The dataset has a 3 : 1 class imbalance and strong image-size differences between the classes, so part of the
  performance may come from acquisition differences rather than lung disease (see notebook sections 5 and 10.8).

## Flood-request simulation (Locust)
Each simulated user sends chest X-rays to `POST /predict` (10 of every 11 requests) and checks `GET /health`, waiting
0.5 to 1.5 s between requests. The same test was run against **1 and 2 API containers** behind the nginx load balancer,
each container limited to the same CPU share (so adding containers adds capacity, as separate servers would).

**Test settings:** 20 simulated users (spawn rate 10/s), 2 minutes per run, 1 CPU per API container. Machine: laptop with
an Intel i5-8350U (4 cores) and 8 GB RAM, with Docker limited to 3.8 GB. Three TensorFlow containers did not fit in that
memory, so 1 and 2 containers were compared.

| Containers | Endpoint | Requests | Failures | Req/s | Median (ms) | Mean (ms) | 95th pct (ms) | 99th pct (ms) | Max (ms) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | /predict | 275 | 0.0% | 2.3 | 5900 | 7159 | 18000 | 21000 | 21468 |
| 2 | /predict | 594 | 0.0% | 5.0 | 2600 | 2807 | 4900 | 6200 | 7204 |
| 1 | All requests | 316 | 0.0% | 2.7 | 5600 | 6250 | 17000 | 20000 | 21468 |
| 2 | All requests | 665 | 0.0% | 5.6 | 2600 | 2515 | 4900 | 6000 | 7204 |

**Interpretation:** Every request succeeded in both runs, so the service stayed available under load. With one
container, the API could process about 2.3 predictions per second, but 20 users asked for more than that, so requests
waited in a queue: the median prediction took 5.9 s and the slowest 21 s. Adding a second container behind the load
balancer more than doubled throughput (5.0 predictions per second) and cut the median response time by 56% (to 2.6 s)
and the 95th percentile by 73% (to 4.9 s). Response times also became much more consistent. The health check stayed
fast in both runs (median 25–28 ms), because it does not wait behind predictions. Most of the latency is therefore
queueing for the CPU, not the model itself, and scaling out containers is an effective way to absorb more traffic.
On a machine with more memory, further containers would keep improving these numbers.

Full HTML reports (open in a browser) and CSV results: `results/locust/`.

## Setup

### 1. Get the code
```bash
git clone https://github.com/Kellia855/pneumonia_project.git && cd pneumonia_project
```
The trained model (`models/`) and the data samples (`data/`, `demo_uploads/`) are already in this repo.

To recreate the data samples from the full dataset, download it from Kaggle into a `full_dataset/` folder (ignored by
Git) together with the notebook's `data_manifest.csv`, then run (needs TensorFlow; the Docker image works too):
```bash
python scripts/make_repo_data.py --manifest data_manifest.csv --data-root full_dataset --out data --demo-out demo_uploads
```

### 2a. Run everything with Docker (recommended)
```bash
docker compose up -d --build              # MongoDB + API + nginx + dashboard
```
- Dashboard: http://localhost:8501
- API (through the load balancer): http://localhost:8000/docs
- Scale the API: `docker compose up -d --scale api=2` (nginx picks up new containers automatically)
- Start from a clean state (deletes uploads, retrained models and the database): `docker compose down -v`

### 2b. Run without Docker
Requires **Python 3.10 to 3.13** (TensorFlow 2.21 has no build for Python 3.14) and a running MongoDB.
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export MONGO_URI="mongodb://localhost:27017"
uvicorn app.api:app --port 8000                   # API + Swagger at http://localhost:8000/docs
API_URL=http://localhost:8000 streamlit run app/ui.py
```

### 3. Tests
```bash
pytest -q          # 17 tests: synthetic images, a tiny model and an in-memory MongoDB (mongomock)
```

### 4. Flood test
```bash
pip install locust
CONTAINERS="1 2" bash scripts/load_test.sh        # 20 users for 2 minutes per run, 1 CPU per container
USERS=30 CONTAINERS="1 2 3" bash scripts/load_test.sh   # on a machine with more memory
locust -f locustfile.py --host http://localhost:8000   # interactive Locust UI at http://localhost:8089
```
The script stops the dashboard during the test to free memory, and writes `results/locust/summary.md`.

## Cloud deployment (AWS EC2)
The whole stack (MongoDB, API, nginx load balancer, dashboard) runs with Docker Compose on one Amazon EC2 instance,
exactly as it runs locally.

**Deployed setup:** Ubuntu Server 24.04 LTS on a `c7i-flex.large` instance (2 vCPU, 4 GB RAM, plus 4 GB swap),
30 GB gp3 storage, region Europe (Stockholm).

1. **Launch an EC2 instance:** Ubuntu Server 24.04 LTS, `c7i-flex.large` or larger, 30 GB gp3 storage.
   Security group inbound rules: SSH (22), TCP 8000 (API) and TCP 8501 (dashboard).
2. **Connect** (EC2 Instance Connect in the console, or `ssh -i your-key.pem ubuntu@<public-ip>`) **and install Docker:**
   ```bash
   curl -fsSL https://get.docker.com | sudo sh
   sudo usermod -aG docker ubuntu     # then reconnect
   ```
3. **Add swap** (recommended on 4 GB instances, as a safety net while retraining):
   ```bash
   sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```
4. **Clone and start:**
   ```bash
   git clone https://github.com/Kellia855/pneumonia_project.git
   cd pneumonia_project
   docker compose up -d --build
   ```
5. **Open:** dashboard at `http://<public-ip>:8501`, API docs at `http://<public-ip>:8000/docs`.

Notes:
- TensorFlow needs memory: instances with 1 GB RAM (t3.micro) are too small for the API and for retraining.
- MongoDB runs in its own container with a persistent Docker volume and no public port, so only the app can reach it.
  Uploads, predictions and retraining history survive restarts; `docker compose down -v` resets everything.
- To inspect the database: `docker compose exec mongo mongosh pneumonia`, then for example `db.uploads.countDocuments()`.
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
- **Upload + database:** uploaded images are validated (JPG/PNG, readable, at least 64 px, and greyscale: images where more
  than 10% of the pixels are clearly coloured are rejected as not being chest X-rays), de-duplicated by MD5 hash,
  converted to RGB, resized to 224×224, saved under `data/uploads/<CLASS>/`, and recorded in the MongoDB `uploads`
  collection (filename, label, path, hash, size, time, status).
- **Trigger:** retraining is recommended when at least 50 new images are waiting (`RETRAIN_MIN_NEW_IMAGES`). The dashboard
  button starts it manually; set `AUTO_RETRAIN=1` to start it automatically when the trigger fires.
- **Process:** the deployed model is loaded as the **pre-trained model**; the last 30 layers of MobileNetV2 are unfrozen
  and fine-tuned (learning rate 1e-5, early stopping) on the replay data in `data/train` plus all uploads, with class
  weights recomputed.
- **Safety rule:** the new model replaces the deployed one only if validation F1 drops by at most 1 point and recall by
  at most 2 points. Every version is kept as `models/pneumonia_model_v<N>.h5`.
- **Observed behaviour:** with the 60 demo images, short retraining (3 epochs) was promoted, while 10 epochs produced a
  slightly worse model that the safety rule rejected. Longer fine-tuning on a small set moves the model away from what it
  learned on the full data, and the rule prevents that from reaching production.
- **Several containers:** the retraining lock and progress live in MongoDB, so only one container retrains at a time;
  every container checks `model_metadata.json` every 5 seconds and loads a newly promoted model.

## Limitations
- Images come from a single source (paediatric patients); results may not transfer to other hospitals, scanners or ages.
- Binary task only (no bacterial vs viral distinction, no other conditions).
- Only exact duplicates were removed; near-duplicates could remain across splits.
- The API rejects colour images, since chest X-rays are greyscale, but any greyscale image is still classified as NORMAL
  or PNEUMONIA: the model cannot recognise that an image is not an X-ray.
- Retraining reuses the threshold chosen for v1 and decides promotion on a 200-image validation set, so small metric
  differences between versions are partly noise. Re-tuning the threshold after each retrain would make it more robust.
- The dashboard has no user accounts: anyone with the link can upload images and start retraining (the safety rule
  limits the damage).

  ## Author:
  **Kellia Kamikazi**

## Acknowledgements
Icons by [Flaticon](https://www.flaticon.com).

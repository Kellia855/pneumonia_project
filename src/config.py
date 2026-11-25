"""Central configuration. Every path/setting can be overridden with an environment variable
(used by Docker and by the tests)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")   # "mongomock://" is used by the tests
MONGO_DB = os.getenv("MONGO_DB", "pneumonia")

TRAIN_DIR = DATA_DIR / "train"      # replay data: a sample of the original training images
VAL_DIR = DATA_DIR / "val"          # decides whether a retrained model is promoted
TEST_DIR = DATA_DIR / "test"        # used by the /evaluate endpoint
UPLOAD_DIR = DATA_DIR / "uploads"   # preprocessed images uploaded by users (also recorded in the DB)

MODEL_PATH = MODELS_DIR / "pneumonia_model.h5"
META_PATH = MODELS_DIR / "model_metadata.json"

CLASSES = ["NORMAL", "PNEUMONIA"]   # label 0 = NORMAL, label 1 = PNEUMONIA (same order as the notebook)
IMG_SIZE = (224, 224)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
SEED = 42
VALID_EXT = {".jpg", ".jpeg", ".png"}

MAX_UPLOAD_MB = 10
MIN_IMAGE_SIDE = 64

# "Is this a chest X-ray?" guard: X-rays are greyscale, so an image where more than MAX_COLOUR_FRACTION of
# the pixels are clearly coloured (channel spread > COLOUR_PIXEL_THRESHOLD out of 255) is rejected.
COLOUR_CHECK = os.getenv("COLOUR_CHECK", "1") == "1"
COLOUR_PIXEL_THRESHOLD = 30
MAX_COLOUR_FRACTION = float(os.getenv("MAX_COLOUR_FRACTION", 0.10))

# Retraining trigger and settings
RETRAIN_MIN_NEW_IMAGES = int(os.getenv("RETRAIN_MIN_NEW_IMAGES", 50))
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", 5))
RETRAIN_LR = float(os.getenv("RETRAIN_LR", 1e-5))
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "0") == "1"   # retrain automatically when the trigger fires
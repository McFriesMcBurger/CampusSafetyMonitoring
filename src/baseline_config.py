import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_CONFIG = os.path.join(BASE_DIR, "datasets", "data.yaml")
MODEL_PATH = os.path.join(BASE_DIR, "models", "baseline.pt")

DEVICE = 0 if torch.cuda.is_available() else "cpu"

SEED = 42

# Training defaults
EPOCHS = 50
IMGSZ = 640
BATCH = 8
WORKERS = 2
CONF = 0.25

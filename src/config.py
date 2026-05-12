import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_CONFIG = os.path.join(BASE_DIR, "datasets", "data.yaml")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt") 

DEVICE = 0 if torch.cuda.is_available() else "cpu"

SEED = 42

# =========================
# BASE TRAINING SETTINGS
# =========================
EPOCHS = 50
IMGSZ = 640
BATCH = 8
WORKERS = 4
CONF = 0.25

# =========================
# CATEGORY A
# =========================
WEIGHT_DECAY = 0.001
PATIENCE = 10

# =========================
# CATEGORY B
# =========================
DEGREES = 10
MIXUP = 0.2

# =========================
# CATEGORY C
# =========================
COS_LR = True
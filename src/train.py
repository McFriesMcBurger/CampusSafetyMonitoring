from ultralytics import YOLO
from config import *


def train_model(run_name, extra_args=None):

    if extra_args is None:
        extra_args = {}

    print(f"\n============================")
    print(f"TRAINING: {run_name}")
    print(f"============================")

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        cache=False,
        device=DEVICE,
        seed=SEED,

        project="runs/experiments",
        name=run_name,
        exist_ok=True,

        **extra_args
    )

    best_model_path = f"{results.save_dir}/weights/best.pt"

    print(f"\nBest model saved to:")
    print(best_model_path)

    return best_model_path
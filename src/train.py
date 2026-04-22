from ultralytics import YOLO
from config import *

def main():
    print(f"Using device: {DEVICE}")

    model = YOLO("yolov8n.pt")

    model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        cache=False,
        device=DEVICE,
        name="campus_safety_model"
    )

if __name__ == "__main__":
    main()
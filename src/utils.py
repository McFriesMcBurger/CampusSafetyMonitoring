import os
from ultralytics import YOLO
from config import *

def load_model(path=MODEL_PATH):
    return YOLO(path)

def check_dataset_structure(base_path="datasets"):
    required_paths = [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val"
    ]

    for path in required_paths:
        full_path = os.path.join(base_path, path)
        if not os.path.exists(full_path):
            print(f"[ERROR] Missing: {full_path}")
        else:
            print(f"[OK] {full_path}")


def count_images_labels(base_path="datasets"):
    for split in ["train", "val", "test"]:
        img_path = os.path.join(base_path, "images", split)
        lbl_path = os.path.join(base_path, "labels", split)

        if os.path.exists(img_path):
            num_imgs = len(os.listdir(img_path))
            num_lbls = len(os.listdir(lbl_path)) if os.path.exists(lbl_path) else 0

            print(f"{split.upper()}: {num_imgs} images, {num_lbls} labels")
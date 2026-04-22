from utils import load_model
from config import DATA_CONFIG
import os
import shutil
from datetime import datetime
from ultralytics import YOLO

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join("evaluation", f"eval_{timestamp}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model()

    metrics = model.val(
        data=DATA_CONFIG,
        split="val",
        save_json=True,
        project=OUTPUT_DIR,
        name="val"
    )

    print("\n===== EVALUATION =====")
    print(f"P: {metrics.box.mp:.4f}")
    print(f"R: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    # Save metrics
    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"P: {metrics.box.mp:.4f}\n")
        f.write(f"R: {metrics.box.mr:.4f}\n")
        f.write(f"mAP50: {metrics.box.map50:.4f}\n")
        f.write(f"mAP50-95: {metrics.box.map:.4f}\n")

    # Copy visuals
    val_dir = metrics.save_dir
    for file in [
        "confusion_matrix.png",
        "PR_curve.png",
        "F1_curve.png"
    ]:
        src = os.path.join(val_dir, file)
        if os.path.exists(src):
            shutil.copy(src, OUTPUT_DIR)

if __name__ == "__main__":
    main()
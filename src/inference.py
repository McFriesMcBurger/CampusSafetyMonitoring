from utils import load_model
from config import CONF
import os

def predict(source, save=True):
    model = load_model()

    results = model.predict(
        source=source,
        conf=CONF,
        save=save
    )

    return results


def predict_single_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = load_model("models/final.pt")

    results = model.predict(
        source=image_path,
        conf=0.25,
        save=True
    )

    print("\nDetections:")
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"{r.names[cls_id]} ({conf:.2f})")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/inference.py <image_path>")
    else:
        predict_single_image(sys.argv[1])
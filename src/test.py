from config import *
from utils import load_model

def main():
    model = load_model()

    metrics = model.val(
        data=DATA_CONFIG,
        split="val"
    )

    print("\n===== METRICS =====")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP@0.5:   {metrics.box.map50:.4f}")
    print(f"mAP@0.5:95:{metrics.box.map:.4f}")

if __name__ == "__main__":
    main()